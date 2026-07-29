"""Application service for crawl, indexing, lexical retrieval, and context building."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import TracebackType

from crawlforge.chunking import TextChunker
from crawlforge.content import ContentProcessor
from crawlforge.context_index import SQLiteContextIndex
from crawlforge.context_models import (
    ContextResult,
    HeuristicTokenEstimator,
    IndexInfo,
    IndexingResult,
    SearchHit,
    SourceDocument,
    TextChunk,
    TokenEstimator,
)
from crawlforge.crawler import AsyncCrawler, CrawledPage
from crawlforge.network_policy import URLNetworkPolicy

IndexablePage = SourceDocument | CrawledPage


class EmptyCrawlError(RuntimeError):
    """Raised when a bounded crawl produced no indexable documents."""


class ContextEngine:
    """Coordinate deterministic processing, local indexing, and retrieval."""

    def __init__(
        self,
        database: str | Path,
        *,
        processor: ContentProcessor | None = None,
        chunker: TextChunker | None = None,
        token_estimator: TokenEstimator | None = None,
        index: SQLiteContextIndex | None = None,
        network_policy: URLNetworkPolicy | None = None,
    ) -> None:
        """Configure the shared application service and its local index."""
        estimator = token_estimator or HeuristicTokenEstimator()
        self.processor = processor or ContentProcessor(estimator=estimator)
        self.chunker = chunker or TextChunker(estimator=estimator)
        self.index = index or SQLiteContextIndex(database)
        self.network_policy = network_policy
        self._closed = False

    async def __aenter__(self) -> ContextEngine:
        """Initialize the index and enter the engine context."""
        await self._ensure_ready()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the owned local index."""
        await self.close()

    async def index_pages(self, pages: Iterable[IndexablePage]) -> IndexingResult:
        """Process and index a finite batch of fetched or cleaned pages."""
        await self._ensure_ready()
        session_id = await self.index.start_session()
        aggregate = _empty_indexing_result(session_id)
        indexed: list[tuple[SourceDocument, Sequence[TextChunk]]] = []
        try:
            for page in pages:
                document, cleaning_time_ms = await self._prepare_page(page)
                indexed.append((document, self.chunker.chunk(document)))
                aggregate = replace(
                    aggregate,
                    cleaning_time_ms=aggregate.cleaning_time_ms + cleaning_time_ms,
                )
            if indexed:
                delta = await self.index.index_documents(
                    indexed,
                    session_id=session_id,
                )
                aggregate = _add_indexing_results(aggregate, delta)
            await self.index.finish_session(aggregate)
        except BaseException:
            await self._finish_failed_session(aggregate)
            raise
        return aggregate

    async def ingest_url(
        self,
        url: str,
        *,
        max_pages: int = 100,
        max_depth: int = 2,
        max_concurrent: int = 10,
        requests_per_second: float = 1.0,
        total_timeout: float | None = None,
        timeout_backoff_factor: float = 1.5,
        max_response_bytes: int | None = None,
        robots_max_response_bytes: int | None = None,
        respect_robots: bool = True,
        same_domain_only: bool = True,
        fail_on_empty: bool = False,
    ) -> IndexingResult:
        """Crawl one local site boundary and stream successful pages into the index."""
        await self._ensure_ready()
        session_id = await self.index.start_session()
        aggregate = _empty_indexing_result(session_id)
        processing_errors: list[Exception] = []

        async def index_page(page: CrawledPage) -> None:
            nonlocal aggregate
            try:
                document, cleaning_time_ms = await self._prepare_page(page)
                delta = await self.index.index_documents(
                    [(document, self.chunker.chunk(document))],
                    session_id=session_id,
                )
            except Exception as error:
                # This boundary records the exact downstream failure because the
                # crawler deliberately isolates individual page-processing errors.
                processing_errors.append(error)
                raise
            aggregate = _add_indexing_results(
                replace(
                    aggregate,
                    cleaning_time_ms=aggregate.cleaning_time_ms + cleaning_time_ms,
                ),
                delta,
            )

        crawler = AsyncCrawler(
            max_concurrent=max_concurrent,
            max_depth=max_depth,
            requests_per_second=requests_per_second,
            total_timeout=total_timeout,
            timeout_backoff_factor=timeout_backoff_factor,
            max_response_bytes=max_response_bytes,
            robots_max_response_bytes=robots_max_response_bytes,
            respect_robots=respect_robots,
            network_policy=self.network_policy,
            page_handler=index_page,
        )
        try:
            await crawler.crawl(
                [url],
                max_pages=max_pages,
                same_domain_only=same_domain_only,
            )
            failure_messages = tuple(crawler.failed_urls.values())
            aggregate = replace(
                aggregate,
                failed_pages=len(failure_messages),
                failure_categories=_failure_categories(failure_messages),
            )
            if processing_errors:
                raise processing_errors[0]
            if fail_on_empty and aggregate.documents_seen == 0:
                raise EmptyCrawlError(
                    _empty_crawl_message(crawler.failed_urls.values())
                )
            await self.index.finish_session(aggregate)
        except BaseException:
            await self._finish_failed_session(aggregate)
            raise
        finally:
            await crawler.close()
        return aggregate

    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Return BM25-ranked lexical matches from the local index."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        await self._ensure_ready()
        return await self.index.search(query, limit=limit)

    async def build_context(
        self,
        query: str,
        *,
        limit: int = 10,
        token_budget: int = 3000,
    ) -> ContextResult:
        """Select complete ranked chunks without exceeding the estimated budget."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if token_budget <= 0:
            raise ValueError("token_budget must be greater than zero")

        started_at = perf_counter()
        candidates = await self.search(query, limit=limit)
        search_time_ms = (perf_counter() - started_at) * 1000

        deduplicated: list[SearchHit] = []
        seen_hashes: set[str] = set()
        for hit in candidates:
            if hit.chunk.content_hash in seen_hashes:
                continue
            seen_hashes.add(hit.chunk.content_hash)
            deduplicated.append(hit)

        selected: list[SearchHit] = []
        estimated_tokens = 0
        total_size_chars = 0
        for hit in deduplicated:
            next_tokens = estimated_tokens + hit.chunk.estimated_tokens
            if next_tokens > token_budget:
                continue
            selected.append(hit)
            estimated_tokens = next_tokens
            total_size_chars += hit.chunk.size_chars

        source_estimated_tokens = sum(
            hit.source.source_estimated_tokens for hit in _unique_sources(deduplicated)
        )
        reduction = (
            max(0.0, min(1.0, 1 - estimated_tokens / source_estimated_tokens))
            if source_estimated_tokens
            else 0.0
        )
        return ContextResult(
            query=query,
            hits=tuple(selected),
            total_size_chars=total_size_chars,
            estimated_tokens=estimated_tokens,
            candidates_considered=len(candidates),
            search_time_ms=search_time_ms,
            limit=limit,
            token_budget=token_budget,
            source_estimated_tokens=source_estimated_tokens,
            estimated_context_reduction=reduction,
            index_hit=bool(candidates),
        )

    async def get_index_info(self) -> IndexInfo:
        """Return a bounded readiness, size, and latest-session summary."""
        await self._ensure_ready()
        return await self.index.get_index_info()

    async def close(self) -> None:
        """Close the local index; repeated calls are safe."""
        if self._closed:
            return
        await self.index.close()
        self._closed = True

    async def _ensure_ready(self) -> None:
        if self._closed:
            raise RuntimeError("ContextEngine is closed")
        await self.index.initialize()

    async def _prepare_page(
        self,
        page: IndexablePage,
    ) -> tuple[SourceDocument, float]:
        if isinstance(page, SourceDocument):
            return page, 0.0
        if not isinstance(page, CrawledPage):
            raise TypeError("pages must contain SourceDocument or CrawledPage values")
        started_at = perf_counter()
        document = await self.processor.process_html(
            page.html,
            url=page.url,
            final_url=page.final_url,
            status_code=page.status_code,
            content_type=page.content_type,
            fetched_at=page.fetched_at,
        )
        return document, (perf_counter() - started_at) * 1000

    async def _finish_failed_session(self, result: IndexingResult) -> None:
        finish_task = asyncio.create_task(self.index.finish_session(result))
        try:
            await asyncio.shield(finish_task)
        except asyncio.CancelledError as cancelled:
            try:
                await finish_task
            except Exception as finish_error:
                raise cancelled from finish_error
            raise


def _empty_indexing_result(session_id: str) -> IndexingResult:
    return IndexingResult(
        session_id=session_id,
        documents_seen=0,
        documents_indexed=0,
        duplicate_documents=0,
        chunks_indexed=0,
        duplicate_chunks=0,
        source_size_bytes=0,
        cleaned_size_bytes=0,
        source_estimated_tokens=0,
        cleaned_estimated_tokens=0,
        cleaning_time_ms=0.0,
        indexing_time_ms=0.0,
    )


def _add_indexing_results(
    total: IndexingResult,
    delta: IndexingResult,
) -> IndexingResult:
    if total.session_id != delta.session_id:
        raise ValueError("cannot combine different indexing sessions")
    return IndexingResult(
        session_id=total.session_id,
        documents_seen=total.documents_seen + delta.documents_seen,
        documents_indexed=total.documents_indexed + delta.documents_indexed,
        duplicate_documents=(total.duplicate_documents + delta.duplicate_documents),
        chunks_indexed=total.chunks_indexed + delta.chunks_indexed,
        duplicate_chunks=total.duplicate_chunks + delta.duplicate_chunks,
        source_size_bytes=total.source_size_bytes + delta.source_size_bytes,
        cleaned_size_bytes=total.cleaned_size_bytes + delta.cleaned_size_bytes,
        source_estimated_tokens=(
            total.source_estimated_tokens + delta.source_estimated_tokens
        ),
        cleaned_estimated_tokens=(
            total.cleaned_estimated_tokens + delta.cleaned_estimated_tokens
        ),
        cleaning_time_ms=total.cleaning_time_ms + delta.cleaning_time_ms,
        indexing_time_ms=total.indexing_time_ms + delta.indexing_time_ms,
        failed_pages=total.failed_pages + delta.failed_pages,
        failure_categories=tuple(
            sorted({*total.failure_categories, *delta.failure_categories})
        ),
    )


def _unique_sources(hits: Sequence[SearchHit]) -> list[SearchHit]:
    unique: list[SearchHit] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.source.document_id in seen:
            continue
        seen.add(hit.source.document_id)
        unique.append(hit)
    return unique


def _empty_crawl_message(errors: Iterable[str]) -> str:
    messages = tuple(errors)
    if any("URLPolicyError" in message for message in messages):
        return "crawl was blocked by the configured URL network policy"
    if any("Blocked by robots.txt" in message for message in messages):
        return "site access is blocked by robots.txt"
    if any("response exceeds" in message for message in messages):
        return "crawl response exceeded the server byte limit"
    return (
        "crawl returned no indexable pages; verify the URL, robots.txt, "
        "and site availability"
    )


def _failure_categories(errors: Iterable[str]) -> tuple[str, ...]:
    categories: set[str] = set()
    for message in errors:
        normalized = message.casefold()
        if "urlpolicyerror" in normalized:
            categories.add("network_policy")
        elif "blocked by robots.txt" in normalized:
            categories.add("robots")
        elif "timed out" in normalized or "timeout" in normalized:
            categories.add("timeout")
        elif "response exceeds" in normalized:
            categories.add("response_limit")
        elif "http " in normalized:
            categories.add("http_error")
        elif "networkerror" in normalized:
            categories.add("network_error")
        elif "parseerror" in normalized:
            categories.add("parse_error")
        else:
            categories.add("request_error")
    return tuple(sorted(categories))
