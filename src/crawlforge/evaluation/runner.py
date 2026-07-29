"""Application services for offline corpus ingestion and retrieval evaluation."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol, runtime_checkable

from crawlforge.context_engine import ContextEngine
from crawlforge.crawler import CrawledPage
from crawlforge.evaluation.dataset import dataset_signature, validate_dataset
from crawlforge.evaluation.metrics import (
    query_metric_values,
    summarize_context_quality,
    summarize_latency,
    summarize_metrics,
)
from crawlforge.evaluation.models import (
    QUERY_CATEGORIES,
    CategorySummary,
    CorpusStatistics,
    EvaluationDataset,
    EvaluationQuery,
    EvaluationRun,
    QueryEvaluation,
    RetrievedItem,
)
from crawlforge.evaluation.relevance import match_retrieved_items


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Minimal future-compatible retrieval boundary used by the evaluator."""

    @property
    def name(self) -> str:
        """Return a stable strategy name for reports."""

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[RetrievedItem]:
        """Return ranked items without evaluator-side score sorting."""


@runtime_checkable
class RetrievalWarnings(Protocol):
    """Optional strategy-specific methodological warnings."""

    @property
    def warnings(self) -> Sequence[str]:
        """Return stable warnings for the evaluation report."""


class BM25ContextEngineStrategy:
    """Map public ContextEngine search hits into strategy-neutral items."""

    name = "bm25-fts5"

    @property
    def warnings(self) -> Sequence[str]:
        return (
            "Negative-query no-result accuracy treats any returned lexical "
            "candidate as a false positive because BM25 scores are not calibrated.",
        )

    def __init__(
        self,
        engine: ContextEngine,
        dataset: EvaluationDataset,
    ) -> None:
        self._engine = engine
        self._document_ids = {
            document.url: document.document_id for document in dataset.documents
        }
        self._section_ids = {
            (document.document_id, section.heading_path): section.section_id
            for document in dataset.documents
            for section in document.sections
        }

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[RetrievedItem]:
        """Delegate ranking to ContextEngine and preserve its returned order."""
        hits = await self._engine.search(query, limit=limit)
        items: list[RetrievedItem] = []
        for hit in hits:
            document_id = self._document_ids.get(
                hit.source.canonical_url,
                self._document_ids.get(
                    hit.source.url,
                    hit.source.document_id,
                ),
            )
            items.append(
                RetrievedItem(
                    rank=hit.rank,
                    document_id=document_id,
                    url=hit.source.url,
                    canonical_url=hit.source.canonical_url,
                    title=hit.source.title,
                    section_id=self._section_ids.get(
                        (document_id, hit.chunk.heading_path)
                    ),
                    heading_path=hit.chunk.heading_path,
                    text=hit.chunk.text,
                    score=hit.bm25_score,
                    estimated_tokens=hit.chunk.estimated_tokens,
                    source_estimated_tokens=hit.source.source_estimated_tokens,
                    content_hash=hit.chunk.content_hash,
                )
            )
        return tuple(items)


async def ingest_evaluation_corpus(
    engine: ContextEngine,
    dataset: EvaluationDataset,
    *,
    timer: Callable[[], float] = perf_counter,
) -> CorpusStatistics:
    """Index the complete local corpus through production processing and chunking."""
    validate_dataset(dataset)
    fetched_at = datetime(2026, 1, 1, tzinfo=UTC)
    pages = tuple(
        CrawledPage(
            url=document.url,
            final_url=document.url,
            html=document.content,
            status_code=200,
            content_type="text/html; charset=utf-8",
            fetched_at=fetched_at,
            depth=0,
        )
        for document in dataset.documents
    )
    started_at = timer()
    indexing = await engine.index_pages(pages)
    elapsed_ms = (timer() - started_at) * 1000
    info = await engine.get_index_info()
    return CorpusStatistics(
        document_count=info.document_count,
        section_count=sum(len(document.sections) for document in dataset.documents),
        chunk_count=info.chunk_count,
        source_size_bytes=indexing.source_size_bytes,
        cleaned_size_bytes=indexing.cleaned_size_bytes,
        source_estimated_tokens=indexing.source_estimated_tokens,
        cleaned_estimated_tokens=indexing.cleaned_estimated_tokens,
        indexing_time_ms=elapsed_ms,
    )


class RetrievalEvaluationRunner:
    """Evaluate one prebuilt retrieval strategy against stable local judgments."""

    def __init__(
        self,
        *,
        dataset: EvaluationDataset,
        retriever: RetrievalStrategy,
        corpus_statistics: CorpusStatistics,
        retrieval_configuration: dict[str, object] | None = None,
        chunking_configuration: dict[str, object] | None = None,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] = perf_counter,
    ) -> None:
        self._dataset = dataset
        self._retriever = retriever
        self._corpus_statistics = corpus_statistics
        self._retrieval_configuration = retrieval_configuration or {}
        self._chunking_configuration = chunking_configuration or {}
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timer = timer

    async def run(
        self,
        *,
        limits: Sequence[int] = (1, 3, 5, 10),
        token_budget: int = 3000,
        repeat_latency: int = 3,
    ) -> EvaluationRun:
        """Run warm-index quality, latency, and bounded-context evaluation."""
        validate_dataset(self._dataset)
        normalized_limits = tuple(sorted(set(limits)))
        if not normalized_limits or normalized_limits[0] <= 0:
            raise ValueError("limit values must contain positive integers")
        if token_budget <= 0:
            raise ValueError("token budget must be greater than zero")
        if repeat_latency <= 0:
            raise ValueError("repeat latency must be greater than zero")

        maximum_limit = normalized_limits[-1]
        warmup_queries = self._dataset.queries[: min(3, len(self._dataset.queries))]
        warnings = [
            "Latency values are warm-index measurements for this machine only.",
        ]
        if isinstance(self._retriever, RetrievalWarnings):
            warnings.extend(self._retriever.warnings)
        for query in warmup_queries:
            try:
                await self._retriever.search(query.query, limit=maximum_limit)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                warnings.append(
                    f"Warm-up {query.query_id} failed with {type(error).__name__}."
                )

        results: list[QueryEvaluation] = []
        failures: list[str] = []
        latency_samples: list[float] = []
        for query in self._dataset.queries:
            evaluation, query_warnings = await self._evaluate_query(
                query,
                limits=normalized_limits,
                maximum_limit=maximum_limit,
                token_budget=token_budget,
                repeat_latency=repeat_latency,
            )
            results.append(evaluation)
            warnings.extend(query_warnings)
            latency_samples.extend(evaluation.latency_samples_ms)
            if evaluation.failure is not None:
                failures.append(f"{query.query_id}: {evaluation.failure}")

        aggregate = summarize_metrics(results, normalized_limits)
        context_quality = summarize_context_quality(results)
        category_metrics = tuple(
            CategorySummary(
                category=category,
                metrics=summarize_metrics(
                    [result for result in results if result.category == category],
                    normalized_limits,
                ),
                context_quality=summarize_context_quality(
                    [result for result in results if result.category == category]
                ),
            )
            for category in QUERY_CATEGORIES
            if any(result.category == category for result in results)
        )
        retrieval_configuration = {
            **self._retrieval_configuration,
            "limit_values": list(normalized_limits),
            "token_budget": token_budget,
            "repeat_latency": repeat_latency,
            "warmup_calls": len(warmup_queries),
        }
        return EvaluationRun(
            dataset_name=self._dataset.name,
            dataset_version=self._dataset.version,
            retrieval_strategy=self._retriever.name,
            retrieval_configuration=retrieval_configuration,
            chunking_configuration=dict(self._chunking_configuration),
            timestamp=self._clock().isoformat(),
            corpus_statistics=self._corpus_statistics,
            query_results=tuple(results),
            aggregate_metrics=aggregate,
            category_metrics=category_metrics,
            latency=summarize_latency(
                latency_samples,
                repeat_count=repeat_latency,
                warmup_count=len(warmup_queries),
            ),
            context_quality=context_quality,
            worst_queries=_worst_query_ids(results, maximum_limit),
            failures=tuple(failures),
            warnings=tuple(dict.fromkeys(warnings)),
            dataset_signature=dataset_signature(self._dataset),
        )

    async def _evaluate_query(
        self,
        query: EvaluationQuery,
        *,
        limits: tuple[int, ...],
        maximum_limit: int,
        token_budget: int,
        repeat_latency: int,
    ) -> tuple[QueryEvaluation, tuple[str, ...]]:
        first_ranking: tuple[RetrievedItem, ...] | None = None
        first_signature: tuple[tuple[str, str | None, str], ...] | None = None
        samples: list[float] = []
        warnings: list[str] = []
        try:
            for _ in range(repeat_latency):
                started_at = self._timer()
                retrieved = tuple(
                    await self._retriever.search(
                        query.query,
                        limit=maximum_limit,
                    )
                )
                elapsed_ms = (self._timer() - started_at) * 1000
                _validate_ranking(retrieved, maximum_limit)
                samples.append(elapsed_ms)
                signature = tuple(
                    (
                        item.document_id,
                        item.section_id,
                        item.content_hash,
                    )
                    for item in retrieved
                )
                if first_ranking is None:
                    first_ranking = retrieved
                    first_signature = signature
                elif signature != first_signature:
                    warnings.append(
                        f"{query.query_id}: retrieval order changed across repeats."
                    )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = type(error).__name__
            return (
                _failed_query_evaluation(
                    query,
                    limits=limits,
                    latency_samples=tuple(samples),
                    failure=failure,
                ),
                tuple(warnings),
            )

        ranking = first_ranking or ()
        relevance = match_retrieved_items(
            self._dataset,
            query,
            ranking,
        )
        context = _select_context(
            relevance.items,
            token_budget=token_budget,
        )
        context_relevance = match_retrieved_items(
            self._dataset,
            query,
            context.items,
        )
        positive_judgment_ids = {
            judgment.judgment_id
            for judgment in query.relevant_sources
            if judgment.relevance > 0
        }
        matched_context_ids = {
            item.matched_judgment_id
            for item in context_relevance.items
            if item.matched_judgment_id is not None
        }
        first_relevant_rank = next(
            (item.rank for item in relevance.items if item.relevance_grade > 0),
            None,
        )
        relevant_tokens = sum(
            item.estimated_tokens
            for item in context_relevance.items
            if item.relevance_grade > 0
        )
        relevant_chunk_count = sum(
            item.relevance_grade > 0 for item in context_relevance.items
        )
        irrelevant_tokens = context.returned_tokens - relevant_tokens
        coverage = (
            len(matched_context_ids) / len(positive_judgment_ids)
            if positive_judgment_ids
            else 0.0
        )
        return (
            QueryEvaluation(
                query_id=query.query_id,
                query=query.query,
                category=query.category,
                expected_sources=query.relevant_sources,
                retrieved_items=relevance.items,
                context_item_ranks=tuple(item.rank for item in context_relevance.items),
                context_relevant_chunk_count=relevant_chunk_count,
                first_relevant_rank=first_relevant_rank,
                missed_judgment_ids=tuple(
                    sorted(positive_judgment_ids - relevance.matched_judgment_ids)
                ),
                metrics=query_metric_values(
                    query,
                    relevance.items,
                    limits,
                ),
                candidate_count=len(relevance.items),
                returned_estimated_tokens=context.returned_tokens,
                relevant_estimated_tokens=relevant_tokens,
                irrelevant_estimated_tokens=irrelevant_tokens,
                source_estimated_tokens=context.source_tokens,
                estimated_context_reduction=context.reduction,
                relevant_chunks_per_1000_estimated_tokens=(
                    relevant_chunk_count * 1000 / context.returned_tokens
                    if context.returned_tokens
                    else 0.0
                ),
                irrelevant_estimated_token_ratio=(
                    irrelevant_tokens / context.returned_tokens
                    if context.returned_tokens
                    else 0.0
                ),
                relevant_source_coverage=coverage,
                latency_samples_ms=tuple(samples),
            ),
            tuple(warnings),
        )


class _ContextSelection:
    def __init__(
        self,
        items: tuple[RetrievedItem, ...],
        returned_tokens: int,
        source_tokens: int,
    ) -> None:
        self.items = items
        self.returned_tokens = returned_tokens
        self.source_tokens = source_tokens
        self.reduction = (
            max(0.0, min(1.0, 1 - returned_tokens / source_tokens))
            if source_tokens
            else 0.0
        )


def _select_context(
    retrieved: Sequence[RetrievedItem],
    *,
    token_budget: int,
) -> _ContextSelection:
    deduplicated: list[RetrievedItem] = []
    seen_hashes: set[str] = set()
    for item in retrieved:
        if item.content_hash in seen_hashes:
            continue
        seen_hashes.add(item.content_hash)
        deduplicated.append(item)

    selected: list[RetrievedItem] = []
    returned_tokens = 0
    for item in deduplicated:
        next_tokens = returned_tokens + item.estimated_tokens
        if next_tokens > token_budget:
            continue
        selected.append(item)
        returned_tokens = next_tokens

    source_tokens = sum(
        item.source_estimated_tokens for item in _unique_documents(deduplicated)
    )
    return _ContextSelection(
        tuple(selected),
        returned_tokens,
        source_tokens,
    )


def _unique_documents(
    items: Sequence[RetrievedItem],
) -> tuple[RetrievedItem, ...]:
    unique: list[RetrievedItem] = []
    seen: set[str] = set()
    for item in items:
        if item.document_id in seen:
            continue
        seen.add(item.document_id)
        unique.append(item)
    return tuple(unique)


def _validate_ranking(
    retrieved: Sequence[RetrievedItem],
    maximum_limit: int,
) -> None:
    if len(retrieved) > maximum_limit:
        raise ValueError("retrieval strategy returned more items than requested")
    expected_ranks = tuple(range(1, len(retrieved) + 1))
    actual_ranks = tuple(item.rank for item in retrieved)
    if actual_ranks != expected_ranks:
        raise ValueError("retrieval strategy returned non-contiguous ranks")
    for item in retrieved:
        if not item.document_id.strip():
            raise ValueError("retrieval strategy returned an empty document_id")
        if not item.url.strip() or not item.canonical_url.strip():
            raise ValueError("retrieval strategy returned an empty source URL")
        if not item.content_hash.strip():
            raise ValueError("retrieval strategy returned an empty content_hash")
        if not item.text.strip():
            raise ValueError("retrieval strategy returned empty text")
        if not math.isfinite(item.score):
            raise ValueError("retrieval strategy returned a non-finite score")
        if item.estimated_tokens <= 0:
            raise ValueError(
                "retrieval strategy returned non-positive estimated_tokens"
            )
        if item.source_estimated_tokens <= 0:
            raise ValueError(
                "retrieval strategy returned non-positive source_estimated_tokens"
            )


def _failed_query_evaluation(
    query: EvaluationQuery,
    *,
    limits: tuple[int, ...],
    latency_samples: tuple[float, ...],
    failure: str,
) -> QueryEvaluation:
    return QueryEvaluation(
        query_id=query.query_id,
        query=query.query,
        category=query.category,
        expected_sources=query.relevant_sources,
        retrieved_items=(),
        context_item_ranks=(),
        context_relevant_chunk_count=0,
        first_relevant_rank=None,
        missed_judgment_ids=tuple(
            judgment.judgment_id
            for judgment in query.relevant_sources
            if judgment.relevance > 0
        ),
        metrics=query_metric_values(query, (), limits),
        candidate_count=0,
        returned_estimated_tokens=0,
        relevant_estimated_tokens=0,
        irrelevant_estimated_tokens=0,
        source_estimated_tokens=0,
        estimated_context_reduction=0.0,
        relevant_chunks_per_1000_estimated_tokens=0.0,
        irrelevant_estimated_token_ratio=0.0,
        relevant_source_coverage=0.0,
        latency_samples_ms=latency_samples,
        failure=failure,
    )


def _worst_query_ids(
    results: Sequence[QueryEvaluation],
    maximum_limit: int,
) -> tuple[str, ...]:
    def score(result: QueryEvaluation) -> tuple[float, str]:
        if result.failure is not None:
            return (10_000.0, result.query_id)
        if result.category == "negative":
            return (
                2_000.0 + result.candidate_count * 10
                if result.candidate_count
                else 0.0,
                result.query_id,
            )
        missed_at_limit = (
            1.0 if result.metrics.hit_rate_at.get(maximum_limit, 0.0) == 0.0 else 0.0
        )
        rank_penalty = float((result.first_relevant_rank or maximum_limit + 1) - 1)
        return (
            missed_at_limit * 1_000
            + len(result.missed_judgment_ids) * 100
            + rank_penalty * 10
            + result.irrelevant_estimated_token_ratio * 50,
            result.query_id,
        )

    ordered = sorted(results, key=score, reverse=True)
    return tuple(result.query_id for result in ordered[:10])
