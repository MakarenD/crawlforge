"""Asynchronous HTTP client and bounded website crawling orchestration."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from types import TracebackType
from typing import TypedDict
from urllib.parse import urldefrag, urlsplit

import aiohttp

from crawlforge.concurrency import SemaphoreManager
from crawlforge.parser import HTMLParser, ParsedPage
from crawlforge.queue import CrawlerQueue
from crawlforge.urls import canonical_hostname

logger = logging.getLogger(__name__)


class CrawlStats(TypedDict):
    """Snapshot of crawl progress and throughput."""

    processed: int
    queued: int
    active: int
    failed: int
    visited: int
    pages_per_second: float


@dataclass(frozen=True, slots=True)
class _CrawlOutcome:
    url: str
    depth: int
    page: ParsedPage | None
    error: str | None


class AsyncCrawler:
    """Download pages and coordinate a bounded, filtered website crawl."""

    def __init__(
        self,
        max_concurrent: int = 10,
        *,
        max_concurrent_per_domain: int | None = None,
        max_depth: int = 2,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
    ) -> None:
        """Configure crawl depth, concurrency, and request timeouts."""
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")
        if max_depth < 0:
            raise ValueError("max_depth must be zero or greater")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than zero")
        if read_timeout <= 0:
            raise ValueError("read_timeout must be greater than zero")

        self._max_concurrent = max_concurrent
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            connect=connect_timeout,
            sock_read=read_timeout,
        )
        self._max_depth = max_depth
        self._semaphores = SemaphoreManager(
            max_concurrent,
            max_concurrent_per_domain,
        )
        self._session_lock = asyncio.Lock()
        self._crawl_lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._parser = HTMLParser()
        self._closed = False
        self._crawl_started_at: float | None = None

        self.queue = CrawlerQueue()
        self.visited_urls: set[str] = set()
        self.failed_urls: dict[str, str] = {}
        self.processed_urls: dict[str, ParsedPage] = {}
        self._url_depths: dict[str, int] = {}
        self._discovered_urls: set[str] = set()

    async def __aenter__(self) -> AsyncCrawler:
        """Enter the crawler's asynchronous resource context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the pooled HTTP session when leaving the context."""
        await self.close()

    async def fetch_url(self, url: str) -> str:
        """Download one URL, returning an empty string when a request fails."""
        content, _error, _final_url = await self._fetch_url_with_error(url)
        return content

    async def fetch_urls(self, urls: list[str]) -> dict[str, str]:
        """Download URLs concurrently and map each URL to its response body."""
        requested_urls = list(urls)
        pages = await asyncio.gather(
            *(self.fetch_url(url) for url in requested_urls),
        )
        return dict(zip(requested_urls, pages, strict=True))

    async def fetch_and_parse(self, url: str) -> ParsedPage:
        """Download one URL and return its structured HTML content."""
        html, _error, final_url = await self._fetch_url_with_error(url)
        page = await self._parser.parse_html(html, final_url)
        page["url"] = url
        return page

    async def crawl(
        self,
        start_urls: list[str],
        max_pages: int = 100,
        *,
        same_domain_only: bool = False,
        exclude_patterns: Sequence[str] | None = None,
        include_patterns: Sequence[str] | None = None,
    ) -> dict[str, ParsedPage]:
        """Crawl reachable pages within depth, URL, and concurrency limits."""
        if max_pages <= 0:
            raise ValueError("max_pages must be greater than zero")
        if self._closed:
            raise RuntimeError("AsyncCrawler is closed")

        normalized_starts = self._normalize_start_urls(start_urls)
        excludes = self._compile_patterns(exclude_patterns, "exclude_patterns")
        includes = self._compile_patterns(include_patterns, "include_patterns")
        allowed_domains = {self._domain_key(url) for url in normalized_starts}

        async with self._crawl_lock:
            self._reset_crawl_state()
            for url in normalized_starts:
                self._enqueue(url, depth=0, priority=0)

            await self._crawl_queued_urls(
                max_pages=max_pages,
                allowed_domains=allowed_domains,
                same_domain_only=same_domain_only,
                excludes=excludes,
                includes=includes,
            )

            return dict(self.processed_urls)

    def get_stats(self) -> CrawlStats:
        """Return queue, result, error, visit, and throughput statistics."""
        queue_stats = self.queue.get_stats()
        elapsed = (
            perf_counter() - self._crawl_started_at
            if self._crawl_started_at is not None
            else 0.0
        )
        completed = len(self.processed_urls) + len(self.failed_urls)
        return {
            "processed": len(self.processed_urls),
            "queued": queue_stats["queued"],
            "active": self._semaphores.active_tasks,
            "failed": len(self.failed_urls),
            "visited": len(self.visited_urls),
            "pages_per_second": completed / elapsed if elapsed > 0 else 0.0,
        }

    async def _fetch_url_with_error(
        self,
        url: str,
    ) -> tuple[str, str | None, str]:
        logger.info("Fetching URL: %s", url)

        try:
            async with self._semaphores.limit(url):
                session = await self._get_session()
                async with session.get(url) as response:
                    response.raise_for_status()
                    content = await response.text()
                    status = response.status
                    final_url = str(response.url)
        except aiohttp.ClientResponseError as error:
            logger.warning(
                "HTTP error for %s: %s (%s)",
                url,
                error.status,
                type(error).__name__,
            )
            return "", f"HTTP {error.status}: {error.message}", url
        except TimeoutError as error:
            logger.warning("Timeout for %s (%s)", url, type(error).__name__)
            return "", f"{type(error).__name__}: request timed out", url
        except aiohttp.ClientError as error:
            logger.warning("Network error for %s (%s)", url, type(error).__name__)
            return "", f"{type(error).__name__}: {error}", url

        logger.info("Fetched URL: %s (HTTP %s)", url, status)
        return content, None, final_url

    async def close(self) -> None:
        """Close the pooled HTTP session; repeated calls are safe."""
        async with self._session_lock:
            self._closed = True
            session = self._session
            if session is None:
                return
            if session.closed:
                self._session = None
                return

            close_task = asyncio.create_task(session.close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                await close_task
                raise
            finally:
                self._discard_session_if_closed(session)

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._closed:
                raise RuntimeError("AsyncCrawler is closed")
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(limit=self._max_concurrent)
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=self._timeout,
                )
            return self._session

    def _discard_session_if_closed(self, session: aiohttp.ClientSession) -> None:
        if session.closed:
            self._session = None

    def _reset_crawl_state(self) -> None:
        self.queue = CrawlerQueue()
        self.visited_urls.clear()
        self.failed_urls.clear()
        self.processed_urls.clear()
        self._url_depths.clear()
        self._discovered_urls.clear()
        self._crawl_started_at = perf_counter()

    def _enqueue(self, url: str, *, depth: int, priority: int) -> None:
        if url in self._discovered_urls:
            return
        self._discovered_urls.add(url)
        self._url_depths[url] = depth
        self.queue.add_url(url, priority)

    async def _next_url(
        self,
        active_by_domain: dict[str, int],
    ) -> tuple[str, int, str] | None:
        deferred: list[tuple[str, int]] = []
        queued = self.queue.get_stats()["queued"]
        for _ in range(queued):
            url = await self.queue.get_next()
            if url is None:
                break
            depth = self._url_depths[url]
            domain = self._domain_key(url)
            if (
                active_by_domain.get(domain, 0)
                >= self._semaphores.max_concurrent_per_domain
            ):
                deferred.append((url, depth))
                continue
            self.visited_urls.add(url)
            for deferred_url, deferred_depth in deferred:
                self.queue.defer_url(deferred_url, priority=-deferred_depth)
            return url, depth, domain
        for url, depth in deferred:
            self.queue.defer_url(url, priority=-depth)
        return None

    async def _crawl_queued_urls(
        self,
        *,
        max_pages: int,
        allowed_domains: set[str],
        same_domain_only: bool,
        excludes: tuple[re.Pattern[str], ...],
        includes: tuple[re.Pattern[str], ...],
    ) -> None:
        tasks: dict[
            asyncio.Task[_CrawlOutcome],
            tuple[str, int, str],
        ] = {}
        active_by_domain: dict[str, int] = {}
        try:
            while tasks or len(self.visited_urls) < max_pages:
                while (
                    len(tasks) < self._max_concurrent
                    and len(self.visited_urls) < max_pages
                ):
                    target = await self._next_url(active_by_domain)
                    if target is None:
                        break
                    url, depth, domain = target
                    task = asyncio.create_task(self._crawl_page(url, depth))
                    tasks[task] = target
                    active_by_domain[domain] = active_by_domain.get(domain, 0) + 1

                if not tasks:
                    break

                completed, _pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in completed:
                    url, _depth, domain = tasks[task]
                    outcome = task.result()
                    self._record_outcome(
                        outcome,
                        allowed_domains=allowed_domains,
                        same_domain_only=same_domain_only,
                        excludes=excludes,
                        includes=includes,
                    )
                    del tasks[task]
                    remaining = active_by_domain[domain] - 1
                    if remaining:
                        active_by_domain[domain] = remaining
                    else:
                        del active_by_domain[domain]
                    self._log_progress()
        except BaseException as error:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            failure = f"{type(error).__name__}: crawl interrupted"
            for url, _depth, _domain in tasks.values():
                if url not in self.processed_urls and url not in self.failed_urls:
                    self.failed_urls[url] = failure
                    self.queue.mark_failed(url, failure)
            raise

    async def _crawl_page(self, url: str, depth: int) -> _CrawlOutcome:
        try:
            html, error, final_url = await self._fetch_url_with_error(url)
            if error is not None:
                return _CrawlOutcome(url, depth, None, error)
            page = await self._parser.parse_html(html, final_url)
            page["url"] = url
            return _CrawlOutcome(url, depth, page, None)
        except Exception as error:
            logger.warning(
                "Page processing error for %s: %s (%s)",
                url,
                error,
                type(error).__name__,
            )
            return _CrawlOutcome(
                url,
                depth,
                None,
                f"{type(error).__name__}: {error}",
            )

    def _record_outcome(
        self,
        outcome: _CrawlOutcome,
        *,
        allowed_domains: set[str],
        same_domain_only: bool,
        excludes: tuple[re.Pattern[str], ...],
        includes: tuple[re.Pattern[str], ...],
    ) -> None:
        if outcome.error is not None or outcome.page is None:
            error = outcome.error or "Unknown crawl failure"
            self.failed_urls[outcome.url] = error
            self.queue.mark_failed(outcome.url, error)
            return

        self.processed_urls[outcome.url] = outcome.page
        self.queue.mark_processed(outcome.url)
        if outcome.depth >= self._max_depth:
            return

        next_depth = outcome.depth + 1
        for link in outcome.page["links"]:
            if self._should_enqueue(
                link,
                allowed_domains=allowed_domains,
                same_domain_only=same_domain_only,
                excludes=excludes,
                includes=includes,
            ):
                self._enqueue(link, depth=next_depth, priority=-next_depth)

    def _should_enqueue(
        self,
        url: str,
        *,
        allowed_domains: set[str],
        same_domain_only: bool,
        excludes: tuple[re.Pattern[str], ...],
        includes: tuple[re.Pattern[str], ...],
    ) -> bool:
        if url in self._discovered_urls:
            return False
        domain = canonical_hostname(url)
        if domain is None:
            return False
        if same_domain_only and domain not in allowed_domains:
            return False
        if excludes and any(pattern.search(url) for pattern in excludes):
            return False
        return not includes or any(pattern.search(url) for pattern in includes)

    def _normalize_start_urls(self, start_urls: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for url in start_urls:
            candidate, _fragment = urldefrag(url.strip())
            try:
                parsed = urlsplit(candidate)
                valid = (
                    parsed.scheme.casefold() in {"http", "https"}
                    and canonical_hostname(candidate) is not None
                    and not any(character.isspace() for character in candidate)
                    and "\\" not in parsed.netloc
                )
                _port = parsed.port
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"invalid start URL: {url!r}")
            if candidate not in seen:
                seen.add(candidate)
                normalized.append(candidate)
        if not normalized:
            raise ValueError("start_urls must contain at least one URL")
        return normalized

    def _compile_patterns(
        self,
        patterns: Sequence[str] | None,
        name: str,
    ) -> tuple[re.Pattern[str], ...]:
        try:
            return tuple(re.compile(pattern) for pattern in patterns or ())
        except re.error as error:
            raise ValueError(f"invalid regex in {name}: {error}") from error

    def _domain_key(self, url: str) -> str:
        return canonical_hostname(url) or url

    def _log_progress(self) -> None:
        stats = self.get_stats()
        logger.info(
            "Crawl progress: processed=%d queued=%d errors=%d rate=%.2f pages/s",
            stats["processed"],
            stats["queued"],
            stats["failed"],
            stats["pages_per_second"],
        )
