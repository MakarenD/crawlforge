"""Asynchronous HTTP client and bounded website crawling orchestration."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import perf_counter
from types import TracebackType
from typing import TypedDict
from urllib.parse import urldefrag, urljoin, urlsplit

import aiohttp

from crawlforge.concurrency import SemaphoreManager
from crawlforge.parser import HTMLParser, ParsedPage
from crawlforge.politeness import RateLimiter, RobotsParser
from crawlforge.queue import CrawlerQueue
from crawlforge.urls import canonical_hostname

logger = logging.getLogger(__name__)

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_REDIRECTS = 10


class CrawlStats(TypedDict):
    """Snapshot of crawl progress and throughput."""

    processed: int
    queued: int
    active: int
    failed: int
    visited: int
    pages_per_second: float
    requests_per_second: float
    average_request_delay: float
    robots_blocked: int


@dataclass(frozen=True, slots=True)
class _CrawlOutcome:
    url: str
    depth: int
    page: ParsedPage | None
    error: str | None


class _RequestPolicyError(Exception):
    """Represent a non-network request policy failure."""


class _RobotsBlocked(_RequestPolicyError):
    """Represent a URL denied by robots.txt."""


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
        requests_per_second: float = 1.0,
        rate_limit_per_domain: bool = True,
        respect_robots: bool = True,
        min_delay: float = 0.0,
        jitter: float = 0.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        user_agent: str = "CrawlForge/0.1",
        user_agents: Sequence[str] | None = None,
    ) -> None:
        """Configure crawl boundaries, transport, and politeness policies."""
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")
        if max_depth < 0:
            raise ValueError("max_depth must be zero or greater")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than zero")
        if read_timeout <= 0:
            raise ValueError("read_timeout must be greater than zero")
        if not math.isfinite(requests_per_second) or requests_per_second <= 0:
            raise ValueError("requests_per_second must be a finite positive value")
        if not math.isfinite(min_delay) or min_delay < 0:
            raise ValueError("min_delay must be a finite non-negative value")
        if not math.isfinite(jitter) or jitter < 0:
            raise ValueError("jitter must be a finite non-negative value")
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater")
        if not math.isfinite(backoff_base) or backoff_base < 0:
            raise ValueError("backoff_base must be a finite non-negative value")
        if not math.isfinite(backoff_max) or backoff_max < backoff_base:
            raise ValueError("backoff_max must be finite and at least backoff_base")

        self._max_concurrent = max_concurrent
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            connect=connect_timeout,
            sock_read=read_timeout,
        )
        self._max_depth = max_depth
        self._respect_robots = respect_robots
        self._min_delay = min_delay
        self._jitter = jitter
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._random = random.Random()
        self._sleep = asyncio.sleep
        self._wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC)
        self._semaphores = SemaphoreManager(
            max_concurrent,
            max_concurrent_per_domain,
        )
        self.rate_limiter = RateLimiter(
            requests_per_second,
            per_domain=rate_limit_per_domain,
        )
        self._session_lock = asyncio.Lock()
        self._crawl_lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._parser = HTMLParser()
        self._closed = False
        self._crawl_started_at: float | None = None
        self._request_started_at: float | None = None
        self._last_request_at: float | None = None
        self._request_interval_total = 0.0
        self._request_count = 0
        self._robots_blocked = 0

        configured_agents = tuple(user_agents) if user_agents is not None else ()
        self._user_agents = configured_agents or (user_agent,)
        if any(
            not agent.strip() or "\r" in agent or "\n" in agent
            for agent in self._user_agents
        ):
            raise ValueError("user agents must be non-empty HTTP header values")
        self._user_agent_index = 0
        self.robots_parser = RobotsParser(
            self._fetch_robots_text,
            request_user_agent=self._user_agents[0],
        )

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
        request_elapsed = (
            perf_counter() - self._request_started_at
            if self._request_started_at is not None
            else 0.0
        )
        return {
            "processed": len(self.processed_urls),
            "queued": queue_stats["queued"],
            "active": self._semaphores.active_tasks,
            "failed": len(self.failed_urls),
            "visited": len(self.visited_urls),
            "pages_per_second": completed / elapsed if elapsed > 0 else 0.0,
            "requests_per_second": (
                self._request_count / request_elapsed if request_elapsed > 0 else 0.0
            ),
            "average_request_delay": (
                self._request_interval_total / (self._request_count - 1)
                if self._request_count > 1
                else 0.0
            ),
            "robots_blocked": self._robots_blocked,
        }

    async def _fetch_url_with_error(
        self,
        url: str,
    ) -> tuple[str, str | None, str]:
        logger.info("Fetching URL: %s", url)
        user_agent = self._next_user_agent()

        for attempt in range(self._max_retries + 1):
            try:
                content, final_url, status = await self._fetch_redirect_chain(
                    url,
                    user_agent,
                )
            except _RobotsBlocked as error:
                logger.warning("Blocked by robots.txt: %s", error)
                return "", f"Blocked by robots.txt: {error}", url
            except _RequestPolicyError as error:
                logger.warning("Request policy error for %s: %s", url, error)
                return "", f"{type(error).__name__}: {error}", url
            except aiohttp.ClientResponseError as error:
                if self._should_retry_status(error.status, attempt):
                    retry_after = (
                        error.headers.get("Retry-After")
                        if error.headers is not None
                        else None
                    )
                    await self._wait_before_retry(
                        url,
                        attempt,
                        error.status,
                        retry_after,
                    )
                    continue
                logger.warning(
                    "HTTP error for %s: %s (%s)",
                    url,
                    error.status,
                    type(error).__name__,
                )
                return "", f"HTTP {error.status}: {error.message}", url
            except TimeoutError as error:
                if attempt < self._max_retries:
                    await self._wait_before_retry(url, attempt, None)
                    continue
                logger.warning("Timeout for %s (%s)", url, type(error).__name__)
                return "", f"{type(error).__name__}: request timed out", url
            except aiohttp.ClientError as error:
                if not isinstance(error, aiohttp.InvalidURL) and (
                    attempt < self._max_retries
                ):
                    await self._wait_before_retry(url, attempt, None)
                    continue
                logger.warning("Network error for %s (%s)", url, type(error).__name__)
                return "", f"{type(error).__name__}: {error}", url
            else:
                logger.info("Fetched URL: %s (HTTP %s)", url, status)
                return content, None, final_url

        raise AssertionError("retry loop exhausted without a request outcome")

    async def _fetch_redirect_chain(
        self,
        url: str,
        user_agent: str,
    ) -> tuple[str, str, int]:
        current_url = url
        for redirect_count in range(_MAX_REDIRECTS + 1):
            minimum_interval = await self._request_minimum_interval(
                current_url,
                user_agent,
            )
            async with self._semaphores.limit(current_url):
                await self.rate_limiter.acquire(
                    canonical_hostname(current_url),
                    minimum_interval=minimum_interval,
                )
                session = await self._get_session()
                self._record_request_start()
                async with session.get(
                    current_url,
                    headers={"User-Agent": user_agent},
                    allow_redirects=False,
                ) as response:
                    location = response.headers.get("Location")
                    if response.status in _REDIRECT_STATUSES and location:
                        if redirect_count == _MAX_REDIRECTS:
                            raise _RequestPolicyError(
                                f"more than {_MAX_REDIRECTS} redirects"
                            )
                        current_url = urljoin(str(response.url), location)
                        continue
                    response.raise_for_status()
                    return (
                        await response.text(),
                        str(response.url),
                        response.status,
                    )

        raise AssertionError("redirect loop exhausted without a request outcome")

    async def _request_minimum_interval(
        self,
        url: str,
        user_agent: str,
    ) -> float:
        crawl_delay = 0.0
        if self._respect_robots:
            try:
                await self.robots_parser.fetch_robots(url)
            except ValueError:
                pass
            else:
                if not self.robots_parser.can_fetch(url, user_agent):
                    self._robots_blocked += 1
                    raise _RobotsBlocked(url)
                crawl_delay = self.robots_parser.get_crawl_delay_for(
                    url,
                    user_agent,
                )

        random_delay = self._random.uniform(0.0, self._jitter) if self._jitter else 0.0
        return max(self._min_delay, crawl_delay) + random_delay

    async def _fetch_robots_text(self, robots_url: str) -> tuple[int, str]:
        random_delay = self._random.uniform(0.0, self._jitter) if self._jitter else 0.0
        async with self._semaphores.limit(robots_url):
            await self.rate_limiter.acquire(
                canonical_hostname(robots_url),
                minimum_interval=self._min_delay + random_delay,
            )
            session = await self._get_session()
            self._record_request_start()
            async with session.get(
                robots_url,
                headers={"User-Agent": self._user_agents[0]},
            ) as response:
                return response.status, await response.text(errors="replace")

    def _next_user_agent(self) -> str:
        user_agent = self._user_agents[self._user_agent_index]
        self._user_agent_index = (self._user_agent_index + 1) % len(self._user_agents)
        return user_agent

    def _record_request_start(self) -> None:
        started_at = perf_counter()
        if self._request_started_at is None:
            self._request_started_at = started_at
        if self._last_request_at is not None:
            self._request_interval_total += started_at - self._last_request_at
        self._last_request_at = started_at
        self._request_count += 1

    def _should_retry_status(self, status: int, attempt: int) -> bool:
        return status in _RETRYABLE_STATUSES and attempt < self._max_retries

    async def _wait_before_retry(
        self,
        url: str,
        attempt: int,
        status: int | None,
        retry_after: str | None = None,
    ) -> None:
        backoff = min(self._backoff_base * (2**attempt), self._backoff_max)
        delay = max(backoff, self._retry_after_delay(retry_after))
        logger.info(
            "Retrying URL after %.2fs: %s (status=%s)",
            delay,
            url,
            status,
        )
        if delay:
            await self._sleep(delay)

    def _retry_after_delay(self, value: str | None) -> float:
        if value is None:
            return 0.0
        try:
            seconds = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return 0.0
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - self._wall_clock()).total_seconds())
        return seconds if math.isfinite(seconds) and seconds >= 0 else 0.0

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
        self._request_started_at = None
        self._last_request_at = None
        self._request_interval_total = 0.0
        self._request_count = 0
        self._robots_blocked = 0

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
