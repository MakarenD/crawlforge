"""Integration tests for crawler error classification and retry reporting."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import AsyncCrawler, NetworkError, RetryStrategy, TransientError


@asynccontextmanager
async def serve(app: web.Application) -> AsyncIterator[TestServer]:
    """Run an aiohttp application on an ephemeral local port."""
    server = TestServer(app)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_503_retries_and_updates_successful_retry_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A service-unavailable response retries and records recovery."""
    hits = 0
    sleeps: list[float] = []

    async def page(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        if hits < 3:
            raise web.HTTPServiceUnavailable()
        return web.Response(text="recovered")

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    app = web.Application()
    app.router.add_get("/", page)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
            max_retries=3,
            backoff_base=0.1,
        ) as crawler,
    ):
        monkeypatch.setattr(crawler, "_sleep", record_sleep)
        url = str(server.make_url("/"))
        result = await crawler.fetch_url(url)

    assert result == "recovered"
    assert hits == 3
    assert sleeps == [0.1, 0.2]
    assert crawler.get_error_stats() == {
        "errors_by_type": {"TransientError": 2},
        "total_retries": 2,
        "successful_retries": 1,
        "average_retry_delay": pytest.approx(0.15),
        "permanent_error_urls": [],
    }
    assert [record.status for record in crawler.error_history] == [503, 503]


@pytest.mark.asyncio
async def test_404_is_permanent_and_never_retried() -> None:
    """A missing page is attempted once and listed as permanent."""
    hits = 0

    async def missing(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        raise web.HTTPNotFound()

    app = web.Application()
    app.router.add_get("/missing", missing)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
            max_retries=3,
            backoff_base=0,
        ) as crawler,
    ):
        url = str(server.make_url("/missing"))
        assert await crawler.fetch_url(url) == ""

    assert hits == 1
    assert crawler.get_error_stats()["permanent_error_urls"] == [url]
    assert crawler.get_error_stats()["total_retries"] == 0


@pytest.mark.asyncio
async def test_429_uses_increased_backoff_without_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate limiting doubles the normal transient backoff when no hint exists."""
    hits = 0
    sleeps: list[float] = []

    async def limited(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        if hits == 1:
            raise web.HTTPTooManyRequests()
        return web.Response(text="ready")

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    app = web.Application()
    app.router.add_get("/", limited)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
            max_retries=1,
            backoff_base=0.1,
        ) as crawler,
    ):
        monkeypatch.setattr(crawler, "_sleep", record_sleep)
        assert await crawler.fetch_url(str(server.make_url("/"))) == "ready"

    assert sleeps == [0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_permanent_http_statuses_are_not_retried(status: int) -> None:
    """Authentication, authorization, and missing-page failures are permanent."""
    hits = 0

    async def failure(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        return web.Response(status=status)

    app = web.Application()
    app.router.add_get("/", failure)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
            max_retries=3,
            backoff_base=0,
        ) as crawler,
    ):
        assert await crawler.fetch_url(str(server.make_url("/"))) == ""

    assert hits == 1


@pytest.mark.asyncio
async def test_500_has_a_stricter_retry_limit() -> None:
    """An internal server error retries once even when the global limit is higher."""
    hits = 0

    async def failure(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        raise web.HTTPInternalServerError()

    app = web.Application()
    app.router.add_get("/", failure)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
            max_retries=3,
            backoff_base=0,
        ) as crawler,
    ):
        assert await crawler.fetch_url(str(server.make_url("/"))) == ""

    assert hits == 2
    assert crawler.get_error_stats()["total_retries"] == 1


@pytest.mark.asyncio
async def test_timeouts_increase_for_each_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connect, read, and total timeout budgets grow between attempts."""
    observed: list[tuple[float | None, float | None, float | None]] = []

    async def fake_fetch(
        _url: str,
        _user_agent: str,
    ) -> tuple[str, str, int]:
        timeout = crawler._request_timeout()
        observed.append((timeout.connect, timeout.sock_read, timeout.total))
        if len(observed) == 1:
            raise TimeoutError
        return "ok", "https://example.test", 200

    async def no_sleep(_delay: float) -> None:
        return None

    crawler = AsyncCrawler(
        respect_robots=False,
        connect_timeout=1.0,
        read_timeout=2.0,
        total_timeout=3.0,
        timeout_backoff_factor=2.0,
        max_retries=1,
        backoff_base=0,
    )
    monkeypatch.setattr(crawler, "_fetch_redirect_chain", fake_fetch)
    monkeypatch.setattr(crawler, "_sleep", no_sleep)
    try:
        assert await crawler.fetch_url("https://example.test") == "ok"
    finally:
        await crawler.close()

    assert observed == [(1.0, 2.0, 3.0), (2.0, 4.0, 6.0)]
    assert crawler.get_error_stats()["errors_by_type"] == {"TransientError": 1}


@pytest.mark.asyncio
async def test_network_error_retries_without_leaking_transport_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection failure retries and leaves request capacity available."""
    attempts = 0

    async def fake_fetch(
        _url: str,
        _user_agent: str,
    ) -> tuple[str, str, int]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise aiohttp.ClientConnectionError("connection refused")
        return "ok", "https://example.test", 200

    async def no_sleep(_delay: float) -> None:
        return None

    crawler = AsyncCrawler(
        respect_robots=False,
        max_retries=1,
        backoff_base=0,
    )
    monkeypatch.setattr(crawler, "_fetch_redirect_chain", fake_fetch)
    monkeypatch.setattr(crawler, "_sleep", no_sleep)
    try:
        assert await crawler.fetch_url("https://example.test") == "ok"
    finally:
        await crawler.close()

    assert attempts == 2
    assert crawler.get_stats()["active"] == 0
    assert crawler.get_error_stats()["errors_by_type"] == {"NetworkError": 1}


@pytest.mark.asyncio
async def test_custom_retry_strategy_controls_crawler_attempts() -> None:
    """A supplied strategy replaces the crawler's compatibility retry settings."""
    hits = 0

    async def unavailable(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        raise web.HTTPServiceUnavailable()

    app = web.Application()
    app.router.add_get("/", unavailable)
    strategy = RetryStrategy(
        max_retries=0,
        backoff_factor=0,
        retry_on=[TransientError],
    )

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
            max_retries=3,
            retry_strategy=strategy,
        ) as crawler,
    ):
        assert await crawler.fetch_url(str(server.make_url("/"))) == ""

    assert hits == 1
    assert crawler.retry_strategy is strategy


@pytest.mark.asyncio
async def test_transient_robots_failure_retries_before_page_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A temporary robots failure recovers before evaluating the target URL."""
    robots_hits = 0
    page_hits = 0

    async def robots(_request: web.Request) -> web.Response:
        nonlocal robots_hits
        robots_hits += 1
        if robots_hits == 1:
            raise web.HTTPServiceUnavailable()
        return web.Response(text="User-agent: *\n")

    async def page(_request: web.Request) -> web.Response:
        nonlocal page_hits
        page_hits += 1
        return web.Response(text="allowed")

    async def no_sleep(_delay: float) -> None:
        return None

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", page)

    async with (
        serve(app) as server,
        AsyncCrawler(
            requests_per_second=1000,
            max_retries=2,
            backoff_base=0,
        ) as crawler,
    ):
        monkeypatch.setattr(crawler, "_sleep", no_sleep)
        assert await crawler.fetch_url(str(server.make_url("/"))) == "allowed"

    assert robots_hits == 2
    assert page_hits == 1
    assert crawler.get_error_stats()["successful_retries"] == 1
    assert crawler.get_error_stats()["errors_by_type"] == {"TransientError": 1}
    assert crawler.error_history[0].url is not None
    assert crawler.error_history[0].url.endswith("/robots.txt")


@pytest.mark.asyncio
async def test_exhausted_robots_retries_fail_closed_and_cache_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Robots failures deny the domain only after bounded retries are exhausted."""
    robots_hits = 0
    page_hits = 0

    async def robots(_request: web.Request) -> web.Response:
        nonlocal robots_hits
        robots_hits += 1
        raise web.HTTPServiceUnavailable()

    async def page(_request: web.Request) -> web.Response:
        nonlocal page_hits
        page_hits += 1
        return web.Response(text="must not load")

    async def no_sleep(_delay: float) -> None:
        return None

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", page)

    async with (
        serve(app) as server,
        AsyncCrawler(
            requests_per_second=1000,
            max_retries=1,
            backoff_base=0,
        ) as crawler,
    ):
        monkeypatch.setattr(crawler, "_sleep", no_sleep)
        url = str(server.make_url("/"))
        assert await crawler.fetch_url(url) == ""
        assert await crawler.fetch_url(url) == ""

    assert robots_hits == 2
    assert page_hits == 0
    assert crawler.get_error_stats()["total_retries"] == 1
    assert crawler.get_stats()["robots_blocked"] == 2


@pytest.mark.asyncio
async def test_network_failure_retries_during_robots_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The robots transport uses the same bounded network retry policy."""
    attempts = 0

    async def fake_fetch(_robots_url: str) -> tuple[int, str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise NetworkError("DNS unavailable")
        return 200, "User-agent: *\n"

    async def no_sleep(_delay: float) -> None:
        return None

    crawler = AsyncCrawler(max_retries=1, backoff_base=0)
    monkeypatch.setattr(crawler, "_fetch_robots_attempt", fake_fetch)
    monkeypatch.setattr(crawler, "_sleep", no_sleep)
    try:
        result = await crawler.robots_parser.fetch_robots("https://example.test/page")
    finally:
        await crawler.close()

    assert attempts == 2
    assert result["status"] == 200
    assert crawler.get_error_stats()["errors_by_type"] == {"NetworkError": 1}
    assert crawler.get_error_stats()["successful_retries"] == 1
