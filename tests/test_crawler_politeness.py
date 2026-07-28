"""Integration tests for polite request behavior."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from itertools import pairwise
from time import perf_counter

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import AsyncCrawler


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
async def test_crawl_blocks_disallowed_start_url_and_logs_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A disallowed URL is recorded without requesting its page."""
    private_hits = 0

    async def robots(_request: web.Request) -> web.Response:
        return web.Response(
            text="User-agent: ExampleBot\nDisallow: /private\n",
        )

    async def private(_request: web.Request) -> web.Response:
        nonlocal private_hits
        private_hits += 1
        return web.Response(text="secret")

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/private", private)

    async with (
        serve(app) as server,
        AsyncCrawler(
            user_agent="ExampleBot/1.0",
            requests_per_second=1000,
        ) as crawler,
    ):
        private_url = str(server.make_url("/private"))
        with caplog.at_level(logging.WARNING, logger="crawlforge.crawler"):
            results = await crawler.crawl([private_url])

    assert results == {}
    assert private_hits == 0
    assert crawler.failed_urls[private_url].startswith("Blocked by robots.txt:")
    assert crawler.get_stats()["robots_blocked"] == 1
    assert f"Blocked by robots.txt: {private_url}" in caplog.text


@pytest.mark.asyncio
async def test_crawl_delay_minimum_and_jitter_reach_rate_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The effective interval combines robots, configured, and random delays."""
    calls: list[tuple[str | None, float]] = []

    class RecordingLimiter:
        async def acquire(
            self,
            domain: str | None = None,
            *,
            minimum_interval: float = 0.0,
        ) -> None:
            calls.append((domain, minimum_interval))

    async def robots(_request: web.Request) -> web.Response:
        return web.Response(
            text="User-agent: ExampleBot\nCrawl-delay: 0.4\n",
        )

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", page)

    async with (
        serve(app) as server,
        AsyncCrawler(
            user_agent="ExampleBot/1.0",
            requests_per_second=1000,
            min_delay=0.25,
            jitter=0.2,
        ) as crawler,
    ):
        monkeypatch.setattr(crawler, "rate_limiter", RecordingLimiter())
        monkeypatch.setattr(crawler._random, "uniform", lambda _start, _end: 0.1)
        assert await crawler.fetch_url(str(server.make_url("/"))) == "ok"

    assert calls == [
        ("127.0.0.1", pytest.approx(0.35)),
        ("127.0.0.1", pytest.approx(0.5)),
    ]


@pytest.mark.asyncio
async def test_redirect_destination_is_checked_against_robots() -> None:
    """Every redirect hop receives its own robots policy check."""
    private_hits = 0

    async def robots(_request: web.Request) -> web.Response:
        return web.Response(text="User-agent: *\nDisallow: /private\n")

    async def start(_request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/private")

    async def private(_request: web.Request) -> web.Response:
        nonlocal private_hits
        private_hits += 1
        return web.Response(text="secret")

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/start", start)
    app.router.add_get("/private", private)

    async with (
        serve(app) as server,
        AsyncCrawler(
            requests_per_second=1000,
        ) as crawler,
    ):
        result = await crawler.fetch_url(str(server.make_url("/start")))

    assert result == ""
    assert private_hits == 0
    assert crawler.get_stats()["robots_blocked"] == 1


@pytest.mark.asyncio
async def test_user_agents_rotate_once_per_logical_request() -> None:
    """User-Agent rotation is deterministic across logical requests."""
    observed: list[str] = []

    async def page(request: web.Request) -> web.Response:
        observed.append(request.headers["User-Agent"])
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/{tail:.*}", page)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
            user_agents=["Bot-A/1.0", "Bot-B/1.0"],
        ) as crawler,
    ):
        for path in ("/one", "/two", "/three"):
            assert await crawler.fetch_url(str(server.make_url(path))) == "ok"

    assert observed == ["Bot-A/1.0", "Bot-B/1.0", "Bot-A/1.0"]


@pytest.mark.asyncio
async def test_retryable_errors_use_exponential_backoff_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient failures retry with a bounded exponential schedule."""
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
            max_retries=2,
            backoff_base=0.1,
            backoff_max=1.0,
        ) as crawler,
    ):
        monkeypatch.setattr(crawler, "_sleep", record_sleep)
        result = await crawler.fetch_url(str(server.make_url("/")))

    assert result == "recovered"
    assert hits == 3
    assert sleeps == [0.1, 0.2]
    assert crawler.get_stats()["active"] == 0


@pytest.mark.asyncio
async def test_retry_after_extends_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry-After is a lower bound for a retryable HTTP response."""
    hits = 0
    sleeps: list[float] = []

    async def page(_request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        if hits == 1:
            return web.Response(status=429, headers={"Retry-After": "3"})
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
            max_retries=1,
            backoff_base=0.1,
        ) as crawler,
    ):
        monkeypatch.setattr(crawler, "_sleep", record_sleep)
        result = await crawler.fetch_url(str(server.make_url("/")))

    assert result == "recovered"
    assert sleeps == [3.0]


def test_retry_after_http_date_uses_wall_clock() -> None:
    """An HTTP-date Retry-After value is converted into a non-negative delay."""
    crawler = AsyncCrawler()
    now = datetime(2026, 7, 28, 10, 0, tzinfo=UTC)
    crawler._wall_clock = lambda: now
    retry_at = format_datetime(now + timedelta(seconds=15), usegmt=True)

    assert crawler._retry_after_delay(retry_at) == 15.0
    assert crawler._retry_after_delay("invalid") == 0.0


@pytest.mark.asyncio
async def test_slow_request_cannot_expire_later_rate_slots() -> None:
    """Queued requests remain spaced after a slow request releases capacity."""
    starts: list[float] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def page(request: web.Request) -> web.Response:
        starts.append(perf_counter())
        if request.path == "/one":
            first_started.set()
            await release_first.wait()
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/{tail:.*}", page)

    async with (
        serve(app) as server,
        AsyncCrawler(
            max_concurrent=3,
            max_concurrent_per_domain=1,
            respect_robots=False,
            requests_per_second=20,
        ) as crawler,
    ):
        task = asyncio.create_task(
            crawler.fetch_urls(
                [
                    str(server.make_url("/one")),
                    str(server.make_url("/two")),
                    str(server.make_url("/three")),
                ]
            )
        )
        await asyncio.wait_for(first_started.wait(), timeout=1)
        await asyncio.sleep(0.08)
        release_first.set()
        await asyncio.wait_for(task, timeout=2)

    assert len(starts) == 3
    assert starts[2] - starts[1] >= 0.04


@pytest.mark.asyncio
async def test_robots_requests_share_global_rate_and_statistics() -> None:
    """Robots and page requests use one global schedule and monitoring stream."""
    starts: list[float] = []

    async def page(request: web.Request) -> web.Response:
        starts.append(perf_counter())
        if request.path == "/robots.txt":
            return web.Response(text="User-agent: *\n")
        return web.Response(text="ok")

    first_app = web.Application()
    first_app.router.add_get("/{tail:.*}", page)
    second_app = web.Application()
    second_app.router.add_get("/{tail:.*}", page)

    async with (
        serve(first_app) as first_server,
        serve(second_app) as second_server,
        AsyncCrawler(
            max_concurrent=4,
            requests_per_second=50,
            rate_limit_per_domain=False,
        ) as crawler,
    ):
        results = await crawler.fetch_urls(
            [
                str(first_server.make_url("/page")),
                str(second_server.make_url("/page")),
            ]
        )
        stats = crawler.get_stats()

    assert all(results.values())
    assert len(starts) == 4
    assert all(later - earlier >= 0.015 for earlier, later in pairwise(starts))
    assert crawler._request_count == 4
    assert stats["average_request_delay"] >= 0.015


@pytest.mark.asyncio
async def test_request_statistics_report_rate_and_average_delay() -> None:
    """Transport statistics expose measured request rate and spacing."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/{tail:.*}", page)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=50,
        ) as crawler,
    ):
        await crawler.fetch_url(str(server.make_url("/one")))
        await crawler.fetch_url(str(server.make_url("/two")))
        stats = crawler.get_stats()

    assert stats["requests_per_second"] > 0
    assert stats["average_request_delay"] >= 0.015
    assert stats["robots_blocked"] == 0
