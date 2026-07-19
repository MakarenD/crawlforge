"""Tests for the asynchronous HTTP crawler."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
async def test_fetch_url_returns_body_and_logs_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A successful request returns the decoded response body."""

    async def page(request: web.Request) -> web.Response:
        return web.Response(text="<h1>Привет</h1>", content_type="text/html")

    app = web.Application()
    app.router.add_get("/page", page)

    async with serve(app) as server, AsyncCrawler() as crawler:
        url = str(server.make_url("/page"))
        with caplog.at_level(logging.INFO, logger="crawlforge.crawler"):
            result = await crawler.fetch_url(url)

    assert result == "<h1>Привет</h1>"
    assert f"Fetching URL: {url}" in caplog.text
    assert f"Fetched URL: {url} (HTTP 200)" in caplog.text


@pytest.mark.asyncio
async def test_fetch_urls_isolates_http_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One HTTP error does not abort the remaining batch requests."""

    async def ok(request: web.Request) -> web.Response:
        return web.Response(text=request.match_info["value"])

    async def missing(request: web.Request) -> web.Response:
        raise web.HTTPNotFound()

    app = web.Application()
    app.router.add_get("/ok/{value}", ok)
    app.router.add_get("/missing", missing)

    async with serve(app) as server, AsyncCrawler(max_concurrent=3) as crawler:
        first_url = str(server.make_url("/ok/first"))
        missing_url = str(server.make_url("/missing"))
        second_url = str(server.make_url("/ok/second"))
        with caplog.at_level(logging.WARNING, logger="crawlforge.crawler"):
            results = await crawler.fetch_urls(
                [first_url, missing_url, second_url],
            )
        reused_session_result = await crawler.fetch_url(first_url)

    assert results == {
        first_url: "first",
        missing_url: "",
        second_url: "second",
    }
    assert reused_session_result == "first"
    assert f"HTTP error for {missing_url}: 404 (ClientResponseError)" in caplog.text


@pytest.mark.asyncio
async def test_invalid_url_is_handled_as_client_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed URL produces an empty result instead of escaping the client."""
    async with AsyncCrawler() as crawler:
        with caplog.at_level(logging.WARNING, logger="crawlforge.crawler"):
            result = await crawler.fetch_url("://invalid")

    assert result == ""
    assert "Network error for ://invalid" in caplog.text


@pytest.mark.asyncio
async def test_read_timeout_is_handled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stalled response returns an empty result and records a timeout."""
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def stalled(request: web.Request) -> web.Response:
        handler_started.set()
        await release_handler.wait()
        return web.Response(text="late")

    app = web.Application()
    app.router.add_get("/stalled", stalled)

    async with serve(app) as server, AsyncCrawler(read_timeout=0.02) as crawler:
        url = str(server.make_url("/stalled"))
        fetch_task = asyncio.create_task(crawler.fetch_url(url))
        await asyncio.wait_for(handler_started.wait(), timeout=5)
        try:
            with caplog.at_level(logging.WARNING, logger="crawlforge.crawler"):
                result = await fetch_task
        finally:
            release_handler.set()

    assert result == ""
    assert f"Timeout for {url} (" in caplog.text


@pytest.mark.asyncio
async def test_concurrency_never_exceeds_configured_limit() -> None:
    """The semaphore admits no more than max_concurrent requests at once."""
    active = 0
    started = 0
    max_active = 0
    limit_reached = asyncio.Event()
    release_requests = asyncio.Event()

    async def blocked(request: web.Request) -> web.Response:
        nonlocal active, max_active, started
        active += 1
        started += 1
        max_active = max(max_active, active)
        if started == 2:
            limit_reached.set()
        try:
            await release_requests.wait()
            return web.Response(text=request.query["id"])
        finally:
            active -= 1

    app = web.Application()
    app.router.add_get("/blocked", blocked)

    async with serve(app) as server, AsyncCrawler(max_concurrent=2) as crawler:
        urls = [str(server.make_url(f"/blocked?id={index}")) for index in range(5)]
        batch = asyncio.create_task(crawler.fetch_urls(urls))
        await asyncio.wait_for(limit_reached.wait(), timeout=5)

        assert active == 2
        assert started == 2

        release_requests.set()
        results = await asyncio.wait_for(batch, timeout=5)

    assert max_active == 2
    assert results == {url: str(index) for index, url in enumerate(urls)}


@pytest.mark.asyncio
async def test_parallel_fetch_overlaps_requests_while_sequential_does_not() -> None:
    """Batch fetching overlaps work that sequential calls perform one at a time."""
    active = 0
    max_active = 0
    parallel_mode = False
    parallel_started = 0
    all_parallel_started = asyncio.Event()
    release_parallel = asyncio.Event()

    async def observed(request: web.Request) -> web.Response:
        nonlocal active, max_active, parallel_started
        active += 1
        max_active = max(max_active, active)
        try:
            if parallel_mode:
                parallel_started += 1
                if parallel_started == 3:
                    all_parallel_started.set()
                await release_parallel.wait()
            return web.Response(text=request.query["id"])
        finally:
            active -= 1

    app = web.Application()
    app.router.add_get("/observed", observed)

    async with serve(app) as server, AsyncCrawler(max_concurrent=3) as crawler:
        urls = [str(server.make_url(f"/observed?id={index}")) for index in range(3)]
        sequential = {url: await crawler.fetch_url(url) for url in urls}

        assert max_active == 1

        parallel_mode = True
        max_active = 0
        batch = asyncio.create_task(crawler.fetch_urls(urls))
        await asyncio.wait_for(all_parallel_started.wait(), timeout=5)

        assert active == 3

        release_parallel.set()
        parallel = await asyncio.wait_for(batch, timeout=5)

    assert max_active == 3
    assert parallel == sequential


@pytest.mark.asyncio
async def test_cancellation_propagates_and_releases_capacity() -> None:
    """Cancellation is not swallowed and does not leak a semaphore permit."""
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def blocked(request: web.Request) -> web.Response:
        request_started.set()
        await release_request.wait()
        return web.Response(text="blocked")

    async def ok(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/blocked", blocked)
    app.router.add_get("/ok", ok)

    async with serve(app) as server, AsyncCrawler(max_concurrent=1) as crawler:
        task = asyncio.create_task(
            crawler.fetch_url(str(server.make_url("/blocked"))),
        )
        await asyncio.wait_for(request_started.wait(), timeout=5)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        release_request.set()
        result = await asyncio.wait_for(
            crawler.fetch_url(str(server.make_url("/ok"))),
            timeout=5,
        )

    assert result == "ok"


@pytest.mark.asyncio
async def test_context_manager_closes_session_after_exception() -> None:
    """The crawler closes its session even when its context exits by exception."""

    class ExpectedError(Exception):
        pass

    async def ok(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/ok", ok)
    crawler = AsyncCrawler()
    session = None

    async with serve(app) as server:
        with pytest.raises(ExpectedError):
            async with crawler:
                assert await crawler.fetch_url(str(server.make_url("/ok"))) == "ok"
                session = crawler._session
                raise ExpectedError

    assert session is not None
    assert session.closed
    await crawler.close()
    await crawler.close()


@pytest.mark.asyncio
async def test_close_finishes_session_cleanup_before_propagating_cancellation() -> None:
    """Cancelling close does not abandon the in-progress session cleanup."""

    class BlockingSession:
        def __init__(self) -> None:
            self.closed = False
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.closed = True

    crawler = AsyncCrawler()
    session = BlockingSession()
    crawler._session = session  # type: ignore[assignment]

    close_task = asyncio.create_task(crawler.close())
    await asyncio.wait_for(session.close_started.wait(), timeout=5)
    close_task.cancel()
    await asyncio.sleep(0)

    assert not close_task.done()

    session.release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert session.closed
    assert crawler._session is None


@pytest.mark.asyncio
async def test_empty_batch_does_not_create_session() -> None:
    """An empty URL list returns immediately without allocating HTTP resources."""
    crawler = AsyncCrawler()

    assert await crawler.fetch_urls([]) == {}
    assert crawler._session is None

    await crawler.close()


@pytest.mark.parametrize("max_concurrent", [0, -1])
def test_invalid_concurrency_is_rejected(max_concurrent: int) -> None:
    """Non-positive concurrency cannot create a permanently blocked crawler."""
    with pytest.raises(ValueError, match="max_concurrent"):
        AsyncCrawler(max_concurrent=max_concurrent)


@pytest.mark.asyncio
async def test_session_pool_and_timeouts_are_reused() -> None:
    """Requests share one pooled session with the configured transport limits."""

    async def ok(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/ok", ok)

    async with (
        serve(app) as server,
        AsyncCrawler(
            max_concurrent=4,
            connect_timeout=2.0,
            read_timeout=3.0,
        ) as crawler,
    ):
        url = str(server.make_url("/ok"))
        await crawler.fetch_url(url)
        first_session = crawler._session
        await crawler.fetch_url(url)

        assert first_session is crawler._session
        assert first_session is not None
        assert first_session.connector is not None
        assert first_session.connector.limit == 4
        assert first_session.timeout.connect == 2.0
        assert first_session.timeout.sock_read == 3.0
