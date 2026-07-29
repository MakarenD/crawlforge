"""Tests for the crawler-to-content processing boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge.crawler import AsyncCrawler, CrawledPage


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
async def test_page_handler_receives_raw_source_and_redirect_provenance() -> None:
    """A handler sees the complete successful response without changing results."""
    observed: list[CrawledPage] = []

    async def redirect(_request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/docs")

    async def docs(_request: web.Request) -> web.Response:
        return web.Response(
            text="<title>Docs</title><main>Useful source</main>",
            content_type="text/html",
        )

    async def handle(page: CrawledPage) -> None:
        observed.append(page)

    app = web.Application()
    app.router.add_get("/", redirect)
    app.router.add_get("/docs", docs)

    async with (
        serve(app) as server,
        AsyncCrawler(
            max_depth=0,
            respect_robots=False,
            requests_per_second=1000,
            page_handler=handle,
        ) as crawler,
    ):
        requested_url = str(server.make_url("/"))
        final_url = str(server.make_url("/docs"))
        results = await crawler.crawl([requested_url], max_pages=1)

    assert results[requested_url]["title"] == "Docs"
    assert observed == [
        CrawledPage(
            url=requested_url,
            final_url=final_url,
            html="<title>Docs</title><main>Useful source</main>",
            status_code=200,
            content_type="text/html",
            fetched_at=observed[0].fetched_at,
            depth=0,
        )
    ]
    assert observed[0].fetched_at.tzinfo is not None


@pytest.mark.asyncio
async def test_page_handler_failure_is_an_observable_page_failure() -> None:
    """A failed downstream processor cannot publish an unprocessed page."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<p>content</p>", content_type="text/html")

    async def fail(_page: CrawledPage) -> None:
        raise OSError("index unavailable")

    app = web.Application()
    app.router.add_get("/", page)

    async with (
        serve(app) as server,
        AsyncCrawler(
            max_depth=0,
            respect_robots=False,
            requests_per_second=1000,
            page_handler=fail,
        ) as crawler,
    ):
        url = str(server.make_url("/"))
        results = await crawler.crawl([url], max_pages=1)

    assert results == {}
    assert crawler.failed_urls[url] == "OSError: index unavailable"
    assert crawler.get_error_stats()["errors_by_type"] == {"OSError": 1}
