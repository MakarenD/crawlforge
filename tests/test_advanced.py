"""End-to-end tests for the integrated advanced crawler."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import AdvancedCrawler, CrawlerConfig, ReportConfig, StorageConfig


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
async def test_advanced_crawler_integrates_sitemap_filters_storage_and_reports(
    tmp_path: Path,
) -> None:
    """Sitemap seeds use the normal queue, storage, statistics, and reports."""

    async def sitemap(request: web.Request) -> web.Response:
        origin = f"{request.scheme}://{request.host}"
        return web.Response(
            text=(
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                f"<url><loc>{origin}/allowed/one</loc></url>"
                f"<url><loc>{origin}/private/hidden</loc></url>"
                "</urlset>"
            ),
            content_type="application/xml",
        )

    async def first(_request: web.Request) -> web.Response:
        return web.Response(
            text='<title>One</title><a href="/allowed/two">Two</a>',
            content_type="text/html",
        )

    async def second(_request: web.Request) -> web.Response:
        return web.Response(
            text="<title>Two</title>",
            content_type="text/html",
        )

    async def hidden(_request: web.Request) -> web.Response:
        raise AssertionError("excluded sitemap URL was requested")

    app = web.Application()
    app.router.add_get("/sitemap.xml", sitemap)
    app.router.add_get("/allowed/one", first)
    app.router.add_get("/allowed/two", second)
    app.router.add_get("/private/hidden", hidden)

    async with serve(app) as server:
        root = str(server.make_url("/"))
        config = CrawlerConfig(
            start_urls=(),
            sitemap_urls=(f"{root}sitemap.xml",),
            max_pages=10,
            max_depth=1,
            rate_limit=1000,
            respect_robots=False,
            same_domain_only=True,
            include_patterns=("/allowed/",),
            exclude_patterns=("/private/",),
            storage=StorageConfig("json", tmp_path / "data/pages.jsonl"),
            reports=ReportConfig(
                json=tmp_path / "reports/results.json",
                html=tmp_path / "reports/report.html",
            ),
        )
        async with AdvancedCrawler(config) as crawler:
            pages = await crawler.crawl()
            stats = crawler.get_stats()

    expected = {f"{root}allowed/one", f"{root}allowed/two"}
    assert set(pages) == expected
    assert stats["total_pages"] == 2
    assert stats["successful"] == 2
    assert stats["failed"] == 0
    assert stats["status_codes"] == {"200": 2}
    assert stats["progress_percent"] == 100.0
    stored = [
        json.loads(line)
        for line in (tmp_path / "data/pages.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {record["url"] for record in stored} == expected
    report = json.loads((tmp_path / "reports/results.json").read_text(encoding="utf-8"))
    assert report["stats"]["total_pages"] == 2
    assert set(report["pages"]) == expected
    html = (tmp_path / "reports/report.html").read_text(encoding="utf-8")
    assert "Status codes" in html
    assert "Top domains" in html
    assert "One" in html


@pytest.mark.asyncio
async def test_advanced_crawler_records_failed_status_distribution(
    tmp_path: Path,
) -> None:
    """One bad sitemap is isolated and page failures still reach statistics."""

    async def sitemap(request: web.Request) -> web.Response:
        origin = f"{request.scheme}://{request.host}"
        return web.Response(
            text=(
                "<urlset>"
                f"<url><loc>{origin}/ok</loc></url>"
                f"<url><loc>{origin}/missing</loc></url>"
                "</urlset>"
            )
        )

    async def ok(_request: web.Request) -> web.Response:
        return web.Response(text="<title>OK</title>")

    app = web.Application()
    app.router.add_get("/sitemap.xml", sitemap)
    app.router.add_get("/ok", ok)

    async with serve(app) as server:
        root = str(server.make_url("/"))
        config = CrawlerConfig(
            start_urls=(),
            sitemap_urls=(f"{root}missing.xml", f"{root}sitemap.xml"),
            max_retries=0,
            rate_limit=1000,
            respect_robots=False,
            reports=ReportConfig(json=tmp_path / "results.json"),
        )
        crawler = AdvancedCrawler(config)
        try:
            await crawler.crawl()
            stats = crawler.get_stats()
        finally:
            await crawler.close()

    assert stats["total_pages"] == 2
    assert stats["successful"] == 1
    assert stats["failed"] == 1
    assert stats["status_codes"] == {"200": 1, "404": 1}
    assert set(stats["sitemap_failures"]) == {f"{root}missing.xml"}
    assert "HTTP 404" in stats["sitemap_failures"][f"{root}missing.xml"]


@pytest.mark.asyncio
async def test_advanced_crawler_closes_resources_after_sitemap_error() -> None:
    """Callers can reliably close the integrated crawler after invalid XML."""

    async def invalid(_request: web.Request) -> web.Response:
        return web.Response(text="<not-closed>")

    app = web.Application()
    app.router.add_get("/sitemap.xml", invalid)

    async with serve(app) as server:
        config = CrawlerConfig(
            start_urls=(),
            sitemap_urls=(str(server.make_url("/sitemap.xml")),),
            rate_limit=1000,
            respect_robots=False,
        )
        crawler = AdvancedCrawler(config)
        with pytest.raises(ValueError, match="sitemap"):
            await crawler.crawl()
        await crawler.close()

    assert crawler._crawler._session is None
