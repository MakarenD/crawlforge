"""Demonstrate retries and save an error report from a deterministic local site."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import aiofiles
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import AsyncCrawler


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the retry and error-report demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("error-report.json"),
        help="path for the JSON error report",
    )
    return parser


def build_demo_site() -> web.Application:
    """Create endpoints with recoverable and permanent HTTP failures."""
    hits: dict[str, int] = {}

    async def index(_request: web.Request) -> web.Response:
        return web.Response(
            text=(
                '<a href="/flaky">Flaky service</a>'
                '<a href="/limited">Rate limited</a>'
                '<a href="/missing">Missing page</a>'
                '<a href="/server-error">Server error</a>'
            ),
            content_type="text/html",
        )

    async def flaky(request: web.Request) -> web.Response:
        hits[request.path] = hits.get(request.path, 0) + 1
        if hits[request.path] < 3:
            raise web.HTTPServiceUnavailable()
        return web.Response(text="<h1>Recovered from 503</h1>")

    async def limited(request: web.Request) -> web.Response:
        hits[request.path] = hits.get(request.path, 0) + 1
        if hits[request.path] == 1:
            return web.Response(status=429, headers={"Retry-After": "0.05"})
        return web.Response(text="<h1>Recovered from 429</h1>")

    async def missing(_request: web.Request) -> web.Response:
        raise web.HTTPNotFound()

    async def server_error(_request: web.Request) -> web.Response:
        raise web.HTTPInternalServerError()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/flaky", flaky)
    app.router.add_get("/limited", limited)
    app.router.add_get("/missing", missing)
    app.router.add_get("/server-error", server_error)
    return app


async def run_demo(output: Path) -> None:
    """Crawl the local endpoints and asynchronously persist an error report."""
    server = TestServer(build_demo_site())
    await server.start_server()
    try:
        root_url = str(server.make_url("/"))
        async with AsyncCrawler(
            max_concurrent=4,
            max_depth=1,
            respect_robots=False,
            requests_per_second=1000,
            max_retries=3,
            backoff_base=0.01,
            total_timeout=2.0,
        ) as crawler:
            pages = await crawler.crawl(
                [root_url],
                max_pages=5,
                same_domain_only=True,
            )
            stats = crawler.get_stats()

        report = {
            "crawl_stats": stats,
            "error_stats": crawler.get_error_stats(),
            "failed_urls": crawler.failed_urls,
            "error_history": [record.to_dict() for record in crawler.error_history],
            "successful_urls": sorted(pages),
        }
        async with aiofiles.open(output, "w", encoding="utf-8") as report_file:
            await report_file.write(json.dumps(report, ensure_ascii=False, indent=2))

        print(json.dumps(report["error_stats"], ensure_ascii=False, indent=2))
        print(f"Saved error report to {output}")
    finally:
        await server.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and run the local retry demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arguments = build_parser().parse_args(argv)
    asyncio.run(run_demo(arguments.output))


if __name__ == "__main__":
    main()
