"""Demonstrate rate limiting and robots.txt enforcement on a local site."""

from __future__ import annotations

import asyncio
import json
import logging

from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import AsyncCrawler


def build_demo_site() -> web.Application:
    """Create a deterministic site with one robots-blocked page."""

    async def robots(_request: web.Request) -> web.Response:
        return web.Response(
            text=("User-agent: PoliteDemoBot\nDisallow: /private\nCrawl-delay: 0.1\n")
        )

    async def index(_request: web.Request) -> web.Response:
        return web.Response(
            text=(
                '<a href="/public">Public page</a><a href="/private">Private page</a>'
            ),
            content_type="text/html",
        )

    async def page(request: web.Request) -> web.Response:
        return web.Response(text=request.path, content_type="text/html")

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", index)
    app.router.add_get("/{tail:.*}", page)
    return app


async def main() -> None:
    """Crawl the demo site and print politeness statistics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = TestServer(build_demo_site())
    await server.start_server()
    try:
        root_url = str(server.make_url("/"))
        async with AsyncCrawler(
            max_concurrent=5,
            max_depth=1,
            requests_per_second=2.0,
            respect_robots=True,
            min_delay=0.05,
            user_agent="PoliteDemoBot/1.0",
        ) as crawler:
            pages = await crawler.crawl([root_url], same_domain_only=True)
            stats = crawler.get_stats()

        print(
            json.dumps(
                {
                    "processed_urls": sorted(pages),
                    "failed_urls": crawler.failed_urls,
                    "request_rate": stats["requests_per_second"],
                    "average_request_delay": stats["average_request_delay"],
                    "robots_blocked": stats["robots_blocked"],
                },
                indent=2,
            )
        )
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(main())
