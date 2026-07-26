"""Crawl one or more websites and save structured page data as JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import aiofiles

from crawlforge import AsyncCrawler


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the website crawling demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+", help="one or more starting URLs")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--per-domain", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("crawl-output.json"))
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="follow links outside the starting domains",
    )
    return parser


async def crawl_and_save(arguments: argparse.Namespace) -> None:
    """Run the crawl and asynchronously save all successful page results."""
    async with AsyncCrawler(
        max_concurrent=arguments.max_concurrent,
        max_concurrent_per_domain=arguments.per_domain,
        max_depth=arguments.max_depth,
    ) as crawler:
        results = await crawler.crawl(
            arguments.urls,
            max_pages=arguments.max_pages,
            same_domain_only=not arguments.allow_external,
        )
        stats = crawler.get_stats()

    payload = {
        "stats": stats,
        "failed_urls": crawler.failed_urls,
        "pages": results,
    }
    async with aiofiles.open(arguments.output, "w", encoding="utf-8") as output:
        await output.write(json.dumps(payload, ensure_ascii=False, indent=2))

    print(
        f"Processed {stats['processed']} pages with {stats['failed']} errors; "
        f"saved to {arguments.output}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the crawl demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(crawl_and_save(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
