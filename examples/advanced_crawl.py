"""Run the integrated crawler from a JSON configuration file."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from crawlforge import AdvancedCrawler


async def crawl(config_path: Path) -> None:
    """Run one configured crawl and print its final summary."""
    crawler = AdvancedCrawler.from_config(config_path)
    try:
        await crawler.crawl()
        stats = crawler.get_stats()
        print(f"Processed: {stats['total_pages']} pages")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
    finally:
        await crawler.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the configuration path and start the crawl."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("advanced_config.json"),
    )
    arguments = parser.parse_args(argv)
    asyncio.run(crawl(arguments.config))


if __name__ == "__main__":
    main()
