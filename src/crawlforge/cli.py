"""Command-line interface for CrawlForge."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from crawlforge import __version__
from crawlforge.advanced import AdvancedCrawler
from crawlforge.config import CrawlerConfig, LoggingConfig, ReportConfig
from crawlforge.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="crawlforge",
        description="High-performance asynchronous web crawler for Python.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show the installed CrawlForge version and exit",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        metavar="URL",
        help="one or more starting URLs",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="maximum number of pages to attempt",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="maximum discovered-link depth",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="maximum active requests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON result file",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        default=None,
        help="standalone HTML report file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file",
    )
    parser.add_argument(
        "--respect-robots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable robots.txt enforcement",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=None,
        metavar="REQUESTS_PER_SECOND",
        help="request rate limit",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="rotating log file",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="console and file logging level",
    )
    return parser


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    """Run one configured crawl and return its final statistics."""
    config = _config_from_arguments(arguments)
    configure_logging(config.logging)
    crawler = AdvancedCrawler(config)
    try:
        await crawler.crawl()
        return crawler.get_stats()
    finally:
        await crawler.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CrawlForge command-line interface."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.urls is None and arguments.config is None:
        parser.print_help()
        return 0
    try:
        stats = asyncio.run(run(arguments))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "total_pages": stats["total_pages"],
                "successful": stats["successful"],
                "failed": stats["failed"],
                "average_speed": stats["average_speed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _config_from_arguments(arguments: argparse.Namespace) -> CrawlerConfig:
    if arguments.config is not None:
        config = CrawlerConfig.from_file(arguments.config)
    else:
        config = CrawlerConfig(
            start_urls=tuple(arguments.urls or ()),
            reports=ReportConfig(
                json=arguments.output or Path("results.json"),
                html=arguments.html_report,
            ),
            logging=LoggingConfig(
                level=arguments.log_level or "INFO",
                file=arguments.log_file,
            ),
        )

    config = config.with_overrides(
        start_urls=arguments.urls,
        max_pages=arguments.max_pages,
        max_depth=arguments.max_depth,
        rate_limit=arguments.rate_limit,
        respect_robots=arguments.respect_robots,
        json_report=arguments.output,
    )
    if arguments.max_concurrent is not None:
        config = replace(config, max_concurrent=arguments.max_concurrent)
    if arguments.html_report is not None:
        config = replace(
            config,
            reports=replace(config.reports, html=arguments.html_report),
        )
    if arguments.log_level is not None or arguments.log_file is not None:
        config = replace(
            config,
            logging=replace(
                config.logging,
                level=arguments.log_level or config.logging.level,
                file=(
                    arguments.log_file
                    if arguments.log_file is not None
                    else config.logging.file
                ),
            ),
        )
    return config
