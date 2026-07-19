"""Command-line interface for CrawlForge."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from crawlforge import __version__


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CrawlForge command-line interface."""
    build_parser().parse_args(argv)
    return 0
