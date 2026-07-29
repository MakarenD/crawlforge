"""Dependency-safe console launcher for the optional MCP adapter."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

from crawlforge.mcp import MCP_INSTALL_HINT
from crawlforge.mcp.config import parse_config


def main(argv: Sequence[str] | None = None) -> int:
    """Parse configuration and start the optional local MCP server."""
    config = parse_config(argv)
    try:
        installed_version = version("mcp")
    except PackageNotFoundError:
        print(
            f"CrawlForge MCP support is not installed. {MCP_INSTALL_HINT}",
            file=sys.stderr,
        )
        return 2
    if not installed_version.startswith("2."):
        print(
            "CrawlForge MCP requires the official mcp SDK version 2.x. "
            f"{MCP_INSTALL_HINT}",
            file=sys.stderr,
        )
        return 2

    from crawlforge.mcp.server import run_server

    return run_server(config)
