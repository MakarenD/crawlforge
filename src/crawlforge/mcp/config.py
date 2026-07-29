"""Immutable startup configuration for the local MCP server."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from crawlforge.network_policy import URLNetworkPolicy

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_MAX_PAGES_CAP = 100
DEFAULT_MAX_DEPTH_CAP = 3
DEFAULT_MAX_SEARCH_LIMIT = 20
DEFAULT_MAX_TOKEN_BUDGET = 12_000
DEFAULT_MAX_RESULT_BYTES = 64 * 1024
DEFAULT_MAX_WARNINGS = 10
DEFAULT_MAX_DIAGNOSTIC_CHARS = 500
DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_ROBOTS_BYTES = 512 * 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_CRAWL_TIMEOUT_SECONDS = 300.0

HARD_MAX_PAGES_CAP = 1_000
HARD_MAX_DEPTH_CAP = 20
HARD_MAX_SEARCH_LIMIT = 100
HARD_MAX_TOKEN_BUDGET = 100_000
HARD_MAX_RESPONSE_BYTES = 100 * 1024 * 1024
HARD_MAX_ROBOTS_BYTES = 10 * 1024 * 1024
HARD_MAX_TIMEOUT_SECONDS = 3_600.0


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Server-owned limits and security policy fixed for one process."""

    database: Path = Path(".crawlforge/index.db")
    max_pages_cap: int = DEFAULT_MAX_PAGES_CAP
    max_depth_cap: int = DEFAULT_MAX_DEPTH_CAP
    max_search_limit: int = DEFAULT_MAX_SEARCH_LIMIT
    max_token_budget: int = DEFAULT_MAX_TOKEN_BUDGET
    allow_private_networks: bool = False
    allowed_domains: tuple[str, ...] = ()
    log_level: LogLevel = "INFO"
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES
    max_warnings: int = DEFAULT_MAX_WARNINGS
    max_diagnostic_chars: int = DEFAULT_MAX_DIAGNOSTIC_CHARS
    requests_per_second: float = 1.0
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_robots_bytes: int = DEFAULT_MAX_ROBOTS_BYTES
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    crawl_timeout_seconds: float = DEFAULT_CRAWL_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        _validate_cap("max_pages_cap", self.max_pages_cap, HARD_MAX_PAGES_CAP)
        _validate_cap("max_depth_cap", self.max_depth_cap, HARD_MAX_DEPTH_CAP)
        _validate_cap(
            "max_search_limit",
            self.max_search_limit,
            HARD_MAX_SEARCH_LIMIT,
        )
        _validate_cap(
            "max_token_budget",
            self.max_token_budget,
            HARD_MAX_TOKEN_BUDGET,
        )
        if self.max_result_bytes < 1_024:
            raise ValueError("max_result_bytes must be at least 1024")
        if self.max_warnings <= 0:
            raise ValueError("max_warnings must be greater than zero")
        if self.max_diagnostic_chars < 100:
            raise ValueError("max_diagnostic_chars must be at least 100")
        if not math.isfinite(self.requests_per_second) or self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be a finite positive value")
        _validate_cap(
            "max_response_bytes",
            self.max_response_bytes,
            HARD_MAX_RESPONSE_BYTES,
        )
        _validate_cap(
            "max_robots_bytes",
            self.max_robots_bytes,
            HARD_MAX_ROBOTS_BYTES,
        )
        _validate_timeout(
            "request_timeout_seconds",
            self.request_timeout_seconds,
        )
        _validate_timeout(
            "crawl_timeout_seconds",
            self.crawl_timeout_seconds,
        )
        URLNetworkPolicy(
            allow_private_networks=self.allow_private_networks,
            allowed_domains=self.allowed_domains,
        )

    @property
    def database_label(self) -> str:
        """Return a safe logical name without exposing parent directories."""
        return self.database.name or "index.db"

    def network_policy(self) -> URLNetworkPolicy:
        """Build the immutable core policy used by the lifecycle engine."""
        return URLNetworkPolicy(
            allow_private_networks=self.allow_private_networks,
            allowed_domains=self.allowed_domains,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone MCP server argument parser."""
    parser = argparse.ArgumentParser(
        prog="crawlforge-mcp",
        description="Run the local CrawlForge MCP server over stdio.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(".crawlforge/index.db"),
        help="fixed local SQLite context index for this server process",
    )
    parser.add_argument(
        "--max-pages-cap",
        type=int,
        default=DEFAULT_MAX_PAGES_CAP,
        help=f"server cap for index_site pages (maximum {HARD_MAX_PAGES_CAP})",
    )
    parser.add_argument(
        "--max-depth-cap",
        type=int,
        default=DEFAULT_MAX_DEPTH_CAP,
        help=f"server cap for crawl depth (maximum {HARD_MAX_DEPTH_CAP})",
    )
    parser.add_argument(
        "--max-search-limit",
        type=int,
        default=DEFAULT_MAX_SEARCH_LIMIT,
        help=f"server cap for returned chunks (maximum {HARD_MAX_SEARCH_LIMIT})",
    )
    parser.add_argument(
        "--max-token-budget",
        type=int,
        default=DEFAULT_MAX_TOKEN_BUDGET,
        help=(
            "server cap for approximate context tokens "
            f"(maximum {HARD_MAX_TOKEN_BUDGET})"
        ),
    )
    parser.add_argument(
        "--allow-private-networks",
        action="store_true",
        help="opt in to loopback, private, and link-local network targets",
    )
    parser.add_argument(
        "--allow-domain",
        action="append",
        default=[],
        metavar="DOMAIN",
        help="allow an exact hostname and its subdomains; may be repeated",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="INFO",
        help="stderr diagnostic level",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=1.0,
        help="polite per-domain request rate used by index_site",
    )
    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=DEFAULT_MAX_RESPONSE_BYTES,
        help="maximum decoded source bytes read for one page response",
    )
    parser.add_argument(
        "--max-robots-bytes",
        type=int,
        default=DEFAULT_MAX_ROBOTS_BYTES,
        help="maximum bytes read from one robots.txt response",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="total timeout for one HTTP request attempt",
    )
    parser.add_argument(
        "--crawl-timeout",
        type=float,
        default=DEFAULT_CRAWL_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="total foreground timeout for one index_site call",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> MCPServerConfig:
    """Parse startup arguments into an immutable validated configuration."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return MCPServerConfig(
            database=arguments.database,
            max_pages_cap=arguments.max_pages_cap,
            max_depth_cap=arguments.max_depth_cap,
            max_search_limit=arguments.max_search_limit,
            max_token_budget=arguments.max_token_budget,
            allow_private_networks=arguments.allow_private_networks,
            allowed_domains=tuple(arguments.allow_domain),
            log_level=cast(LogLevel, arguments.log_level),
            requests_per_second=arguments.requests_per_second,
            max_response_bytes=arguments.max_response_bytes,
            max_robots_bytes=arguments.max_robots_bytes,
            request_timeout_seconds=arguments.request_timeout,
            crawl_timeout_seconds=arguments.crawl_timeout,
        )
    except ValueError as error:
        parser.error(str(error))


def _validate_cap(name: str, value: int, hard_maximum: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if value > hard_maximum:
        raise ValueError(f"{name} must not exceed {hard_maximum}")


def _validate_timeout(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive value")
    if value > HARD_MAX_TIMEOUT_SECONDS:
        raise ValueError(f"{name} must not exceed {HARD_MAX_TIMEOUT_SECONDS}")
