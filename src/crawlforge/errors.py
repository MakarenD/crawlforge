"""Error types exposed by CrawlForge."""

from __future__ import annotations


class CrawlError(Exception):
    """Base class for classified crawl failures."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status: int | None = None,
        retry_after: float = 0.0,
        backoff_multiplier: float = 1.0,
        retry_limit: int | None = None,
    ) -> None:
        """Store structured context for logging and retry decisions."""
        super().__init__(message)
        self.url: str | None = url
        self.status: int | None = status
        self.retry_after: float = retry_after
        self.backoff_multiplier: float = backoff_multiplier
        self.retry_limit: int | None = retry_limit


class TransientError(CrawlError):
    """Represent a temporary HTTP or timeout failure."""


class PermanentError(CrawlError):
    """Represent a failure that another attempt cannot resolve."""


class NetworkError(CrawlError):
    """Represent a connection, DNS, or transport failure."""


class ParseError(CrawlError):
    """Represent a page decoding or parsing failure."""
