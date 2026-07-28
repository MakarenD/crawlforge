"""Tests for package metadata and public exports."""

from crawlforge import (
    CrawlerQueue,
    CSVStorage,
    DataStorage,
    ErrorRecord,
    HTMLParser,
    JSONStorage,
    NetworkError,
    ParseError,
    PermanentError,
    RateLimiter,
    RetryStrategy,
    RobotsParser,
    SemaphoreManager,
    SQLiteStorage,
    TransientError,
    __version__,
)


def test_package_version_is_exposed() -> None:
    """The package exposes a non-empty version string."""
    assert __version__ == "0.1.0"


def test_html_parser_is_exposed() -> None:
    """The package root exposes the public HTML parser."""
    assert HTMLParser.__name__ == "HTMLParser"


def test_crawl_control_types_are_exposed() -> None:
    """The package root exposes queue and concurrency controls."""
    assert CrawlerQueue.__name__ == "CrawlerQueue"
    assert SemaphoreManager.__name__ == "SemaphoreManager"
    assert RateLimiter.__name__ == "RateLimiter"
    assert RobotsParser.__name__ == "RobotsParser"


def test_retry_and_error_types_are_exposed() -> None:
    """The package root exposes retry orchestration and error classification."""
    assert RetryStrategy.__name__ == "RetryStrategy"
    assert ErrorRecord.__name__ == "ErrorRecord"
    assert TransientError.__name__ == "TransientError"
    assert PermanentError.__name__ == "PermanentError"
    assert NetworkError.__name__ == "NetworkError"
    assert ParseError.__name__ == "ParseError"


def test_storage_types_are_exposed() -> None:
    """The package root exposes every asynchronous storage backend."""
    assert DataStorage.__name__ == "DataStorage"
    assert JSONStorage.__name__ == "JSONStorage"
    assert CSVStorage.__name__ == "CSVStorage"
    assert SQLiteStorage.__name__ == "SQLiteStorage"
