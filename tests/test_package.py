"""Tests for package metadata and public exports."""

from crawlforge import CrawlerQueue, HTMLParser, SemaphoreManager, __version__


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
