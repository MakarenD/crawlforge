"""Tests for package metadata."""

from crawlforge import HTMLParser, __version__


def test_package_version_is_exposed() -> None:
    """The package exposes a non-empty version string."""
    assert __version__ == "0.1.0"


def test_html_parser_is_exposed() -> None:
    """The package root exposes the public HTML parser."""
    assert HTMLParser.__name__ == "HTMLParser"
