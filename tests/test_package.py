"""Tests for package metadata."""

from crawlforge import __version__


def test_package_version_is_exposed() -> None:
    """The package exposes a non-empty version string."""
    assert __version__ == "0.1.0"
