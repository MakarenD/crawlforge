"""Tests for the command-line interface."""

from __future__ import annotations

import subprocess
import sys

from crawlforge import __version__


def test_module_help_is_available() -> None:
    """The module entry point prints its usage information."""
    result = subprocess.run(
        [sys.executable, "-m", "crawlforge", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "High-performance asynchronous web crawler" in result.stdout
    assert "--version" in result.stdout


def test_module_version_is_available() -> None:
    """The module entry point reports the package version."""
    result = subprocess.run(
        [sys.executable, "-m", "crawlforge", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"crawlforge {__version__}"
