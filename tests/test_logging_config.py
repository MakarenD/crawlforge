"""Tests for console and rotating-file logging setup."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from crawlforge import LoggingConfig
from crawlforge.logging_config import configure_logging

_MANAGED_NAMES = {"crawlforge.console", "crawlforge.file"}


@pytest.fixture(autouse=True)
def restore_logging() -> Iterator[None]:
    """Remove handlers created by each logging test."""
    root = logging.getLogger()
    previous_level = root.level
    yield
    for handler in tuple(root.handlers):
        if handler.get_name() in _MANAGED_NAMES:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(previous_level)


def test_logging_writes_timestamped_console_and_rotating_file(
    tmp_path: Path,
) -> None:
    """Configured records reach a file and rotate at the requested bound."""
    path = tmp_path / "crawler.log"
    configure_logging(
        LoggingConfig(
            level="DEBUG",
            file=path,
            max_bytes=120,
            backup_count=1,
        )
    )
    logger = logging.getLogger("crawlforge.test")

    for index in range(10):
        logger.info("record %d %s", index, "x" * 40)
    for handler in logging.getLogger().handlers:
        handler.flush()

    rendered = path.read_text(encoding="utf-8")
    rotated = path.with_name("crawler.log.1")
    assert rotated.exists()
    assert "INFO crawlforge.test:" in rendered
    assert rendered[:4].isdigit()


def test_logging_reconfiguration_does_not_duplicate_managed_handlers(
    tmp_path: Path,
) -> None:
    """Repeated setup replaces only CrawlForge-owned handlers."""
    config = LoggingConfig(level="INFO", file=tmp_path / "crawler.log")

    configure_logging(config)
    configure_logging(config)

    managed = [
        handler
        for handler in logging.getLogger().handlers
        if handler.get_name() in _MANAGED_NAMES
    ]
    assert [handler.get_name() for handler in managed] == [
        "crawlforge.console",
        "crawlforge.file",
    ]
