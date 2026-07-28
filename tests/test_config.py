"""Tests for validated JSON crawler configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crawlforge import CrawlerConfig


def write_config(path: Path, payload: object) -> None:
    """Write a test configuration as UTF-8 JSON."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_config_loads_crawler_filters_storage_logging_and_reports(
    tmp_path: Path,
) -> None:
    """Every supported section maps to typed settings with relative paths."""
    path = tmp_path / "crawler.json"
    write_config(
        path,
        {
            "urls": ["https://example.com/"],
            "sitemaps": ["https://example.com/sitemap.xml"],
            "crawler": {
                "max_pages": 500,
                "max_depth": 3,
                "max_concurrent": 20,
                "max_concurrent_per_domain": 4,
                "rate_limit": 5,
                "rate_limit_per_domain": False,
                "respect_robots": True,
                "min_delay": 0.2,
                "jitter": 0.1,
                "max_retries": 4,
                "connect_timeout": 2,
                "read_timeout": 8,
                "total_timeout": 12,
            },
            "filters": {
                "same_domain_only": True,
                "include": ["/docs/"],
                "exclude": ["/private/"],
            },
            "storage": {
                "format": "json",
                "path": "data/pages.json",
                "json_lines": False,
                "indent": 2,
            },
            "logging": {
                "level": "debug",
                "file": "logs/crawl.log",
                "max_bytes": 1024,
                "backup_count": 2,
            },
            "reports": {
                "json": "reports/results.json",
                "html": "reports/results.html",
            },
        },
    )

    config = CrawlerConfig.from_file(path)

    assert config.start_urls == ("https://example.com/",)
    assert config.sitemap_urls == ("https://example.com/sitemap.xml",)
    assert config.max_pages == 500
    assert config.max_concurrent_per_domain == 4
    assert config.rate_limit == 5.0
    assert not config.rate_limit_per_domain
    assert config.same_domain_only
    assert config.include_patterns == ("/docs/",)
    assert config.exclude_patterns == ("/private/",)
    assert config.storage is not None
    assert config.storage.path == tmp_path / "data/pages.json"
    assert config.storage.indent == 2
    assert config.logging.level == "DEBUG"
    assert config.logging.file == tmp_path / "logs/crawl.log"
    assert config.reports.json == tmp_path / "reports/results.json"
    assert config.reports.html == tmp_path / "reports/results.html"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "requires urls or sitemaps"),
        ({"urls": ["ftp://example.com/"]}, "invalid configured URL"),
        (
            {"urls": ["https://example.com/"], "unknown": True},
            "unknown configuration option",
        ),
        (
            {
                "urls": ["https://example.com/"],
                "filters": {"include": ["["]},
            },
            "invalid regex",
        ),
        (
            {
                "urls": ["https://example.com/"],
                "storage": {"format": "json", "path": "out", "indent": 2},
            },
            "indent requires",
        ),
        (
            {
                "urls": ["https://example.com/"],
                "storage": {
                    "format": "sqlite",
                    "path": "pages.db",
                    "encoding": "utf-8",
                },
            },
            "unknown storage option",
        ),
        (
            {
                "urls": ["https://example.com/"],
                "crawler": {"max_concurrent": True},
            },
            "must be an integer",
        ),
    ],
)
def test_config_rejects_invalid_values(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    """Invalid and unknown options fail before any crawl starts."""
    path = tmp_path / "invalid.json"
    write_config(path, payload)

    with pytest.raises(ValueError, match=message):
        CrawlerConfig.from_file(path)


def test_cli_style_overrides_preserve_unspecified_config_values() -> None:
    """Only explicitly supplied command-line values replace file settings."""
    config = CrawlerConfig(
        start_urls=("https://example.com/",),
        max_pages=500,
        max_depth=4,
        rate_limit=3.0,
        respect_robots=True,
    )

    overridden = config.with_overrides(
        max_pages=25,
        respect_robots=False,
        json_report=Path("custom.json"),
    )

    assert overridden.max_pages == 25
    assert overridden.max_depth == 4
    assert overridden.rate_limit == 3.0
    assert not overridden.respect_robots
    assert overridden.reports.json == Path("custom.json")


def test_documented_advanced_configuration_is_loadable() -> None:
    """The published example stays synchronized with the configuration schema."""
    config = CrawlerConfig.from_file("examples/advanced_config.json")

    assert config.start_urls == ("https://example.com/",)
    assert config.max_pages == 100
    assert config.storage is not None
    assert config.storage.format == "json"
    assert config.reports.html is not None
