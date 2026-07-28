"""Tests for advanced crawl statistics and reports."""

from __future__ import annotations

import json
from pathlib import Path

from crawlforge.statistics import CrawlerStats, render_html_report


def test_crawler_stats_tracks_speed_eta_statuses_and_domains() -> None:
    """A deterministic clock produces exact live and final metrics."""
    now = 10.0
    stats = CrawlerStats(clock=lambda: now)
    stats.reset(target_pages=4)
    now = 12.0
    stats.record_page(
        "https://example.com/one",
        successful=True,
        status_code=200,
    )
    stats.record_page(
        "https://example.com/missing",
        successful=False,
        status_code=404,
    )

    live = stats.get_stats(active_tasks=2, queued_pages=3)

    assert live == {
        "total_pages": 2,
        "successful": 1,
        "failed": 1,
        "average_speed": 1.0,
        "status_codes": {"200": 1, "404": 1},
        "top_domains": [{"domain": "example.com", "pages": 2}],
        "elapsed_seconds": 2.0,
        "progress_percent": 50.0,
        "estimated_remaining_seconds": 2.0,
        "active_tasks": 2,
        "queued_pages": 3,
    }

    now = 14.0
    stats.finish()
    final = stats.get_stats()
    assert final["elapsed_seconds"] == 4.0
    assert final["progress_percent"] == 100.0
    assert final["estimated_remaining_seconds"] is None


def test_empty_statistics_have_zero_rate_and_unknown_eta() -> None:
    """An idle collector never emits divisions by zero or non-finite values."""
    stats = CrawlerStats(clock=lambda: 5.0)
    stats.reset(target_pages=100)

    snapshot = stats.get_stats()

    assert snapshot["average_speed"] == 0.0
    assert snapshot["progress_percent"] == 0.0
    assert snapshot["estimated_remaining_seconds"] is None


def test_statistics_export_valid_json_and_standalone_html(tmp_path: Path) -> None:
    """Both exporters create complete dependency-free report files."""
    stats = CrawlerStats(clock=lambda: 1.0)
    stats.reset(target_pages=1)
    stats.record_page(
        "https://example.com/",
        successful=True,
        status_code=200,
    )
    stats.finish()
    json_path = tmp_path / "stats.json"
    html_path = tmp_path / "stats.html"

    stats.export_to_json(json_path)
    stats.export_to_html_report(html_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["successful"] == 1
    html = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "Status codes" in html
    assert 'class="bar"' in html


def test_html_report_escapes_page_and_failure_values() -> None:
    """Report tables cannot turn crawled content into executable markup."""
    html = render_html_report(
        {
            "total_pages": 2,
            "successful": 1,
            "failed": 1,
            "average_speed": 2.0,
            "elapsed_seconds": 1.0,
            "status_codes": {"200": 1, "500": 1},
            "top_domains": [{"domain": "<script>", "pages": 2}],
        },
        pages=[{"url": "https://example.com/?x=<tag>", "title": "<script>"}],
        failed_urls={"https://bad.invalid/": "<img src=x onerror=alert(1)>"},
    )

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
