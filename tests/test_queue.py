"""Tests for crawl target queueing and lifecycle statistics."""

from __future__ import annotations

import pytest

from crawlforge import CrawlerQueue


@pytest.mark.asyncio
async def test_queue_returns_highest_priority_then_insertion_order() -> None:
    """Higher priorities win while equal priorities remain FIFO."""
    queue = CrawlerQueue()
    queue.add_url("https://example.com/low", priority=-1)
    queue.add_url("https://example.com/high-first", priority=5)
    queue.add_url("https://example.com/high-second", priority=5)

    assert await queue.get_next() == "https://example.com/high-first"
    assert await queue.get_next() == "https://example.com/high-second"
    assert await queue.get_next() == "https://example.com/low"
    assert await queue.get_next() is None


@pytest.mark.asyncio
async def test_queue_deduplicates_urls_across_lifecycle_states() -> None:
    """Queued, active, processed, and failed URLs cannot be added twice."""
    queue = CrawlerQueue()
    url = "https://example.com/page"

    queue.add_url(url)
    queue.add_url(url, priority=10)
    assert queue.get_stats()["queued"] == 1

    assert await queue.get_next() == url
    queue.add_url(url)
    assert queue.get_stats()["active"] == 1

    queue.mark_processed(url)
    queue.add_url(url)
    assert queue.get_stats() == {
        "queued": 0,
        "active": 0,
        "processed": 1,
        "failed": 0,
        "total": 1,
    }


@pytest.mark.asyncio
async def test_queue_records_failed_url_and_error_count() -> None:
    """A failed active URL moves into the failed lifecycle state."""
    queue = CrawlerQueue()
    url = "https://example.com/failure"
    queue.add_url(url)

    assert await queue.get_next() == url
    queue.mark_failed(url, "HTTP 500")

    assert queue.get_stats() == {
        "queued": 0,
        "active": 0,
        "processed": 0,
        "failed": 1,
        "total": 1,
    }


@pytest.mark.asyncio
async def test_queue_can_defer_active_url_without_duplication() -> None:
    """Scheduler deferral returns an active URL to its priority position."""
    queue = CrawlerQueue()
    url = "https://example.com/deferred"
    queue.add_url(url, priority=3)

    assert await queue.get_next() == url
    queue.defer_url(url, priority=3)

    assert queue.get_stats()["queued"] == 1
    assert queue.get_stats()["active"] == 0
    assert await queue.get_next() == url
