"""Priority queue and lifecycle accounting for crawl targets."""

from __future__ import annotations

import heapq
from itertools import count
from typing import TypedDict


class QueueStats(TypedDict):
    """Snapshot of URL queue lifecycle counts."""

    queued: int
    active: int
    processed: int
    failed: int
    total: int


class CrawlerQueue:
    """Manage unique crawl targets in priority and insertion order."""

    def __init__(self) -> None:
        """Create an empty URL queue."""
        self._items: list[tuple[int, int, str]] = []
        self._sequence = count()
        self._queued: set[str] = set()
        self._active: set[str] = set()
        self._processed: set[str] = set()
        self._failed: dict[str, str] = {}

    def add_url(self, url: str, priority: int = 0) -> None:
        """Add a URL unless it is already known to this queue."""
        if (
            url in self._queued
            or url in self._active
            or url in self._processed
            or url in self._failed
        ):
            return

        heapq.heappush(self._items, (-priority, next(self._sequence), url))
        self._queued.add(url)

    async def get_next(self) -> str | None:
        """Return the highest-priority URL, or ``None`` when the queue is empty."""
        if not self._items:
            return None

        _priority, _sequence, url = heapq.heappop(self._items)
        self._queued.remove(url)
        self._active.add(url)
        return url

    def defer_url(self, url: str, priority: int = 0) -> None:
        """Return an active URL to the queue without changing its lifecycle."""
        if url not in self._active:
            return
        self._active.remove(url)
        heapq.heappush(self._items, (-priority, next(self._sequence), url))
        self._queued.add(url)

    def mark_processed(self, url: str) -> None:
        """Record that an active URL completed successfully."""
        if url not in self._active:
            return
        self._active.discard(url)
        self._processed.add(url)
        self._failed.pop(url, None)

    def mark_failed(self, url: str, error: str) -> None:
        """Record that an active URL failed with the supplied error."""
        if url not in self._active:
            return
        self._active.discard(url)
        self._processed.discard(url)
        self._failed[url] = error

    def get_stats(self) -> QueueStats:
        """Return current queue, active, processed, and failed counts."""
        queued = len(self._queued)
        active = len(self._active)
        processed = len(self._processed)
        failed = len(self._failed)
        return {
            "queued": queued,
            "active": active,
            "processed": processed,
            "failed": failed,
            "total": queued + active + processed + failed,
        }
