"""Global and per-domain concurrency management."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypedDict

from crawlforge.urls import canonical_hostname


@dataclass(slots=True)
class _DomainLimiter:
    semaphore: asyncio.Semaphore
    users: int = 0


class SemaphoreStats(TypedDict):
    """Snapshot of active request counts."""

    active_tasks: int
    peak_active_tasks: int
    active_by_domain: dict[str, int]


class SemaphoreManager:
    """Limit global and per-domain requests while tracking active work."""

    def __init__(
        self,
        max_concurrent: int,
        max_concurrent_per_domain: int | None = None,
    ) -> None:
        """Configure global and per-domain request limits."""
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")
        if max_concurrent_per_domain is not None and max_concurrent_per_domain <= 0:
            raise ValueError("max_concurrent_per_domain must be greater than zero")

        self._per_domain_limit = max_concurrent_per_domain or max_concurrent
        self._global = asyncio.Semaphore(max_concurrent)
        self._domains: dict[str, _DomainLimiter] = {}
        self._active_tasks = 0
        self._peak_active_tasks = 0
        self._active_by_domain: dict[str, int] = {}

    @property
    def max_concurrent_per_domain(self) -> int:
        """Return the configured per-domain request limit."""
        return self._per_domain_limit

    @property
    def active_tasks(self) -> int:
        """Return the number of requests currently holding permits."""
        return self._active_tasks

    @property
    def tracked_domains(self) -> int:
        """Return domain limiters currently held or awaited by requests."""
        return len(self._domains)

    @asynccontextmanager
    async def limit(self, url: str) -> AsyncIterator[None]:
        """Acquire global and URL-domain capacity for one request."""
        domain = self._domain_key(url)
        limiter = self._domains.get(domain)
        if limiter is None:
            limiter = _DomainLimiter(asyncio.Semaphore(self._per_domain_limit))
            self._domains[domain] = limiter
        limiter.users += 1

        # Domain capacity is acquired first so one saturated host cannot occupy
        # every global permit while its requests wait for the same host slot.
        try:
            async with limiter.semaphore, self._global:
                self._active_tasks += 1
                self._peak_active_tasks = max(
                    self._peak_active_tasks,
                    self._active_tasks,
                )
                self._active_by_domain[domain] = (
                    self._active_by_domain.get(domain, 0) + 1
                )
                try:
                    yield
                finally:
                    self._active_tasks -= 1
                    remaining = self._active_by_domain[domain] - 1
                    if remaining:
                        self._active_by_domain[domain] = remaining
                    else:
                        del self._active_by_domain[domain]
        finally:
            limiter.users -= 1
            if limiter.users == 0 and self._domains.get(domain) is limiter:
                del self._domains[domain]

    def get_stats(self) -> SemaphoreStats:
        """Return active and peak request counts."""
        return {
            "active_tasks": self._active_tasks,
            "peak_active_tasks": self._peak_active_tasks,
            "active_by_domain": dict(self._active_by_domain),
        }

    def _domain_key(self, url: str) -> str:
        return canonical_hostname(url) or url
