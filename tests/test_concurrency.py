"""Tests for global and per-domain semaphore management."""

from __future__ import annotations

import asyncio

import pytest

from crawlforge import SemaphoreManager


@pytest.mark.asyncio
async def test_manager_enforces_global_and_per_domain_limits() -> None:
    """A saturated domain leaves global capacity available to another domain."""
    manager = SemaphoreManager(max_concurrent=2, max_concurrent_per_domain=1)
    release = asyncio.Event()
    first_entered = asyncio.Event()
    other_entered = asyncio.Event()
    same_domain_entries = 0

    async def hold(url: str, entered: asyncio.Event | None = None) -> None:
        nonlocal same_domain_entries
        async with manager.limit(url):
            if urlsplit_domain(url) == "one.example":
                same_domain_entries += 1
            if entered is not None:
                entered.set()
            await release.wait()

    first = asyncio.create_task(
        hold("https://one.example/first", first_entered),
    )
    await asyncio.wait_for(first_entered.wait(), timeout=5)
    same_domain_waiter = asyncio.create_task(hold("https://one.example/second"))
    other = asyncio.create_task(
        hold("https://two.example/page", other_entered),
    )
    await asyncio.wait_for(other_entered.wait(), timeout=5)

    assert same_domain_entries == 1
    assert manager.get_stats() == {
        "active_tasks": 2,
        "peak_active_tasks": 2,
        "active_by_domain": {
            "one.example": 1,
            "two.example": 1,
        },
    }

    release.set()
    await asyncio.gather(first, same_domain_waiter, other)

    assert same_domain_entries == 2
    assert manager.active_tasks == 0
    assert manager.get_stats()["active_by_domain"] == {}
    assert manager.tracked_domains == 0


@pytest.mark.asyncio
async def test_domain_limit_normalizes_port_userinfo_and_trailing_dot() -> None:
    """Equivalent forms of one hostname share the same domain permit."""
    manager = SemaphoreManager(max_concurrent=2, max_concurrent_per_domain=1)
    release_first = asyncio.Event()
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()

    async def hold(url: str, entered: asyncio.Event, release: asyncio.Event) -> None:
        async with manager.limit(url):
            entered.set()
            await release.wait()

    first = asyncio.create_task(
        hold(
            "https://example.com/page",
            first_entered,
            release_first,
        )
    )
    await asyncio.wait_for(first_entered.wait(), timeout=5)
    release_second = asyncio.Event()
    second = asyncio.create_task(
        hold(
            "https://user@EXAMPLE.com.:443/other",
            second_entered,
            release_second,
        )
    )
    await asyncio.sleep(0)

    assert not second_entered.is_set()
    assert manager.get_stats()["active_by_domain"] == {"example.com": 1}

    release_first.set()
    await asyncio.wait_for(second_entered.wait(), timeout=5)
    release_second.set()
    await asyncio.gather(first, second)

    assert manager.get_stats()["peak_active_tasks"] == 1
    assert manager.tracked_domains == 0


@pytest.mark.asyncio
async def test_domain_identity_matches_unicode_transport_hosts() -> None:
    """Unicode and punycode share a permit without merging a different ASCII host."""
    manager = SemaphoreManager(max_concurrent=2, max_concurrent_per_domain=1)
    release_unicode = asyncio.Event()
    unicode_entered = asyncio.Event()
    punycode_entered = asyncio.Event()
    ascii_entered = asyncio.Event()

    async def hold(url: str, entered: asyncio.Event, release: asyncio.Event) -> None:
        async with manager.limit(url):
            entered.set()
            await release.wait()

    unicode_task = asyncio.create_task(
        hold("https://faß.de", unicode_entered, release_unicode)
    )
    await asyncio.wait_for(unicode_entered.wait(), timeout=5)
    release_punycode = asyncio.Event()
    punycode_task = asyncio.create_task(
        hold("https://xn--fa-hia.de", punycode_entered, release_punycode)
    )
    release_ascii = asyncio.Event()
    ascii_task = asyncio.create_task(
        hold("https://fass.de", ascii_entered, release_ascii)
    )
    await asyncio.wait_for(ascii_entered.wait(), timeout=5)

    assert not punycode_entered.is_set()
    assert manager.get_stats()["active_by_domain"] == {
        "xn--fa-hia.de": 1,
        "fass.de": 1,
    }

    release_unicode.set()
    await asyncio.wait_for(punycode_entered.wait(), timeout=5)
    release_punycode.set()
    release_ascii.set()
    await asyncio.gather(unicode_task, punycode_task, ascii_task)

    assert manager.tracked_domains == 0


@pytest.mark.asyncio
async def test_domain_registry_is_released_after_hostname_churn() -> None:
    """Completed requests do not leave per-domain semaphore entries behind."""
    manager = SemaphoreManager(max_concurrent=5, max_concurrent_per_domain=1)

    for index in range(1_000):
        async with manager.limit(f"https://host-{index}.example"):
            assert manager.tracked_domains == 1

    assert manager.tracked_domains == 0


def urlsplit_domain(url: str) -> str:
    """Return the host part needed by the deterministic test assertion."""
    return url.removeprefix("https://").split("/", maxsplit=1)[0]


@pytest.mark.parametrize(
    ("global_limit", "domain_limit", "message"),
    [
        (0, None, "max_concurrent"),
        (1, 0, "max_concurrent_per_domain"),
    ],
)
def test_manager_rejects_non_positive_limits(
    global_limit: int,
    domain_limit: int | None,
    message: str,
) -> None:
    """A non-positive limit cannot create a permanently blocked manager."""
    with pytest.raises(ValueError, match=message):
        SemaphoreManager(global_limit, domain_limit)
