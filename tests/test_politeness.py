"""Tests for rate limiting and robots.txt policy evaluation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

import crawlforge.politeness as politeness
from crawlforge import RateLimiter, RobotsParser


@dataclass
class _FakeClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


@pytest.mark.asyncio
async def test_rate_limiter_spaces_requests_for_one_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One domain receives request slots at the configured steady rate."""
    clock = _FakeClock()
    monkeypatch.setattr(politeness, "monotonic", clock.monotonic)
    monkeypatch.setattr(politeness, "_sleep", clock.sleep)
    limiter = RateLimiter(requests_per_second=2.0)

    await limiter.acquire("example.com")
    await limiter.acquire("example.com")
    await limiter.acquire("example.com")

    assert clock.sleeps == [0.5, 0.5]
    assert clock.now == 1.0


@pytest.mark.asyncio
async def test_rate_limiter_keeps_domains_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first request for each domain receives an independent slot."""
    clock = _FakeClock()
    monkeypatch.setattr(politeness, "monotonic", clock.monotonic)
    monkeypatch.setattr(politeness, "_sleep", clock.sleep)
    limiter = RateLimiter(requests_per_second=1.0)

    await limiter.acquire("first.example")
    await limiter.acquire("second.example")

    assert clock.sleeps == []


@pytest.mark.asyncio
async def test_dynamic_minimum_interval_applies_before_current_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newly discovered crawl delay constrains the next request immediately."""
    clock = _FakeClock()
    monkeypatch.setattr(politeness, "monotonic", clock.monotonic)
    monkeypatch.setattr(politeness, "_sleep", clock.sleep)
    limiter = RateLimiter(requests_per_second=10.0)

    await limiter.acquire("example.com")
    await limiter.acquire("example.com", minimum_interval=2.0)

    assert clock.sleeps == [2.0]


@pytest.mark.asyncio
async def test_global_rate_limiter_spaces_different_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global mode shares one request schedule across all domains."""
    clock = _FakeClock()
    monkeypatch.setattr(politeness, "monotonic", clock.monotonic)
    monkeypatch.setattr(politeness, "_sleep", clock.sleep)
    limiter = RateLimiter(requests_per_second=4.0, per_domain=False)

    await limiter.acquire("first.example")
    await limiter.acquire("second.example")

    assert clock.sleeps == [0.25]


@pytest.mark.asyncio
async def test_cancelled_rate_waiter_does_not_create_a_request_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled reservation remains conservative for later requests."""
    clock = _FakeClock()
    sleep_started = asyncio.Event()
    never_release = asyncio.Event()

    async def blocked_sleep(_delay: float) -> None:
        sleep_started.set()
        await never_release.wait()

    monkeypatch.setattr(politeness, "monotonic", clock.monotonic)
    monkeypatch.setattr(politeness, "_sleep", blocked_sleep)
    limiter = RateLimiter(requests_per_second=1.0)
    await limiter.acquire("example.com")

    waiting = asyncio.create_task(limiter.acquire("example.com"))
    await asyncio.wait_for(sleep_started.wait(), timeout=1)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    monkeypatch.setattr(politeness, "_sleep", clock.sleep)
    await limiter.acquire("example.com")

    assert clock.sleeps == [1.0]


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan")])
def test_rate_limiter_rejects_invalid_rates(rate: float) -> None:
    """A request rate must describe a positive usable interval."""
    with pytest.raises(ValueError, match="requests_per_second"):
        RateLimiter(rate)


@pytest.mark.asyncio
async def test_robots_parser_applies_specific_rules_delay_and_cache() -> None:
    """Specific allow rules win and one origin is fetched only once."""
    fetched: list[str] = []

    async def fetcher(url: str) -> tuple[int, str]:
        fetched.append(url)
        return (
            200,
            """
            User-agent: *
            Disallow: /private
            Allow: /private/public
            Crawl-delay: 0.25

            User-agent: ExampleBot
            Disallow: /bot-only
            Crawl-delay: 0.5
            """,
        )

    parser = RobotsParser(fetcher)
    first = await parser.fetch_robots("https://example.com/a")
    second = await parser.fetch_robots("https://example.com/other")

    assert first["cached"] is False
    assert second["cached"] is True
    assert fetched == ["https://example.com/robots.txt"]
    assert not parser.can_fetch("https://example.com/bot-only", "ExampleBot/1.0")
    assert parser.can_fetch("https://example.com/private", "ExampleBot/1.0")
    assert parser.get_crawl_delay("ExampleBot/1.0") == 0.5
    assert parser.get_crawl_delay_for("https://example.com/a", "*") == 0.25


@pytest.mark.asyncio
async def test_robots_parser_uses_longest_rule_with_allow_tie_break() -> None:
    """Robots wildcard matches use longest specificity and allow on ties."""

    async def fetcher(_url: str) -> tuple[int, str]:
        return (
            200,
            """
            User-agent: *
            Disallow: /files/*
            Allow: /files/public$
            """,
        )

    parser = RobotsParser(fetcher)
    await parser.fetch_robots("https://example.com")

    assert parser.can_fetch("https://example.com/files/public")
    assert not parser.can_fetch("https://example.com/files/private")


@pytest.mark.asyncio
async def test_robots_parser_normalizes_percent_encoded_paths() -> None:
    """Encoded unreserved bytes cannot bypass a matching Disallow rule."""

    async def fetcher(_url: str) -> tuple[int, str]:
        return (
            200,
            """
            User-agent: *
            Disallow: /private
            Disallow: /encoded%2Fslash
            Disallow: /café
            """,
        )

    parser = RobotsParser(fetcher)
    await parser.fetch_robots("https://example.com")

    assert not parser.can_fetch("https://example.com/%70rivate")
    assert not parser.can_fetch("https://example.com/encoded%2fslash")
    assert parser.can_fetch("https://example.com/encoded/slash")
    assert not parser.can_fetch("https://example.com/café")
    assert not parser.can_fetch("https://example.com/caf%C3%A9")


@pytest.mark.asyncio
async def test_robots_cache_separates_and_coalesces_origins() -> None:
    """Origins have separate entries while concurrent same-origin reads coalesce."""
    started = asyncio.Event()
    release = asyncio.Event()
    fetched: list[str] = []

    async def fetcher(url: str) -> tuple[int, str]:
        fetched.append(url)
        started.set()
        await release.wait()
        return 404, ""

    parser = RobotsParser(fetcher)
    first = asyncio.create_task(parser.fetch_robots("http://example.com:8000/a"))
    second = asyncio.create_task(parser.fetch_robots("http://example.com:8000/b"))
    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await asyncio.gather(first, second)
    await parser.fetch_robots("http://example.com:8001/a")

    assert fetched == [
        "http://example.com:8000/robots.txt",
        "http://example.com:8001/robots.txt",
    ]


@pytest.mark.parametrize(
    ("status", "allowed"),
    [
        (404, True),
        (403, False),
        (503, False),
    ],
)
@pytest.mark.asyncio
async def test_robots_unavailable_status_policy(status: int, allowed: bool) -> None:
    """Unavailable, forbidden, and transient robots responses are explicit."""

    async def fetcher(_url: str) -> tuple[int, str]:
        return status, ""

    parser = RobotsParser(fetcher)
    await parser.fetch_robots("https://example.com")

    assert parser.can_fetch("https://example.com/page") is allowed
