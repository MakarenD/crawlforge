"""Rate limiting and robots.txt policy support."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import TypedDict
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from crawlforge.errors import NetworkError, PermanentError, TransientError
from crawlforge.urls import canonical_hostname

_sleep = asyncio.sleep
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


class RobotsGroupData(TypedDict):
    """Serializable rules for one robots.txt user-agent group."""

    user_agents: list[str]
    allow: list[str]
    disallow: list[str]
    crawl_delay: float | None


class RobotsData(TypedDict):
    """Serializable cached robots.txt result for one origin."""

    origin: str
    robots_url: str
    status: int | None
    cached: bool
    groups: list[RobotsGroupData]


@dataclass(frozen=True, slots=True)
class _RobotsGroup:
    user_agents: tuple[str, ...]
    allow: tuple[str, ...]
    disallow: tuple[str, ...]
    crawl_delay: float | None


@dataclass(frozen=True, slots=True)
class _RobotsEntry:
    origin: str
    robots_url: str
    status: int | None
    groups: tuple[_RobotsGroup, ...]


@dataclass(slots=True)
class _GroupBuilder:
    user_agents: list[str]
    allow: list[str]
    disallow: list[str]
    crawl_delay: float | None = None
    has_directives: bool = False


@dataclass(slots=True)
class _RateState:
    lock: asyncio.Lock
    last_started: float | None = None


type RobotsFetcher = Callable[[str], Awaitable[tuple[int, str]]]


class RateLimiter:
    """Space request starts globally or independently for each domain."""

    def __init__(
        self,
        requests_per_second: float = 1.0,
        per_domain: bool = True,
    ) -> None:
        """Configure the steady request rate and limiting scope."""
        if math.isnan(requests_per_second) or requests_per_second <= 0:
            raise ValueError("requests_per_second must be greater than zero")

        self._interval = 1.0 / requests_per_second
        self._per_domain = per_domain
        self._states: dict[str, _RateState] = {}

    async def acquire(
        self,
        domain: str | None = None,
        *,
        minimum_interval: float = 0.0,
    ) -> None:
        """Wait until the next request slot for the selected limiting scope."""
        if not math.isfinite(minimum_interval) or minimum_interval < 0:
            raise ValueError("minimum_interval must be a finite non-negative value")

        key = self._key(domain)
        interval = max(self._interval, minimum_interval)
        state = self._states.setdefault(key, _RateState(asyncio.Lock()))
        async with state.lock:
            now = monotonic()
            if state.last_started is not None:
                delay = state.last_started + interval - now
                if delay > 0:
                    await _sleep(delay)
            state.last_started = monotonic()

    def _key(self, domain: str | None) -> str:
        if not self._per_domain:
            return "*"
        if domain is None:
            return ""
        return domain.strip().casefold().rstrip(".")


class RobotsParser:
    """Fetch, cache, and evaluate robots.txt rules for multiple origins."""

    def __init__(
        self,
        fetcher: RobotsFetcher | None = None,
        *,
        request_user_agent: str = "CrawlForge/0.1",
    ) -> None:
        """Configure the asynchronous robots.txt transport."""
        if not request_user_agent.strip():
            raise ValueError("request_user_agent must not be empty")
        self._fetcher = fetcher or self._default_fetch
        self._request_user_agent = request_user_agent
        self._cache: dict[str, _RobotsEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_origin: str | None = None

    async def fetch_robots(self, base_url: str) -> RobotsData:
        """Fetch and cache the robots.txt policy for a URL origin."""
        origin, robots_url = self._origin_and_robots_url(base_url)
        cached = self._cache.get(origin)
        if cached is not None:
            self._last_origin = origin
            return self._snapshot(cached, cached=True)

        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            cached = self._cache.get(origin)
            if cached is not None:
                self._last_origin = origin
                return self._snapshot(cached, cached=True)

            status: int | None
            groups: tuple[_RobotsGroup, ...]
            try:
                status, content = await self._fetcher(robots_url)
            except (
                TimeoutError,
                aiohttp.ClientError,
                TransientError,
                NetworkError,
                PermanentError,
            ):
                status = None
                groups = (self._deny_all_group(),)
            else:
                if 200 <= status < 300:
                    groups = self._parse(content)
                elif status in {401, 403} or status >= 500:
                    groups = (self._deny_all_group(),)
                else:
                    groups = ()

            entry = _RobotsEntry(origin, robots_url, status, groups)
            self._cache[origin] = entry
            self._last_origin = origin
            return self._snapshot(entry, cached=False)

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Return whether cached rules permit a user agent to fetch a URL."""
        origin, _robots_url = self._origin_and_robots_url(url)
        entry = self._cache.get(origin)
        if entry is None:
            return True

        groups = self._matching_groups(entry.groups, user_agent)
        if not groups:
            return True

        parsed = urlsplit(url)
        target = self._normalize_percent_encoding(parsed.path or "/")
        if parsed.query:
            target = f"{target}?{self._normalize_percent_encoding(parsed.query)}"

        matches: list[tuple[int, bool]] = []
        for group in groups:
            matches.extend(
                (self._specificity(pattern), True)
                for pattern in group.allow
                if self._matches(pattern, target)
            )
            matches.extend(
                (self._specificity(pattern), False)
                for pattern in group.disallow
                if pattern and self._matches(pattern, target)
            )
        if not matches:
            return True

        _length, allowed = max(matches, key=lambda match: (match[0], match[1]))
        return allowed

    def get_crawl_delay(self, user_agent: str = "*") -> float:
        """Return the crawl delay for the most recently fetched origin."""
        if self._last_origin is None:
            return 0.0
        entry = self._cache[self._last_origin]
        return self._delay_for_groups(entry.groups, user_agent)

    def get_crawl_delay_for(self, url: str, user_agent: str = "*") -> float:
        """Return the cached crawl delay for a URL origin."""
        origin, _robots_url = self._origin_and_robots_url(url)
        entry = self._cache.get(origin)
        if entry is None:
            return 0.0
        return self._delay_for_groups(entry.groups, user_agent)

    async def _default_fetch(self, robots_url: str) -> tuple[int, str]:
        timeout = aiohttp.ClientTimeout(total=30.0)
        headers = {"User-Agent": self._request_user_agent}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(robots_url) as response:
                return response.status, await response.text(errors="replace")

    def _origin_and_robots_url(self, url: str) -> tuple[str, str]:
        parsed = urlsplit(url)
        hostname = canonical_hostname(url)
        if parsed.scheme.casefold() not in {"http", "https"} or hostname is None:
            raise ValueError(f"invalid HTTP URL: {url!r}")

        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError(f"invalid HTTP URL: {url!r}") from error

        default_port = 80 if parsed.scheme.casefold() == "http" else 443
        effective_port = port or default_port
        host_for_url = f"[{hostname}]" if ":" in hostname else hostname
        netloc = (
            host_for_url
            if effective_port == default_port
            else f"{host_for_url}:{effective_port}"
        )
        scheme = parsed.scheme.casefold()
        origin = f"{scheme}://{hostname}:{effective_port}"
        return origin, urlunsplit((scheme, netloc, "/robots.txt", "", ""))

    def _parse(self, content: str) -> tuple[_RobotsGroup, ...]:
        builders: list[_GroupBuilder] = []
        current: _GroupBuilder | None = None

        for raw_line in content.splitlines():
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if not line or ":" not in line:
                continue
            field, value = (part.strip() for part in line.split(":", maxsplit=1))
            field = field.casefold()

            if field == "user-agent":
                if not value:
                    continue
                if current is None or current.has_directives:
                    current = _GroupBuilder([], [], [])
                    builders.append(current)
                current.user_agents.append(value.casefold())
                continue

            if current is None or not current.user_agents:
                continue
            if field == "allow":
                current.allow.append(value)
                current.has_directives = True
            elif field == "disallow":
                current.disallow.append(value)
                current.has_directives = True
            elif field == "crawl-delay":
                try:
                    delay = float(value)
                except ValueError:
                    continue
                if math.isfinite(delay) and delay >= 0:
                    current.crawl_delay = delay
                    current.has_directives = True

        return tuple(
            _RobotsGroup(
                tuple(builder.user_agents),
                tuple(builder.allow),
                tuple(builder.disallow),
                builder.crawl_delay,
            )
            for builder in builders
        )

    def _matching_groups(
        self,
        groups: Sequence[_RobotsGroup],
        user_agent: str,
    ) -> tuple[_RobotsGroup, ...]:
        normalized = user_agent.casefold()
        ranked: list[tuple[int, _RobotsGroup]] = []
        for group in groups:
            matches = [
                0 if agent == "*" else len(agent)
                for agent in group.user_agents
                if agent == "*" or agent in normalized
            ]
            if matches:
                ranked.append((max(matches), group))
        if not ranked:
            return ()
        best = max(rank for rank, _group in ranked)
        return tuple(group for rank, group in ranked if rank == best)

    def _delay_for_groups(
        self,
        groups: Sequence[_RobotsGroup],
        user_agent: str,
    ) -> float:
        delays = [
            group.crawl_delay
            for group in self._matching_groups(groups, user_agent)
            if group.crawl_delay is not None
        ]
        return max(delays, default=0.0)

    def _matches(self, pattern: str, target: str) -> bool:
        pattern = self._normalize_percent_encoding(pattern)
        anchored = pattern.endswith("$")
        body = pattern[:-1] if anchored else pattern
        expression = re.escape(body).replace(r"\*", ".*")
        suffix = "$" if anchored else ""
        return re.match(f"^{expression}{suffix}", target) is not None

    def _specificity(self, pattern: str) -> int:
        pattern = self._normalize_percent_encoding(pattern)
        return len(pattern.replace("*", "").removesuffix("$"))

    def _normalize_percent_encoding(self, value: str) -> str:
        def replace(match: re.Match[str]) -> str:
            character = chr(int(match.group(1), 16))
            if character in _UNRESERVED:
                return character
            return f"%{match.group(1).upper()}"

        normalized = _PERCENT_ESCAPE.sub(replace, value)
        return "".join(
            character
            if character.isascii()
            else "".join(f"%{byte:02X}" for byte in character.encode())
            for character in normalized
        )

    def _deny_all_group(self) -> _RobotsGroup:
        return _RobotsGroup(("*",), (), ("/",), None)

    def _snapshot(self, entry: _RobotsEntry, *, cached: bool) -> RobotsData:
        return {
            "origin": entry.origin,
            "robots_url": entry.robots_url,
            "status": entry.status,
            "cached": cached,
            "groups": [
                {
                    "user_agents": list(group.user_agents),
                    "allow": list(group.allow),
                    "disallow": list(group.disallow),
                    "crawl_delay": group.crawl_delay,
                }
                for group in entry.groups
            ],
        }
