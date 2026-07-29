"""Reusable URL and resolved-address policy for outbound HTTP requests."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.resolver import DefaultResolver

from crawlforge.urls import canonical_hostname


class URLPolicyError(RuntimeError):
    """Raised when an outbound URL violates the configured network policy."""


@dataclass(frozen=True, slots=True)
class URLNetworkPolicy:
    """Restrict outbound HTTP targets by scheme, hostname, and resolved address."""

    allow_private_networks: bool = False
    allowed_domains: tuple[str, ...] = ()
    _normalized_domains: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(
                {_normalize_allowed_domain(domain) for domain in self.allowed_domains}
            )
        )
        object.__setattr__(self, "_normalized_domains", normalized)

    def validate_url(self, url: str) -> str:
        """Validate one URL before an outbound request and return its hostname."""
        if (
            not url
            or url != url.strip()
            or any(character.isspace() for character in url)
        ):
            raise URLPolicyError("URL must be a non-empty value without whitespace")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as error:
            raise URLPolicyError("URL is malformed") from error
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise URLPolicyError("only http and https URLs are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise URLPolicyError("URL user information is not allowed")
        if "\\" in parsed.netloc:
            raise URLPolicyError("URL authority is malformed")
        hostname = canonical_hostname(url)
        if hostname is None:
            raise URLPolicyError("URL must contain a valid hostname")
        if port is not None and not 1 <= port <= 65535:
            raise URLPolicyError("URL port must be between 1 and 65535")
        self.validate_hostname(hostname)
        return hostname

    def validate_hostname(self, hostname: str) -> None:
        """Validate a canonical hostname or literal address."""
        normalized = hostname.rstrip(".").casefold()
        if normalized == "localhost" or normalized.endswith(".localhost"):
            if not self.allow_private_networks:
                raise URLPolicyError("localhost is blocked by the network policy")

        if self._normalized_domains and not any(
            _hostname_matches_allowed_domain(normalized, allowed)
            for allowed in self._normalized_domains
        ):
            raise URLPolicyError("hostname is not in the server allowlist")

        try:
            address = ipaddress.ip_address(normalized)
        except ValueError:
            return
        self.validate_address(address)

    def validate_resolved_addresses(
        self,
        hostname: str,
        addresses: list[str] | tuple[str, ...],
    ) -> None:
        """Fail closed unless every address returned for a hostname is permitted."""
        self.validate_hostname(hostname)
        if not addresses:
            raise URLPolicyError("hostname did not resolve to an address")
        for value in addresses:
            try:
                address = ipaddress.ip_address(value)
            except ValueError as error:
                raise URLPolicyError(
                    "hostname resolved to an invalid address"
                ) from error
            self.validate_address(address)

    def validate_address(
        self,
        address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> None:
        """Validate one literal or resolved IP address."""
        mapped = (
            address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) else None
        )
        if mapped is not None:
            self.validate_address(mapped)
            return

        if address.is_multicast or address.is_unspecified:
            raise URLPolicyError("non-routable network addresses are blocked")
        if address.is_loopback:
            if self.allow_private_networks:
                return
            raise URLPolicyError("private and non-public network addresses are blocked")
        if isinstance(address, ipaddress.IPv6Address) and address.is_site_local:
            if self.allow_private_networks:
                return
            raise URLPolicyError("private and non-public network addresses are blocked")
        if address.is_reserved:
            raise URLPolicyError("non-routable network addresses are blocked")
        if address.is_global:
            return
        if self.allow_private_networks and (
            address.is_private or address.is_link_local
        ):
            return
        raise URLPolicyError("private and non-public network addresses are blocked")


class PolicyResolver(AbstractResolver):
    """Validate the exact DNS answers supplied to aiohttp's connector."""

    def __init__(self, policy: URLNetworkPolicy) -> None:
        self._policy = policy
        self._resolver = DefaultResolver()
        self._closed = False

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        """Resolve a hostname and reject every unsafe or mixed answer set."""
        if self._closed:
            raise RuntimeError("PolicyResolver is closed")
        self._policy.validate_hostname(host)
        results = await self._resolver.resolve(host, port, family)
        self._policy.validate_resolved_addresses(
            host,
            tuple(result["host"] for result in results),
        )
        return results

    async def close(self) -> None:
        """Close the owned system resolver; repeated calls are safe."""
        if self._closed:
            return
        self._closed = True
        await self._resolver.close()


def _normalize_allowed_domain(value: str) -> str:
    candidate = value.strip().rstrip(".").casefold()
    if (
        not candidate
        or "://" in candidate
        or "/" in candidate
        or "@" in candidate
        or any(character.isspace() for character in candidate)
    ):
        raise ValueError(f"invalid allowed domain: {value!r}")
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        wrapped = (
            f"http://[{candidate}]/" if ":" in candidate else f"http://{candidate}/"
        )
        hostname = canonical_hostname(wrapped)
        if hostname is None:
            raise ValueError(f"invalid allowed domain: {value!r}") from None
        return hostname


def _hostname_matches_allowed_domain(hostname: str, allowed: str) -> bool:
    if hostname == allowed:
        return True
    try:
        ipaddress.ip_address(allowed)
    except ValueError:
        return hostname.endswith(f".{allowed}")
    return False
