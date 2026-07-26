"""URL identity helpers shared by crawling and concurrency controls."""

from __future__ import annotations

import ipaddress
import re
from typing import Protocol, cast

import aiohttp.client

_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


class _TransportURL(Protocol):
    raw_host: str | None


class _URLFactory(Protocol):
    def __call__(self, value: str) -> _TransportURL: ...


# Reuse aiohttp's URL factory so hostname identity exactly matches its transport
# without exposing aiohttp's required yarl dependency as part of our public API.
_transport_url = cast(_URLFactory, vars(aiohttp.client)["URL"])


def canonical_hostname(url: str) -> str | None:
    """Return aiohttp-compatible ASCII hostname identity for a valid host."""
    try:
        raw_host = _transport_url(url).raw_host
    except (TypeError, ValueError, UnicodeError):
        return None
    if raw_host is None:
        return None

    hostname = raw_host.rstrip(".").casefold()
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        labels = hostname.split(".")
        if (
            not hostname
            or len(hostname) > 253
            or any(_DNS_LABEL.fullmatch(label) is None for label in labels)
        ):
            return None
    return hostname
