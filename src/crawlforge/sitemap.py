"""Asynchronous sitemap fetching and parsing."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from types import TracebackType
from urllib.parse import urlsplit
from xml.etree import ElementTree

import aiohttp

from crawlforge.urls import canonical_hostname

type SitemapFetcher = Callable[[str], Awaitable[str | bytes]]

_DEFAULT_MAX_DEPTH = 10
_DEFAULT_MAX_SITEMAPS = 100
_DEFAULT_MAX_URLS = 50_000
_DEFAULT_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_XML_DECLARATION = re.compile(
    r"^\ufeff?<\?xml[ \t\r\n]+version[ \t\r\n]*=[ \t\r\n]*"
    r"(?P<version_quote>['\"])1\.[0-9]+(?P=version_quote)"
    r"(?P<encoding>[ \t\r\n]+encoding[ \t\r\n]*=[ \t\r\n]*"
    r"(?P<encoding_quote>['\"])[A-Za-z][A-Za-z0-9._-]*"
    r"(?P=encoding_quote))?"
    r"(?:[ \t\r\n]+standalone[ \t\r\n]*=[ \t\r\n]*"
    r"(?P<standalone_quote>['\"])(?:yes|no)(?P=standalone_quote))?"
    r"[ \t\r\n]*\?>"
)


class SitemapParser:
    """Fetch regular and indexed XML sitemaps within configured limits."""

    def __init__(
        self,
        fetcher: SitemapFetcher | None = None,
        *,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        max_sitemaps: int = _DEFAULT_MAX_SITEMAPS,
        max_urls: int = _DEFAULT_MAX_URLS,
        max_document_bytes: int = _DEFAULT_MAX_DOCUMENT_BYTES,
    ) -> None:
        """Configure the sitemap transport and traversal boundaries."""
        if max_depth < 0:
            raise ValueError("max_depth must be zero or greater")
        if max_sitemaps <= 0:
            raise ValueError("max_sitemaps must be greater than zero")
        if max_urls <= 0:
            raise ValueError("max_urls must be greater than zero")
        if max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be greater than zero")

        self._fetcher = fetcher
        self._max_depth = max_depth
        self._max_sitemaps = max_sitemaps
        self._max_urls = max_urls
        self._max_document_bytes = max_document_bytes
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._closed = False

    async def __aenter__(self) -> SitemapParser:
        """Enter the parser's asynchronous resource context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the owned HTTP session."""
        await self.close()

    async def fetch_sitemap(self, sitemap_url: str) -> list[str]:
        """Fetch a sitemap tree and return unique page URLs in document order."""
        self._validate_url(sitemap_url, label="sitemap URL")
        if self._closed:
            raise RuntimeError("SitemapParser is closed")

        seen_sitemaps: set[str] = set()
        seen_urls: set[str] = set()
        urls: list[str] = []
        await self._visit_sitemap(
            sitemap_url,
            depth=0,
            seen_sitemaps=seen_sitemaps,
            seen_urls=seen_urls,
            urls=urls,
        )
        return urls

    async def close(self) -> None:
        """Close the owned HTTP session safely and idempotently."""
        close_task = asyncio.create_task(self._close_session())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await close_task
            except Exception as close_error:
                raise cancelled from close_error
            raise

    async def _close_session(self) -> None:
        async with self._session_lock:
            self._closed = True
            session = self._session
            if session is None:
                return
            try:
                if not session.closed:
                    await session.close()
            finally:
                if session.closed:
                    self._session = None

    async def _visit_sitemap(
        self,
        sitemap_url: str,
        *,
        depth: int,
        seen_sitemaps: set[str],
        seen_urls: set[str],
        urls: list[str],
    ) -> None:
        if sitemap_url in seen_sitemaps:
            return
        if depth > self._max_depth:
            raise ValueError(
                f"maximum sitemap depth ({self._max_depth}) exceeded at {sitemap_url!r}"
            )
        if len(seen_sitemaps) >= self._max_sitemaps:
            raise ValueError(f"maximum sitemap count ({self._max_sitemaps}) exceeded")
        seen_sitemaps.add(sitemap_url)

        document = await self._fetch_document(sitemap_url)
        root_type, locations = await asyncio.to_thread(
            self._parse_document,
            document,
            sitemap_url,
        )
        if root_type == "urlset":
            for location in locations:
                self._validate_url(location, label="sitemap entry URL")
                if location in seen_urls:
                    continue
                if len(urls) >= self._max_urls:
                    raise ValueError(f"maximum URL count ({self._max_urls}) exceeded")
                seen_urls.add(location)
                urls.append(location)
            return

        for location in locations:
            self._validate_url(location, label="nested sitemap URL")
            await self._visit_sitemap(
                location,
                depth=depth + 1,
                seen_sitemaps=seen_sitemaps,
                seen_urls=seen_urls,
                urls=urls,
            )

    async def _fetch_document(self, sitemap_url: str) -> str | bytes:
        if self._fetcher is not None:
            try:
                content = await self._fetcher(sitemap_url)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise RuntimeError(
                    f"could not fetch sitemap {sitemap_url!r}: {error}"
                ) from error
            if isinstance(content, str):
                try:
                    document_size = len(content.encode())
                except UnicodeError as error:
                    raise ValueError(
                        f"invalid sitemap XML at {sitemap_url!r}: {error}"
                    ) from error
            elif isinstance(content, bytes):
                document_size = len(content)
            else:
                raise RuntimeError(
                    "sitemap fetcher must return str or bytes, "
                    f"got {type(content).__name__}"
                )
            if not content:
                raise ValueError(f"empty sitemap document at {sitemap_url!r}")
            self._check_document_size(document_size, sitemap_url)
            return content

        session = await self._get_session()
        try:
            async with session.get(sitemap_url) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"could not fetch sitemap {sitemap_url!r}: "
                        f"HTTP {response.status}"
                    )
                content_length = response.content_length
                if (
                    content_length is not None
                    and content_length > self._max_document_bytes
                ):
                    raise ValueError(
                        f"sitemap document {sitemap_url!r} exceeds "
                        f"max_document_bytes ({self._max_document_bytes})"
                    )

                buffer = bytearray()
                async for chunk in response.content.iter_chunked(_READ_CHUNK_BYTES):
                    buffer.extend(chunk)
                    if len(buffer) > self._max_document_bytes:
                        raise ValueError(
                            f"sitemap document {sitemap_url!r} exceeds "
                            f"max_document_bytes ({self._max_document_bytes})"
                        )
                return bytes(buffer)
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, TimeoutError) as error:
            raise RuntimeError(
                f"could not fetch sitemap {sitemap_url!r}: {error}"
            ) from error

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._closed:
                raise RuntimeError("SitemapParser is closed")
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30.0),
                )
            return self._session

    def _check_document_size(self, document_size: int, sitemap_url: str) -> None:
        if document_size > self._max_document_bytes:
            raise ValueError(
                f"sitemap document {sitemap_url!r} exceeds "
                f"max_document_bytes ({self._max_document_bytes})"
            )

    @staticmethod
    def _parse_document(
        document: str | bytes,
        sitemap_url: str,
    ) -> tuple[str, list[str]]:
        if isinstance(document, str):
            has_declaration = any(
                marker in document.upper() for marker in ("<!DOCTYPE", "<!ENTITY")
            )
        else:
            has_declaration = any(
                marker in document.upper() for marker in (b"<!DOCTYPE", b"<!ENTITY")
            )
        if has_declaration:
            raise ValueError(
                f"invalid sitemap XML at {sitemap_url!r}: "
                "document type declarations are not supported"
            )
        parse_input = document
        if isinstance(document, str):
            declaration = _XML_DECLARATION.match(document)
            if declaration is not None and declaration["encoding"] is not None:
                encoding_start, encoding_end = declaration.span("encoding")
                parse_input = document[:encoding_start] + document[encoding_end:]
        try:
            root = ElementTree.fromstring(parse_input)
        except (ElementTree.ParseError, LookupError, UnicodeError, ValueError) as error:
            raise ValueError(
                f"invalid sitemap XML at {sitemap_url!r}: {error}"
            ) from error

        root_type = SitemapParser._local_name(root.tag)
        if root_type not in {"urlset", "sitemapindex"}:
            raise ValueError(
                f"invalid sitemap root at {sitemap_url!r}: "
                f"expected urlset or sitemapindex, got {root_type!r}"
            )

        entry_name = "url" if root_type == "urlset" else "sitemap"
        locations: list[str] = []
        for entry in root:
            if SitemapParser._local_name(entry.tag) != entry_name:
                continue
            location = next(
                (
                    child.text.strip()
                    for child in entry
                    if SitemapParser._local_name(child.tag) == "loc"
                    and child.text is not None
                    and child.text.strip()
                ),
                None,
            )
            if location is None:
                raise ValueError(
                    f"invalid {root_type} entry at {sitemap_url!r}: missing loc"
                )
            locations.append(location)
        return root_type, locations

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    @staticmethod
    def _validate_url(url: str, *, label: str) -> None:
        if not isinstance(url, str) or not url:
            raise ValueError(f"{label} must be an absolute HTTP or HTTPS URL")
        try:
            parsed = urlsplit(url)
            _ = parsed.port
        except (TypeError, ValueError):
            parsed = None
        if (
            parsed is None
            or parsed.scheme.casefold() not in {"http", "https"}
            or canonical_hostname(url) is None
            or any(character.isspace() for character in url)
            or "\\" in parsed.netloc
        ):
            raise ValueError(f"{label} must be an absolute HTTP or HTTPS URL: {url!r}")
