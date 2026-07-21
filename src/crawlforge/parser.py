"""HTML parsing and structured data extraction."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TypedDict, TypeVar
from urllib.parse import urldefrag, urljoin, urlsplit

from bs4 import BeautifulSoup, Comment, Tag

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_IGNORED_TEXT_TAGS = frozenset({"script", "style", "template", "noscript"})


class Metadata(TypedDict):
    """Page metadata extracted from the document head."""

    title: str
    description: str
    keywords: str


class ImageData(TypedDict):
    """An image URL and its alternative text."""

    src: str
    alt: str


class HeadingData(TypedDict):
    """A heading level and normalized text."""

    level: int
    text: str


class ListData(TypedDict):
    """An ordered or unordered HTML list."""

    type: str
    items: list[str]


class ParsedPage(TypedDict):
    """Structured data extracted from one HTML page."""

    url: str
    title: str
    text: str
    links: list[str]
    metadata: Metadata
    images: list[ImageData]
    headings: list[HeadingData]
    tables: list[list[list[str]]]
    lists: list[ListData]


class HTMLParser:
    """Parse HTML documents and extract crawl-friendly structured data."""

    async def parse_html(self, html: str, url: str) -> ParsedPage:
        """Parse HTML without blocking the event loop."""
        return await asyncio.to_thread(self._parse_html, html, url)

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Return unique absolute HTTP links in document order."""
        links: list[str] = []
        seen: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            normalized = self._normalize_url(href, base_url)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            links.append(normalized)

        return links

    def extract_text(
        self,
        soup: BeautifulSoup,
        selector: str | None = None,
    ) -> str:
        """Return normalized visible text, optionally below a CSS selector."""
        root: BeautifulSoup | Tag | None
        root = soup.select_one(selector) if selector is not None else soup.body or soup
        if root is None:
            return ""
        return self._normalized_text(root)

    def extract_metadata(self, soup: BeautifulSoup) -> Metadata:
        """Return title, description, and keywords metadata."""
        metadata: Metadata = {
            "title": "",
            "description": "",
            "keywords": "",
        }
        if soup.title is not None:
            metadata["title"] = soup.title.get_text(" ", strip=True)

        for meta in soup.find_all("meta"):
            name = meta.get("name")
            if not isinstance(name, str):
                continue
            key = name.strip().casefold()
            if key not in {"description", "keywords"}:
                continue
            content = meta.get("content")
            if isinstance(content, str):
                normalized = " ".join(content.split())
                if key == "description":
                    metadata["description"] = normalized
                else:
                    metadata["keywords"] = normalized

        return metadata

    def extract_images(
        self,
        soup: BeautifulSoup,
        base_url: str,
    ) -> list[ImageData]:
        """Return images with absolute HTTP sources and alternative text."""
        images: list[ImageData] = []
        for image in soup.find_all("img", src=True):
            src = image.get("src")
            if not isinstance(src, str):
                continue
            normalized = self._normalize_url(src, base_url)
            if normalized is None:
                continue
            alt = image.get("alt")
            images.append(
                {
                    "src": normalized,
                    "alt": " ".join(alt.split()) if isinstance(alt, str) else "",
                }
            )
        return images

    def extract_headings(self, soup: BeautifulSoup) -> list[HeadingData]:
        """Return h1, h2, and h3 elements in document order."""
        headings: list[HeadingData] = []
        for heading in soup.find_all(["h1", "h2", "h3"]):
            headings.append(
                {
                    "level": int(heading.name[1]),
                    "text": heading.get_text(" ", strip=True),
                }
            )
        return headings

    def extract_tables(self, soup: BeautifulSoup) -> list[list[list[str]]]:
        """Return table rows as normalized cell values."""
        tables: list[list[list[str]]] = []
        for table in soup.find_all("table"):
            rows: list[list[str]] = []
            for row in table.find_all("tr"):
                if row.find_parent("table") is not table:
                    continue
                cells = [
                    self._normalized_text(cell, table=table)
                    for cell in row.find_all(["th", "td"], recursive=False)
                ]
                if cells:
                    rows.append(cells)
            tables.append(rows)
        return tables

    def extract_lists(self, soup: BeautifulSoup) -> list[ListData]:
        """Return ordered and unordered lists with their direct items."""
        lists: list[ListData] = []
        for html_list in soup.find_all(["ul", "ol"]):
            lists.append(
                {
                    "type": html_list.name,
                    "items": [
                        self._normalized_text(item, html_list=html_list)
                        for item in html_list.find_all("li", recursive=False)
                    ],
                }
            )
        return lists

    def _parse_html(self, html: str, url: str) -> ParsedPage:
        result = self._empty_page(url)
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as error:
            logger.warning(
                "Could not parse HTML from %s: %s (%s)",
                url,
                error,
                type(error).__name__,
            )
            return result

        empty_metadata: Metadata = {
            "title": "",
            "description": "",
            "keywords": "",
        }
        metadata = self._extract_safely(
            "metadata",
            url,
            lambda: self.extract_metadata(soup),
            empty_metadata,
        )
        result["metadata"] = metadata
        result["title"] = metadata["title"]
        result["text"] = self._extract_safely(
            "text",
            url,
            lambda: self.extract_text(soup),
            "",
        )
        result["links"] = self._extract_safely(
            "links",
            url,
            lambda: self.extract_links(soup, url),
            [],
        )
        result["images"] = self._extract_safely(
            "images",
            url,
            lambda: self.extract_images(soup, url),
            [],
        )
        result["headings"] = self._extract_safely(
            "headings",
            url,
            lambda: self.extract_headings(soup),
            [],
        )
        result["tables"] = self._extract_safely(
            "tables",
            url,
            lambda: self.extract_tables(soup),
            [],
        )
        result["lists"] = self._extract_safely(
            "lists",
            url,
            lambda: self.extract_lists(soup),
            [],
        )
        return result

    def _extract_safely(
        self,
        field: str,
        url: str,
        extract: Callable[[], _T],
        default: _T,
    ) -> _T:
        try:
            return extract()
        except Exception as error:
            logger.warning(
                "Could not extract %s from %s: %s (%s)",
                field,
                url,
                error,
                type(error).__name__,
            )
            return default

    def _normalize_url(self, value: str, base_url: str) -> str | None:
        candidate = value.strip()
        if not candidate:
            return None

        try:
            absolute, _fragment = urldefrag(urljoin(base_url, candidate))
            parsed = urlsplit(absolute)
            if (
                parsed.scheme.casefold() not in {"http", "https"}
                or parsed.hostname is None
                or any(character.isspace() for character in absolute)
                or "\\" in parsed.netloc
            ):
                return None
            _port = parsed.port
        except ValueError:
            return None
        return absolute

    def _normalized_text(
        self,
        root: BeautifulSoup | Tag,
        *,
        table: Tag | None = None,
        html_list: Tag | None = None,
    ) -> str:
        parts: list[str] = []
        for value in root.find_all(string=True):
            if isinstance(value, Comment):
                continue
            if any(
                isinstance(parent, Tag) and parent.name in _IGNORED_TEXT_TAGS
                for parent in value.parents
            ):
                continue
            if table is not None and value.find_parent("table") is not table:
                continue
            if (
                html_list is not None
                and value.find_parent(["ul", "ol"]) is not html_list
            ):
                continue
            normalized = " ".join(str(value).split())
            if normalized:
                parts.append(normalized)
        return " ".join(parts)

    def _empty_page(self, url: str) -> ParsedPage:
        return {
            "url": url,
            "title": "",
            "text": "",
            "links": [],
            "metadata": {"title": "", "description": "", "keywords": ""},
            "images": [],
            "headings": [],
            "tables": [],
            "lists": [],
        }
