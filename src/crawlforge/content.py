"""Deterministic conversion of fetched HTML into clean source documents."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from datetime import datetime
from urllib.parse import SplitResult, urldefrag, urljoin, urlsplit

from bs4 import BeautifulSoup, Comment, Tag
from bs4.element import NavigableString

from crawlforge.async_utils import run_lifecycle_owned_thread
from crawlforge.context_models import (
    BlockKind,
    DocumentBlock,
    HeuristicTokenEstimator,
    SourceDocument,
    TokenEstimator,
)

logger = logging.getLogger(__name__)

_REMOVED_TAGS = frozenset({"script", "style", "noscript", "template"})
_NAVIGATION_NAMES = frozenset({"footer", "nav"})
_NAVIGATION_ROLES = frozenset({"menu", "menubar", "navigation"})
_NAVIGATION_MARKERS = frozenset(
    {
        "footer",
        "main-menu",
        "main-nav",
        "menu",
        "nav",
        "navbar",
        "navigation",
        "primary-menu",
        "primary-nav",
        "site-footer",
        "site-menu",
        "site-nav",
    }
)
_BLOCK_TAGS = frozenset(
    {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "code",
        "ol",
        "p",
        "pre",
        "table",
        "ul",
    }
)
_INLINE_TAGS = frozenset(
    {
        "a",
        "abbr",
        "b",
        "bdi",
        "bdo",
        "cite",
        "code",
        "data",
        "del",
        "em",
        "i",
        "ins",
        "kbd",
        "mark",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
        "var",
    }
)


class ContentProcessor:
    """Clean HTML and retain its ordered semantic block structure."""

    def __init__(self, estimator: TokenEstimator | None = None) -> None:
        self._estimator = estimator or HeuristicTokenEstimator()

    async def process_html(
        self,
        html: str,
        url: str,
        *,
        final_url: str | None = None,
        status_code: int,
        content_type: str,
        fetched_at: datetime,
        metadata: Mapping[str, object] | None = None,
    ) -> SourceDocument:
        """Process DOM work in a worker thread without blocking the event loop."""
        return await run_lifecycle_owned_thread(
            self._process_html,
            html,
            url,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            fetched_at=fetched_at,
            metadata=metadata,
        )

    def _process_html(
        self,
        html: str,
        url: str,
        *,
        final_url: str | None,
        status_code: int,
        content_type: str,
        fetched_at: datetime,
        metadata: Mapping[str, object] | None,
    ) -> SourceDocument:
        source_size_bytes = len(html.encode("utf-8"))
        source_estimated_tokens = self._estimator.count(html)
        effective_url = _validated_http_url(final_url or "")
        if effective_url is None:
            effective_url = _validated_http_url(url)

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as error:
            logger.warning(
                "Could not construct HTML parser for %s: %s (%s)",
                url,
                error,
                type(error).__name__,
            )
            return self._build_document(
                url=url,
                canonical_url=effective_url or "",
                title="",
                blocks=(),
                status_code=status_code,
                content_type=content_type,
                fetched_at=fetched_at,
                metadata=metadata,
                source_size_bytes=source_size_bytes,
                source_estimated_tokens=source_estimated_tokens,
            )

        title = ""
        if soup.title is not None:
            title = _normalize_inline_text(soup.title.get_text(" ", strip=True))

        canonical_url = self._canonical_url(soup, effective_url)
        self._remove_non_content(soup)
        root = soup.body or soup
        if not title:
            first_h1 = root.find("h1")
            if isinstance(first_h1, Tag):
                title = _normalize_inline_text(first_h1.get_text(" ", strip=True))
        blocks = self._extract_blocks(root, effective_url)

        return self._build_document(
            url=url,
            canonical_url=canonical_url,
            title=title,
            blocks=blocks,
            status_code=status_code,
            content_type=content_type,
            fetched_at=fetched_at,
            metadata=metadata,
            source_size_bytes=source_size_bytes,
            source_estimated_tokens=source_estimated_tokens,
        )

    def _canonical_url(
        self,
        soup: BeautifulSoup,
        effective_url: str | None,
    ) -> str:
        for link in soup.find_all("link"):
            relation = link.get("rel")
            if isinstance(relation, str):
                relations = relation.split()
            elif isinstance(relation, list):
                relations = [str(value) for value in relation]
            else:
                continue
            if "canonical" not in {value.casefold() for value in relations}:
                continue
            href = link.get("href")
            if not isinstance(href, str) or effective_url is None:
                continue
            canonical_url = _resolve_http_url(href, effective_url)
            if canonical_url is not None and _same_origin(
                canonical_url,
                effective_url,
            ):
                return canonical_url
        return effective_url or ""

    def _remove_non_content(self, soup: BeautifulSoup) -> None:
        for element in list(soup.find_all(_REMOVED_TAGS)):
            element.decompose()

        navigation = [
            element
            for element in soup.find_all(True)
            if self._is_navigation_container(element)
        ]
        for element in navigation:
            if element.parent is not None:
                element.decompose()

    def _is_navigation_container(self, element: Tag) -> bool:
        if element.name in _NAVIGATION_NAMES:
            return True

        role = element.get("role")
        if isinstance(role, str) and role.strip().casefold() in _NAVIGATION_ROLES:
            return True

        markers: set[str] = set()
        identifier = element.get("id")
        if isinstance(identifier, str):
            markers.add(identifier.strip().casefold())
        classes = element.get("class")
        if isinstance(classes, list):
            markers.update(str(value).strip().casefold() for value in classes)
        elif isinstance(classes, str):
            markers.update(value.casefold() for value in classes.split())
        return bool(markers & _NAVIGATION_MARKERS)

    def _extract_blocks(
        self,
        root: BeautifulSoup | Tag,
        base_url: str | None,
    ) -> tuple[DocumentBlock, ...]:
        blocks: list[DocumentBlock] = []
        heading_stack: list[tuple[int, str]] = []

        def heading_path() -> tuple[str, ...]:
            return tuple(text for _level, text in heading_stack)

        def append_block(kind: BlockKind, text: str, markdown: str) -> None:
            if not text:
                return
            blocks.append(
                DocumentBlock(
                    kind=kind,
                    text=text,
                    markdown=markdown,
                    heading_path=heading_path(),
                )
            )

        def emit_element(element: Tag) -> None:
            name = element.name.casefold()
            if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                plain, markdown = self._render_inline(element, base_url)
                text = _normalize_inline_text(plain)
                if not text:
                    return
                level = int(name[1])
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                append_block(
                    "heading",
                    text,
                    f"{'#' * level} {_normalize_inline_text(markdown)}",
                )
                return

            if name == "p":
                plain, markdown = self._render_inline(element, base_url)
                append_block(
                    "paragraph",
                    _normalize_inline_text(plain),
                    _normalize_inline_text(markdown),
                )
                return

            if name in {"ul", "ol"}:
                text, markdown = self._render_list(element, base_url)
                append_block("list", text, markdown)
                return

            if name in {"code", "pre"}:
                text = _normalize_code(element.get_text())
                if text:
                    fence = "`" * max(3, _longest_run(text, "`") + 1)
                    append_block("code", text, f"{fence}\n{text}\n{fence}")
                return

            if name == "table":
                text, markdown = self._render_table(element, base_url)
                append_block("table", text, markdown)

        def walk(container: BeautifulSoup | Tag) -> None:
            pending_plain: list[str] = []
            pending_markdown: list[str] = []

            def flush_pending() -> None:
                plain = _normalize_inline_text(" ".join(pending_plain))
                markdown = _normalize_inline_text(" ".join(pending_markdown))
                pending_plain.clear()
                pending_markdown.clear()
                append_block("paragraph", plain, markdown)

            for child in container.children:
                if isinstance(child, Comment):
                    continue
                if isinstance(child, NavigableString):
                    value = _normalize_inline_text(str(child))
                    if value:
                        pending_plain.append(value)
                        pending_markdown.append(value)
                    continue
                if not isinstance(child, Tag):
                    continue
                name = child.name.casefold()
                if name in _BLOCK_TAGS:
                    flush_pending()
                    emit_element(child)
                    continue
                if child.find(_BLOCK_TAGS) is not None:
                    flush_pending()
                    walk(child)
                    continue
                plain, markdown = self._render_inline(child, base_url)
                markdown = self._decorate_inline(
                    child,
                    plain,
                    markdown,
                    base_url,
                )
                if name not in _INLINE_TAGS:
                    flush_pending()
                    append_block(
                        "paragraph",
                        _normalize_inline_text(plain),
                        _normalize_inline_text(markdown),
                    )
                    continue
                if plain:
                    pending_plain.append(plain)
                    pending_markdown.append(markdown)
            flush_pending()

        walk(root)
        return tuple(blocks)

    def _render_inline(
        self,
        element: Tag,
        base_url: str | None,
    ) -> tuple[str, str]:
        plain_parts: list[str] = []
        markdown_parts: list[str] = []

        for child in element.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                value = str(child)
                plain_parts.append(value)
                markdown_parts.append(value)
                continue
            if not isinstance(child, Tag):
                continue
            if child.name == "br":
                plain_parts.append("\n")
                markdown_parts.append("\n")
                continue

            child_plain, child_markdown = self._render_inline(child, base_url)
            child_markdown = self._decorate_inline(
                child,
                child_plain,
                child_markdown,
                base_url,
            )

            if child.name not in _INLINE_TAGS:
                plain_parts.extend((" ", child_plain, " "))
                markdown_parts.extend((" ", child_markdown, " "))
            else:
                plain_parts.append(child_plain)
                markdown_parts.append(child_markdown)

        return (
            _normalize_inline_text("".join(plain_parts)),
            _normalize_inline_text("".join(markdown_parts)),
        )

    def _decorate_inline(
        self,
        element: Tag,
        plain: str,
        markdown: str,
        base_url: str | None,
    ) -> str:
        if element.name == "a":
            href = element.get("href")
            target = (
                _resolve_http_url(href, base_url)
                if isinstance(href, str) and base_url is not None
                else None
            )
            normalized_label = _normalize_inline_text(markdown)
            if target is not None and normalized_label:
                return f"[{normalized_label}]({target})"
        elif element.name == "code":
            code = _normalize_inline_text(plain)
            if code:
                fence = "`" * max(1, _longest_run(code, "`") + 1)
                return f"{fence}{code}{fence}"
        return markdown

    def _render_list(
        self,
        html_list: Tag,
        base_url: str | None,
        *,
        depth: int = 0,
    ) -> tuple[str, str]:
        plain_lines: list[str] = []
        markdown_lines: list[str] = []
        items = html_list.find_all("li", recursive=False)

        for index, item in enumerate(items, start=1):
            plain_parts: list[str] = []
            markdown_parts: list[str] = []
            for child in item.children:
                if isinstance(child, Comment):
                    continue
                if isinstance(child, NavigableString):
                    plain_parts.append(str(child))
                    markdown_parts.append(str(child))
                    continue
                if not isinstance(child, Tag) or child.name in {"ul", "ol"}:
                    continue
                plain, markdown = self._render_inline(child, base_url)
                markdown = self._decorate_inline(
                    child,
                    plain,
                    markdown,
                    base_url,
                )
                plain_parts.append(plain)
                markdown_parts.append(markdown)

            plain = _normalize_inline_text(" ".join(plain_parts))
            markdown = _normalize_inline_text(" ".join(markdown_parts))
            prefix = f"{index}." if html_list.name == "ol" else "-"
            indentation = "  " * depth
            if plain:
                plain_lines.append(f"{indentation}{prefix} {plain}")
                markdown_lines.append(f"{indentation}{prefix} {markdown}")

            nested_lists = [
                nested
                for nested in item.find_all(["ul", "ol"])
                if nested.find_parent(["ul", "ol"]) is html_list
            ]
            for nested in nested_lists:
                nested_plain, nested_markdown = self._render_list(
                    nested,
                    base_url,
                    depth=depth + 1,
                )
                if nested_plain:
                    plain_lines.append(nested_plain)
                    markdown_lines.append(nested_markdown)

        return "\n".join(plain_lines), "\n".join(markdown_lines)

    def _render_table(
        self,
        table: Tag,
        base_url: str | None,
    ) -> tuple[str, str]:
        rows: list[list[tuple[str, str]]] = []
        for row in table.find_all("tr"):
            if row.find_parent("table") is not table:
                continue
            cells: list[tuple[str, str]] = []
            for cell in row.find_all(["th", "td"], recursive=False):
                plain, markdown = self._render_inline(cell, base_url)
                cells.append(
                    (
                        _normalize_inline_text(plain),
                        _normalize_inline_text(markdown),
                    )
                )
            if cells:
                rows.append(cells)

        if not rows:
            return "", ""

        width = max(len(row) for row in rows)
        padded_rows = [
            row + [("", "") for _index in range(width - len(row))] for row in rows
        ]
        plain = "\n".join(" | ".join(cell[0] for cell in row) for row in padded_rows)
        header = padded_rows[0]
        markdown_lines = [
            "| " + " | ".join(_escape_table_cell(cell[1]) for cell in header) + " |",
            "| " + " | ".join("---" for _cell in header) + " |",
        ]
        markdown_lines.extend(
            "| " + " | ".join(_escape_table_cell(cell[1]) for cell in row) + " |"
            for row in padded_rows[1:]
        )
        return plain, "\n".join(markdown_lines)

    def _build_document(
        self,
        *,
        url: str,
        canonical_url: str,
        title: str,
        blocks: tuple[DocumentBlock, ...],
        status_code: int,
        content_type: str,
        fetched_at: datetime,
        metadata: Mapping[str, object] | None,
        source_size_bytes: int,
        source_estimated_tokens: int,
    ) -> SourceDocument:
        text = "\n\n".join(block.text for block in blocks)
        rendered_markdown = "\n\n".join(block.markdown for block in blocks)
        markdown = rendered_markdown or None
        cleaned_size_bytes = len(text.encode("utf-8"))
        content_hash = _sha256("\0".join((title, text, rendered_markdown)))
        document_id = _sha256(canonical_url or url)

        return SourceDocument(
            id=document_id,
            url=url,
            canonical_url=canonical_url,
            title=title,
            text=text,
            markdown=markdown,
            status_code=status_code,
            content_type=content_type,
            fetched_at=fetched_at,
            content_hash=content_hash,
            metadata=dict(metadata or {}),
            source_size_bytes=source_size_bytes,
            cleaned_size_bytes=cleaned_size_bytes,
            source_estimated_tokens=source_estimated_tokens,
            cleaned_estimated_tokens=self._estimator.count(text),
            blocks=blocks,
        )


def _validated_http_url(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    try:
        absolute, _fragment = urldefrag(candidate)
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


def _resolve_http_url(value: str, base_url: str) -> str | None:
    return _validated_http_url(urljoin(base_url, value.strip()))


def _same_origin(left: str, right: str) -> bool:
    left_url = urlsplit(left)
    right_url = urlsplit(right)

    def origin(url: SplitResult) -> tuple[str, str, int]:
        scheme = url.scheme.casefold()
        hostname = (url.hostname or "").casefold()
        port = url.port
        if port is None:
            port = 443 if scheme == "https" else 80
        return scheme, hostname, port

    return origin(left_url) == origin(right_url)


def _normalize_inline_text(value: str) -> str:
    return " ".join(value.split())


def _normalize_code(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _longest_run(value: str, character: str) -> int:
    longest = 0
    current = 0
    for item in value:
        if item == character:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
