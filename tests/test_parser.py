"""Tests for HTML parsing and structured data extraction."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest
from bs4 import BeautifulSoup

from crawlforge import HTMLParser

VALID_HTML = """
<!doctype html>
<html>
  <head>
    <title> CrawlForge Parser </title>
    <meta name="description" content=" Structured   crawling ">
    <meta name="KEYWORDS" content="crawler, html">
  </head>
  <body>
    <script>ignored()</script>
    <!-- hidden comment -->
    <main class="content">
      <h1>Primary heading</h1>
      <p>Visible <strong>page</strong> text.</p>
    </main>
    <h2>Second heading</h2>
    <h3>Third heading</h3>
    <h4>Ignored heading</h4>
    <a href="/about">About</a>
    <a href="../guide#intro">Guide</a>
    <a href="//cdn.example.com/asset">CDN</a>
    <a href="https://external.example/path#details">External</a>
    <a href="/about#team">Duplicate</a>
    <a href="mailto:team@example.com">Email</a>
    <a href="javascript:void(0)">Script</a>
    <a href="data:text/plain,hello">Data</a>
    <a href="https://example.com:invalid/port">Bad port</a>
    <a href="https://bad host.example/path">Bad host</a>
    <a href="https://example.com\\@evil.example/path">Bad authority</a>
    <img src="/images/logo.png" alt=" CrawlForge   logo ">
    <img src="https://cdn.example.com/photo.jpg">
    <img src="data:image/png;base64,ignored" alt="inline">
    <table>
      <tr><th>Name</th><th>Value</th></tr>
      <tr><td>links</td><td>4</td></tr>
    </table>
    <ul><li>first</li><li>second</li></ul>
    <ol><li>one</li><li>two</li></ol>
  </body>
</html>
"""


@pytest.mark.asyncio
async def test_parse_html_extracts_structured_page_data() -> None:
    """A valid document produces a complete structured result."""
    parser = HTMLParser()

    result = await parser.parse_html(
        VALID_HTML,
        "https://example.com/articles/page.html",
    )

    assert result["url"] == "https://example.com/articles/page.html"
    assert result["title"] == "CrawlForge Parser"
    assert result["metadata"] == {
        "title": "CrawlForge Parser",
        "description": "Structured crawling",
        "keywords": "crawler, html",
    }
    assert "Visible page text." in result["text"]
    assert "ignored()" not in result["text"]
    assert "hidden comment" not in result["text"]
    assert result["links"] == [
        "https://example.com/about",
        "https://example.com/guide",
        "https://cdn.example.com/asset",
        "https://external.example/path",
    ]
    assert result["images"] == [
        {
            "src": "https://example.com/images/logo.png",
            "alt": "CrawlForge logo",
        },
        {"src": "https://cdn.example.com/photo.jpg", "alt": ""},
    ]
    assert result["headings"] == [
        {"level": 1, "text": "Primary heading"},
        {"level": 2, "text": "Second heading"},
        {"level": 3, "text": "Third heading"},
    ]
    assert result["tables"] == [
        [["Name", "Value"], ["links", "4"]],
    ]
    assert result["lists"] == [
        {"type": "ul", "items": ["first", "second"]},
        {"type": "ol", "items": ["one", "two"]},
    ]


def test_extract_text_supports_css_selector_and_missing_match() -> None:
    """Text extraction can be limited to one selected subtree."""
    parser = HTMLParser()
    soup = BeautifulSoup(VALID_HTML, "lxml")

    assert parser.extract_text(soup, ".content") == (
        "Primary heading Visible page text."
    )
    assert parser.extract_text(soup, ".missing") == ""


@pytest.mark.asyncio
async def test_malformed_html_returns_available_data() -> None:
    """The tolerant lxml parser recovers useful data from broken markup."""
    parser = HTMLParser()

    result = await parser.parse_html(
        "<html><head><title>Broken</title></head><body><h1>Still useful<p>Text",
        "https://example.com/broken",
    )

    assert result["url"] == "https://example.com/broken"
    assert result["title"]
    assert result["text"]
    assert result["headings"] == [{"level": 1, "text": "Still useful"}]
    assert "Text" in result["text"]


@pytest.mark.asyncio
async def test_parse_failure_logs_warning_and_returns_empty_shape(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A parser failure is isolated behind the stable result contract."""

    def fail_parser(html: str, parser: str) -> BeautifulSoup:
        raise ValueError("parser unavailable")

    monkeypatch.setattr("crawlforge.parser.BeautifulSoup", fail_parser)
    parser = HTMLParser()

    with caplog.at_level(logging.WARNING, logger="crawlforge.parser"):
        result = await parser.parse_html("<p>content</p>", "https://example.com")

    assert result == {
        "url": "https://example.com",
        "title": "",
        "text": "",
        "links": [],
        "metadata": {"title": "", "description": "", "keywords": ""},
        "images": [],
        "headings": [],
        "tables": [],
        "lists": [],
    }
    assert "Could not parse HTML from https://example.com" in caplog.text


@pytest.mark.asyncio
async def test_extractor_failure_preserves_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One failed extractor does not discard fields already available."""
    parser = HTMLParser()

    def fail_tables(soup: BeautifulSoup) -> list[list[list[str]]]:
        raise RuntimeError("table extraction failed")

    monkeypatch.setattr(parser, "extract_tables", fail_tables)

    with caplog.at_level(logging.WARNING, logger="crawlforge.parser"):
        result = await parser.parse_html(VALID_HTML, "https://example.com/base")

    assert result["title"] == "CrawlForge Parser"
    assert result["links"]
    assert result["images"]
    assert result["tables"] == []
    assert "Could not extract tables from https://example.com/base" in caplog.text


@pytest.mark.asyncio
async def test_parse_html_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synchronous parser work runs outside the event-loop thread."""
    parser = HTMLParser()
    worker_started = threading.Event()
    release_worker = threading.Event()
    event_loop_thread = threading.get_ident()
    parser_thread: int | None = None
    original_parse = parser._parse_html

    def blocked_parse(html: str, url: str) -> object:
        nonlocal parser_thread
        parser_thread = threading.get_ident()
        worker_started.set()
        release_worker.wait(timeout=5)
        return original_parse(html, url)

    monkeypatch.setattr(parser, "_parse_html", blocked_parse)
    task = asyncio.create_task(parser.parse_html("<p>ok</p>", "https://example.com"))
    await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=5)

    assert parser_thread is not None
    assert parser_thread != event_loop_thread

    release_worker.set()
    result = await asyncio.wait_for(task, timeout=5)
    assert result["text"] == "ok"


@pytest.mark.asyncio
async def test_parse_html_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling parser work is not converted into a partial success."""
    parser = HTMLParser()
    worker_started = threading.Event()
    release_worker = threading.Event()

    def blocked_parse(html: str, url: str) -> object:
        worker_started.set()
        release_worker.wait(timeout=5)
        return parser._empty_page(url)

    monkeypatch.setattr(parser, "_parse_html", blocked_parse)
    task = asyncio.create_task(parser.parse_html("<p>ok</p>", "https://example.com"))
    await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=5)
    task.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_worker.set()


def test_missing_metadata_and_empty_collections_are_explicit() -> None:
    """Absent optional HTML fields retain stable empty values."""
    parser = HTMLParser()
    soup = BeautifulSoup("<p>Only text</p>", "lxml")

    assert parser.extract_metadata(soup) == {
        "title": "",
        "description": "",
        "keywords": "",
    }
    assert parser.extract_tables(soup) == []
    assert parser.extract_lists(soup) == []


def test_nested_tables_and_lists_keep_parent_data_separate() -> None:
    """Nested structures are extracted without duplicating child content."""
    parser = HTMLParser()
    soup = BeautifulSoup(
        """
        <table>
          <tr>
            <td>outer<table><tr><td>inner</td></tr></table></td>
            <td>end</td>
          </tr>
        </table>
        <ul><li>parent<ul><li>child</li></ul></li></ul>
        """,
        "lxml",
    )

    assert parser.extract_tables(soup) == [
        [["outer", "end"]],
        [["inner"]],
    ]
    assert parser.extract_lists(soup) == [
        {"type": "ul", "items": ["parent"]},
        {"type": "ul", "items": ["child"]},
    ]
