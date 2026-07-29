"""Tests for deterministic web-content cleaning."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime

import pytest

from crawlforge.content import ContentProcessor

FETCHED_AT = datetime(2026, 7, 28, 9, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_cleaning_removes_non_content_without_selecting_only_main() -> None:
    """Scripts, styles, templates, and obvious navigation are excluded."""
    document = await ContentProcessor().process_html(
        """
        <html>
          <body>
            <script>script secret</script>
            <style>.style-secret { display: none }</style>
            <noscript>noscript secret</noscript>
            <template>template secret</template>
            <nav>semantic navigation</nav>
            <footer>semantic footer</footer>
            <div role="navigation">role navigation</div>
            <div id="main-nav">id navigation</div>
            <div class="site-footer">class footer</div>
            <main><p>Main content</p></main>
            <section><p>Useful section</p></section>
            <widget-pane>Useful unusual container</widget-pane>
            <div class="menu-article">Useful conservatively kept menu article</div>
            <custom-group><div>Nested one</div><div>Nested two</div></custom-group>
          </body>
        </html>
        """,
        "https://example.com/articles/page",
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )

    assert document.text == (
        "Main content\n\n"
        "Useful section\n\n"
        "Useful unusual container\n\n"
        "Useful conservatively kept menu article\n\n"
        "Nested one Nested two"
    )
    for removed in (
        "secret",
        "navigation",
        "footer",
    ):
        assert removed not in document.text


@pytest.mark.asyncio
async def test_unicode_title_fallback_and_heading_hierarchy_are_preserved() -> None:
    """Unicode h1-h6 text becomes the title and ordered heading path."""
    document = await ContentProcessor().process_html(
        """
        <body>
          <h1>Привет 世界</h1>
          <p>Первый абзац — полезный.</p>
          <h3>Раздел 三</h3>
          <p>Содержимое café.</p>
          <h2>Новый раздел</h2>
          <h6>Глубина</h6>
          <p>Финал 🚀</p>
        </body>
        """,
        "https://example.com/unicode",
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )

    assert document.title == "Привет 世界"
    assert [block.kind for block in document.blocks] == [
        "heading",
        "paragraph",
        "heading",
        "paragraph",
        "heading",
        "heading",
        "paragraph",
    ]
    assert document.blocks[3].heading_path == ("Привет 世界", "Раздел 三")
    assert document.blocks[-1].heading_path == (
        "Привет 世界",
        "Новый раздел",
        "Глубина",
    )
    assert "Финал 🚀" in document.text


@pytest.mark.asyncio
async def test_title_fallback_uses_h1_only() -> None:
    """A lower-level section heading is not promoted to the document title."""
    document = await ContentProcessor().process_html(
        "<main><h2>Section only</h2><p>Useful content</p></main>",
        "https://example.com/no-title",
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )

    assert document.title == ""
    assert document.blocks[0].text == "Section only"


@pytest.mark.asyncio
async def test_nested_lists_links_and_canonical_url_retain_meaning() -> None:
    """Nested list markers and validated absolute links survive in Markdown."""
    document = await ContentProcessor().process_html(
        """
        <head>
          <title>Guide</title>
          <link rel="alternate" href="/wrong">
          <link rel="canonical" href="../guide#details">
        </head>
        <body>
          <ol>
            <li>
              Open <a href="/start">the start page</a>
              <ul>
                <li>Read <a href="../docs">the docs</a></li>
              </ul>
            </li>
            <li>Finish</li>
          </ol>
        </body>
        """,
        "https://example.com/articles/original",
        final_url="https://example.com/articles/final",
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )
    canonical_document = await ContentProcessor().process_html(
        "<title>Canonical target</title><p>Target content</p>",
        "https://example.com/guide",
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )

    assert document.canonical_url == "https://example.com/guide"
    assert document.id == canonical_document.id
    assert document.text == ("1. Open the start page\n  - Read the docs\n2. Finish")
    assert document.markdown == (
        "1. Open [the start page](https://example.com/start)\n"
        "  - Read [the docs](https://example.com/docs)\n"
        "2. Finish"
    )


@pytest.mark.asyncio
async def test_cross_origin_canonical_cannot_replace_document_identity() -> None:
    """An untrusted external canonical remains advisory, not document identity."""
    processor = ContentProcessor()
    attacker = await processor.process_html(
        """
        <head>
          <title>Attacker</title>
          <link rel="canonical" href="https://trusted.example/document">
        </head>
        <body><p>PoisonOnlyKeyword</p></body>
        """,
        "https://attacker.example/page",
        final_url="https://attacker.example/page",
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )
    trusted = await processor.process_html(
        "<title>Trusted</title><p>TrustedOnlyKeyword</p>",
        "https://trusted.example/document",
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )

    assert attacker.canonical_url == "https://attacker.example/page"
    assert trusted.canonical_url == "https://trusted.example/document"
    assert attacker.id != trusted.id


@pytest.mark.asyncio
async def test_code_newlines_and_simple_table_are_rendered() -> None:
    """Preformatted indentation, fenced Markdown, and table rows are retained."""
    document = await ContentProcessor().process_html(
        """
        <body>
          <h2>Examples</h2>
          <pre><code>    first_line()
if ready:
    second_line()
</code></pre>
          <code>standalone()</code>
          <table>
            <tr><th>Name</th><th>Docs</th></tr>
            <tr><td>crawl</td><td><a href="/api">API</a></td></tr>
          </table>
        </body>
        """,
        "https://example.com/guide",
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )

    code_blocks = [block for block in document.blocks if block.kind == "code"]
    code = code_blocks[0]
    table = next(block for block in document.blocks if block.kind == "table")
    assert len(code_blocks) == 2
    assert code.text == "    first_line()\nif ready:\n    second_line()"
    assert code.markdown == ("```\n    first_line()\nif ready:\n    second_line()\n```")
    assert code_blocks[1].markdown == "```\nstandalone()\n```"
    assert table.text == "Name | Docs\ncrawl | API"
    assert table.markdown == (
        "| Name | Docs |\n| --- | --- |\n| crawl | [API](https://example.com/api) |"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("html", "expected_title", "expected_fragment"),
    [
        ("", "", ""),
        (
            "<html><head><title>Broken</title></head><body><h1>Available<p>partial",
            "Broken",
            "partial",
        ),
    ],
)
async def test_empty_and_malformed_html_are_safe_partial_results(
    html: str,
    expected_title: str,
    expected_fragment: str,
) -> None:
    """Empty input stays empty and malformed input keeps recoverable content."""
    document = await ContentProcessor().process_html(
        html,
        "https://example.com/broken",
        status_code=206,
        content_type="text/html",
        fetched_at=FETCHED_AT,
    )

    assert document.title == expected_title
    assert expected_fragment in document.text
    assert document.status_code == 206
    assert len(document.id) == 64
    assert len(document.content_hash) == 64


@pytest.mark.asyncio
async def test_processing_is_deterministic_and_normalizes_visible_whitespace() -> None:
    """Equivalent repeated processing produces exact IDs, hashes, and blocks."""
    html = """
    <html>
      <head>
        <title>  Stable    title </title>
        <link rel="canonical" href="javascript:invalid">
      </head>
      <body>
        <p> Visible
            text\twith    spaces. </p>
      </body>
    </html>
    """
    processor = ContentProcessor()
    arguments = {
        "final_url": "https://example.com/final#fragment",
        "status_code": 200,
        "content_type": "text/html",
        "fetched_at": FETCHED_AT,
        "metadata": {"private_note": "metadata only"},
    }

    first = await processor.process_html(
        html,
        "https://example.com/requested",
        **arguments,
    )
    second = await processor.process_html(
        html,
        "https://example.com/requested",
        **arguments,
    )

    assert first == second
    assert first.canonical_url == "https://example.com/final"
    assert first.text == "Visible text with spaces."
    assert first.blocks == second.blocks
    assert first.id == second.id
    assert first.content_hash == second.content_hash
    assert first.source_size_bytes == len(html.encode("utf-8"))
    assert first.cleaned_size_bytes == len(first.text.encode("utf-8"))
    assert first.source_estimated_tokens > first.cleaned_estimated_tokens
    assert first.metadata == {"private_note": "metadata only"}
    assert "example.com" not in first.text
    assert "metadata only" not in first.text


@pytest.mark.asyncio
async def test_dom_processing_runs_outside_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synchronous cleaning is offloaded to a worker thread."""
    processor = ContentProcessor()
    worker_started = threading.Event()
    release_worker = threading.Event()
    event_loop_thread = threading.get_ident()
    worker_thread: int | None = None
    original = processor._process_html

    def blocked_process(*args: object, **kwargs: object) -> object:
        nonlocal worker_thread
        worker_thread = threading.get_ident()
        worker_started.set()
        release_worker.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(processor, "_process_html", blocked_process)
    task = asyncio.create_task(
        processor.process_html(
            "<p>content</p>",
            "https://example.com",
            status_code=200,
            content_type="text/html",
            fetched_at=FETCHED_AT,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=5)

    assert worker_thread is not None
    assert worker_thread != event_loop_thread

    release_worker.set()
    document = await asyncio.wait_for(task, timeout=5)
    assert document.text == "content"


@pytest.mark.asyncio
async def test_dom_processing_cancellation_waits_for_owned_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation cannot leave DOM cleaning alive after task exit."""
    processor = ContentProcessor()
    worker_started = threading.Event()
    release_worker = threading.Event()
    original = processor._process_html

    def blocked_process(*args: object, **kwargs: object) -> object:
        worker_started.set()
        release_worker.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(processor, "_process_html", blocked_process)
    task = asyncio.create_task(
        processor.process_html(
            "<p>content</p>",
            "https://example.com",
            status_code=200,
            content_type="text/html",
            fetched_at=FETCHED_AT,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=5)
    task.cancel()

    await asyncio.sleep(0)
    assert not task.done()
    release_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
