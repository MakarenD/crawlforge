"""Application-service and local crawl-to-context tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge.context_engine import ContextEngine, EmptyCrawlError
from crawlforge.crawler import CrawledPage
from crawlforge.network_policy import URLNetworkPolicy


@asynccontextmanager
async def serve(app: web.Application) -> AsyncIterator[TestServer]:
    """Run an aiohttp application on an ephemeral local port."""
    server = TestServer(app)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_local_site_crawl_indexes_and_retrieves_the_right_section(
    tmp_path: Path,
) -> None:
    """A local site flows through crawl, clean, chunk, FTS, and context selection."""

    async def root(_request: web.Request) -> web.Response:
        return web.Response(
            text="""
                <html>
                  <head><title>Documentation</title></head>
                  <body>
                    <nav>Global navigation noise</nav>
                    <main>
                      <h1>Documentation</h1>
                      <p>Choose a topic.</p>
                      <a href="/docs/retries">Retry configuration</a>
                    </main>
                  </body>
                </html>
            """,
            content_type="text/html",
        )

    async def retries(_request: web.Request) -> web.Response:
        return web.Response(
            text="""
                <html>
                  <head><title>Retry guide</title></head>
                  <body>
                    <script>secretNavigationBootstrap()</script>
                    <main>
                      <h1>Networking</h1>
                      <h2>Retry-After and backoff</h2>
                      <p>
                        Configure exponential backoff with the backoff_base and
                        backoff_max settings. Retry-After is a lower bound.
                      </p>
                      <pre><code class="language-python">RetryStrategy(
    max_retries=3,
    backoff_factor=1.0,
)</code></pre>
                    </main>
                    <footer>Unrelated footer noise</footer>
                  </body>
                </html>
            """,
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/", root)
    app.router.add_get("/docs/retries", retries)

    async with serve(app) as server, ContextEngine(tmp_path / "context.db") as engine:
        root_url = str(server.make_url("/"))
        retry_url = str(server.make_url("/docs/retries"))
        indexed = await engine.ingest_url(
            root_url,
            max_pages=10,
            max_depth=1,
            max_concurrent=2,
            requests_per_second=1000,
            respect_robots=False,
        )
        hits = await engine.search("Retry-After exponential backoff", limit=5)
        context = await engine.build_context(
            "How are retries configured?",
            limit=5,
            token_budget=200,
        )

    assert indexed.documents_seen == 2
    assert indexed.documents_indexed == 2
    assert indexed.chunks_indexed >= 2
    assert indexed.source_size_bytes > indexed.cleaned_size_bytes
    assert hits
    assert hits[0].rank == 1
    assert hits[0].source.url == retry_url
    assert "backoff_base" in hits[0].chunk.text
    assert "secretNavigationBootstrap" not in hits[0].chunk.text
    assert context.hits
    assert context.hits[0].source.url == retry_url
    assert context.estimated_tokens <= context.token_budget
    assert context.estimated_context_reduction > 0
    assert context.index_hit


@pytest.mark.asyncio
async def test_context_budget_limit_and_empty_results(tmp_path: Path) -> None:
    """Selection obeys complete-chunk budgets and handles misses predictably."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(
            text=(
                "<title>Budget guide</title><h1>Budget</h1>"
                "<p>alpha beta gamma delta epsilon zeta eta theta</p>"
            ),
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/", page)

    async with serve(app) as server, ContextEngine(tmp_path / "context.db") as engine:
        await engine.ingest_url(
            str(server.make_url("/")),
            max_pages=1,
            max_depth=0,
            requests_per_second=1000,
            respect_robots=False,
        )
        limited = await engine.build_context("alpha", limit=1, token_budget=1)
        missing = await engine.build_context("term-not-present", limit=3)
        empty_query = await engine.search("", limit=3)

    assert limited.candidates_considered == 1
    assert limited.hits == ()
    assert limited.estimated_tokens == 0
    assert missing.hits == ()
    assert missing.candidates_considered == 0
    assert not missing.index_hit
    assert empty_query == []


@pytest.mark.asyncio
async def test_cross_origin_canonical_cannot_poison_existing_index(
    tmp_path: Path,
) -> None:
    """A page cannot replace another origin through its canonical declaration."""
    fetched_at = datetime(2026, 7, 28, tzinfo=UTC)
    trusted_url = "https://trusted.example/document"
    attacker_url = "https://attacker.example/page"
    trusted = CrawledPage(
        url=trusted_url,
        final_url=trusted_url,
        html="<title>Trusted</title><p>TrustedOnlyKeyword</p>",
        status_code=200,
        content_type="text/html",
        fetched_at=fetched_at,
        depth=0,
    )
    attacker = CrawledPage(
        url=attacker_url,
        final_url=attacker_url,
        html=(
            "<title>Attacker</title>"
            f'<link rel="canonical" href="{trusted_url}">'
            "<p>PoisonOnlyKeyword</p>"
        ),
        status_code=200,
        content_type="text/html",
        fetched_at=fetched_at,
        depth=0,
    )

    async with ContextEngine(tmp_path / "context.db") as engine:
        await engine.index_pages([trusted])
        await engine.index_pages([attacker])
        trusted_hits = await engine.search("TrustedOnlyKeyword")
        attacker_hits = await engine.search("PoisonOnlyKeyword")

    assert [hit.source.url for hit in trusted_hits] == [trusted_url]
    assert [hit.source.url for hit in attacker_hits] == [attacker_url]
    assert trusted_hits[0].source.document_id != attacker_hits[0].source.document_id


@pytest.mark.asyncio
async def test_context_engine_validates_limits_and_closed_lifecycle(
    tmp_path: Path,
) -> None:
    """Expected user errors are explicit and a closed engine cannot reopen."""
    engine = ContextEngine(tmp_path / "context.db")

    with pytest.raises(ValueError, match="limit"):
        await engine.search("query", limit=0)
    with pytest.raises(ValueError, match="token_budget"):
        await engine.build_context("query", token_budget=0)

    await engine.close()
    await engine.close()

    with pytest.raises(RuntimeError, match="closed"):
        await engine.search("query")


@pytest.mark.asyncio
async def test_context_engine_exposes_bounded_index_info(tmp_path: Path) -> None:
    engine = ContextEngine(tmp_path / "context.db")

    info = await engine.get_index_info()
    await engine.close()

    assert info.schema_version == 3
    assert info.document_count == 0
    assert info.chunk_count == 0
    assert info.database_ready
    assert info.fts5_available


@pytest.mark.asyncio
async def test_fail_on_empty_surfaces_network_policy_without_requesting_localhost(
    tmp_path: Path,
) -> None:
    requests = 0

    async def page(_request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.Response(text="<p>must not be fetched</p>")

    app = web.Application()
    app.router.add_get("/", page)

    async with serve(app) as server:
        engine = ContextEngine(
            tmp_path / "context.db",
            network_policy=URLNetworkPolicy(),
        )
        with pytest.raises(EmptyCrawlError, match="network policy"):
            await engine.ingest_url(
                str(server.make_url("/")),
                max_pages=1,
                max_depth=0,
                requests_per_second=1000,
                respect_robots=False,
                fail_on_empty=True,
            )
        await engine.close()

    assert requests == 0
