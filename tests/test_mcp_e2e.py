"""End-to-end local HTTP tests through the official in-memory MCP client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from mcp import Client

from crawlforge.mcp.config import MCPServerConfig
from crawlforge.mcp.server import create_server


@asynccontextmanager
async def serve(app: web.Application) -> AsyncIterator[TestServer]:
    server = TestServer(app)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_index_search_context_info_and_reindex_local_site(
    tmp_path: Path,
) -> None:
    hits = {"robots": 0, "root": 0, "retry": 0, "private": 0, "deep": 0}

    async def robots(_request: web.Request) -> web.Response:
        hits["robots"] += 1
        return web.Response(
            text="User-agent: *\nDisallow: /private\n",
            content_type="text/plain",
        )

    async def root(_request: web.Request) -> web.Response:
        hits["root"] += 1
        return web.Response(
            text="""
                <title>CrawlForge docs</title>
                <main>
                  <h1>Documentation</h1>
                  <p>Choose a bounded crawler topic.</p>
                  <a href="/retry">Retry guide</a>
                  <a href="/private">Private instructions</a>
                </main>
            """,
            content_type="text/html",
        )

    async def retry(_request: web.Request) -> web.Response:
        hits["retry"] += 1
        return web.Response(
            text="""
                <title>Retry guide</title>
                <main>
                  <h1>Networking</h1>
                  <h2>RetryStrategy</h2>
                  <p>
                    AsyncCrawler applies exponential backoff and Retry-After
                    before retrying a bounded request.
                  </p>
                  <a href="/deep">Deep page</a>
                </main>
            """,
            content_type="text/html",
        )

    async def private(_request: web.Request) -> web.Response:
        hits["private"] += 1
        return web.Response(
            text="<p>NeverFetchedPrivateMarker</p>",
            content_type="text/html",
        )

    async def deep(_request: web.Request) -> web.Response:
        hits["deep"] += 1
        return web.Response(
            text="<p>NeverFetchedDepthMarker</p>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", root)
    app.router.add_get("/retry", retry)
    app.router.add_get("/private", private)
    app.router.add_get("/deep", deep)

    async with serve(app) as site:
        retry_url = str(site.make_url("/retry"))
        config = MCPServerConfig(
            database=tmp_path / "index.db",
            max_pages_cap=2,
            max_depth_cap=1,
            allow_private_networks=True,
            requests_per_second=1000,
        )
        async with Client(create_server(config)) as client:
            over_cap = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/")),
                    "max_pages": 3,
                    "max_depth": 1,
                },
            )
            indexed = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/")),
                    "max_pages": 2,
                    "max_depth": 1,
                },
            )
            search = await client.call_tool(
                "search_index",
                {"query": "RetryStrategy exponential backoff", "limit": 5},
            )
            context = await client.call_tool(
                "build_context",
                {
                    "query": "How does RetryStrategy backoff work?",
                    "limit": 5,
                    "token_budget": 200,
                },
            )
            private_search = await client.call_tool(
                "search_index",
                {"query": "NeverFetchedPrivateMarker"},
            )
            deep_search = await client.call_tool(
                "search_index",
                {"query": "NeverFetchedDepthMarker"},
            )
            info = await client.call_tool("get_index_info", {})
            reindexed = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/")),
                    "max_pages": 2,
                    "max_depth": 1,
                },
            )

    assert over_cap.is_error
    assert hits["root"] == 2
    assert hits["retry"] == 2
    assert hits["private"] == 0
    assert hits["deep"] == 0
    assert hits["robots"] >= 2
    assert not indexed.is_error
    assert indexed.structured_content is not None
    assert indexed.structured_content["indexed_documents"] == 2
    assert indexed.structured_content["created_chunks"] >= 2
    assert indexed.structured_content["database"] == "index.db"
    assert str(tmp_path) not in str(indexed.structured_content)
    assert not search.is_error
    assert search.structured_content is not None
    assert search.structured_content["results"][0]["title"] == "Retry guide"
    assert not context.is_error
    assert context.structured_content is not None
    assert context.structured_content["chunks"]
    assert context.structured_content["chunks"][0]["url"] == retry_url
    assert private_search.structured_content is not None
    assert private_search.structured_content["results"] == []
    assert deep_search.structured_content is not None
    assert deep_search.structured_content["results"] == []
    assert info.structured_content is not None
    assert info.structured_content["document_count"] == 2
    assert info.structured_content["last_session_summary"] is not None
    assert not reindexed.is_error
    assert reindexed.structured_content is not None
    assert reindexed.structured_content["indexed_documents"] == 0
    assert reindexed.structured_content["deduplicated_documents"] == 2


@pytest.mark.asyncio
async def test_default_network_policy_blocks_local_site_before_request(
    tmp_path: Path,
) -> None:
    requests = 0

    async def root(_request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.Response(text="<p>must not be fetched</p>")

    app = web.Application()
    app.router.add_get("/", root)

    async with serve(app) as site:
        async with Client(
            create_server(MCPServerConfig(database=tmp_path / "index.db"))
        ) as client:
            blocked = await client.call_tool(
                "index_site",
                {"url": str(site.make_url("/")), "max_pages": 1, "max_depth": 0},
            )
            file_url = await client.call_tool(
                "index_site",
                {"url": "file:///etc/passwd", "max_pages": 1, "max_depth": 0},
            )

    assert requests == 0
    assert blocked.is_error
    assert "private" in blocked.content[0].text
    assert file_url.is_error
    assert "http and https" in file_url.content[0].text


@pytest.mark.asyncio
async def test_domain_allowlist_is_enforced_before_network_access(
    tmp_path: Path,
) -> None:
    config = MCPServerConfig(
        database=tmp_path / "index.db",
        allowed_domains=("docs.example.com",),
    )

    async with Client(create_server(config)) as client:
        blocked = await client.call_tool(
            "index_site",
            {"url": "https://evil-example.com/", "max_pages": 1, "max_depth": 0},
        )

    assert blocked.is_error
    assert "allowlist" in blocked.content[0].text


@pytest.mark.asyncio
async def test_in_memory_cancellation_stops_index_and_server_remains_usable(
    tmp_path: Path,
) -> None:
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def robots(_request: web.Request) -> web.Response:
        return web.Response(text="User-agent: *\nAllow: /\n")

    async def blocked(_request: web.Request) -> web.Response:
        request_started.set()
        await release_request.wait()
        return web.Response(
            text="<title>Late</title><p>late response</p>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", blocked)

    async with serve(app) as site:
        config = MCPServerConfig(
            database=tmp_path / "index.db",
            allow_private_networks=True,
            requests_per_second=1000,
        )
        async with Client(create_server(config)) as client:
            call = asyncio.create_task(
                client.call_tool(
                    "index_site",
                    {
                        "url": str(site.make_url("/")),
                        "max_pages": 1,
                        "max_depth": 0,
                    },
                )
            )
            await asyncio.wait_for(request_started.wait(), timeout=5)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            release_request.set()
            info = await asyncio.wait_for(
                client.call_tool("get_index_info", {}),
                timeout=5,
            )

    assert not info.is_error
    assert info.structured_content is not None
    assert info.structured_content["document_count"] == 0


@pytest.mark.asyncio
async def test_partial_http_failure_is_reported_as_bounded_warning(
    tmp_path: Path,
) -> None:
    async def robots(_request: web.Request) -> web.Response:
        return web.Response(text="User-agent: *\nAllow: /\n")

    async def root(_request: web.Request) -> web.Response:
        return web.Response(
            text=(
                "<title>Partial docs</title><p>RootSucceededMarker</p>"
                '<a href="/broken">Broken</a>'
            ),
            content_type="text/html",
        )

    async def broken(_request: web.Request) -> web.Response:
        raise web.HTTPInternalServerError()

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", root)
    app.router.add_get("/broken", broken)

    async with serve(app) as site:
        config = MCPServerConfig(
            database=tmp_path / "partial.db",
            allow_private_networks=True,
            requests_per_second=1000,
            crawl_timeout_seconds=5,
        )
        async with Client(create_server(config)) as client:
            indexed = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/")),
                    "max_pages": 2,
                    "max_depth": 1,
                },
            )
            search = await client.call_tool(
                "search_index",
                {"query": "RootSucceededMarker"},
            )

    assert not indexed.is_error
    assert indexed.structured_content is not None
    assert indexed.structured_content["indexed_documents"] == 1
    assert indexed.structured_content["failed_pages"] == 1
    assert len(indexed.structured_content["warnings"]) == 1
    assert "http_error" in indexed.structured_content["warnings"][0]
    assert "/broken" not in indexed.structured_content["warnings"][0]
    fallback = indexed.content[0].text
    assert "Omitted 1 page(s)" in fallback
    assert "http_error" in fallback
    assert "/broken" not in fallback
    assert not search.is_error


@pytest.mark.asyncio
async def test_page_byte_caps_reject_content_length_and_chunked_bodies(
    tmp_path: Path,
) -> None:
    async def robots(_request: web.Request) -> web.Response:
        return web.Response(text="User-agent: *\nAllow: /\n")

    async def content_length(_request: web.Request) -> web.Response:
        return web.Response(
            body=b"<p>" + b"x" * 1_024 + b"</p>",
            content_type="text/html",
        )

    async def chunked(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/html"})
        await response.prepare(request)
        try:
            for _ in range(8):
                await response.write(b"x" * 100)
        except (ConnectionError, RuntimeError):
            pass
        return response

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/length", content_length)
    app.router.add_get("/chunked", chunked)

    async with serve(app) as site:
        config = MCPServerConfig(
            database=tmp_path / "bounded.db",
            allow_private_networks=True,
            requests_per_second=1000,
            max_response_bytes=256,
            request_timeout_seconds=1,
            crawl_timeout_seconds=2,
        )
        async with Client(create_server(config)) as client:
            length_result = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/length")),
                    "max_pages": 1,
                    "max_depth": 0,
                },
            )
            chunked_result = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/chunked")),
                    "max_pages": 1,
                    "max_depth": 0,
                },
            )
            info = await client.call_tool("get_index_info", {})

    assert length_result.is_error
    assert chunked_result.is_error
    assert "byte limit" in length_result.content[0].text
    assert "byte limit" in chunked_result.content[0].text
    assert not info.is_error
    assert info.structured_content is not None
    assert info.structured_content["document_count"] == 0


@pytest.mark.asyncio
async def test_robots_byte_cap_fails_closed_before_page_request(
    tmp_path: Path,
) -> None:
    page_requests = 0

    async def large_robots(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        try:
            for _ in range(8):
                await response.write(b"x" * 100)
        except (ConnectionError, RuntimeError):
            pass
        return response

    async def root(_request: web.Request) -> web.Response:
        nonlocal page_requests
        page_requests += 1
        return web.Response(text="<p>must not be fetched</p>")

    app = web.Application()
    app.router.add_get("/robots.txt", large_robots)
    app.router.add_get("/", root)

    async with serve(app) as site:
        config = MCPServerConfig(
            database=tmp_path / "robots.db",
            allow_private_networks=True,
            requests_per_second=1000,
            max_robots_bytes=256,
            crawl_timeout_seconds=2,
        )
        async with Client(create_server(config)) as client:
            result = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/")),
                    "max_pages": 1,
                    "max_depth": 0,
                },
            )
            info = await client.call_tool("get_index_info", {})

    assert page_requests == 0
    assert result.is_error
    assert "byte limit" in result.content[0].text
    assert not info.is_error


@pytest.mark.asyncio
async def test_robots_content_length_cap_reports_byte_limit(
    tmp_path: Path,
) -> None:
    page_requests = 0

    async def large_robots(_request: web.Request) -> web.Response:
        return web.Response(body=b"x" * 1_024, content_type="text/plain")

    async def root(_request: web.Request) -> web.Response:
        nonlocal page_requests
        page_requests += 1
        return web.Response(text="<p>must not be fetched</p>")

    app = web.Application()
    app.router.add_get("/robots.txt", large_robots)
    app.router.add_get("/", root)

    async with serve(app) as site:
        config = MCPServerConfig(
            database=tmp_path / "robots-length.db",
            allow_private_networks=True,
            requests_per_second=1000,
            max_robots_bytes=256,
            crawl_timeout_seconds=2,
        )
        async with Client(create_server(config)) as client:
            result = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/")),
                    "max_pages": 1,
                    "max_depth": 0,
                },
            )
            info = await client.call_tool("get_index_info", {})

    assert page_requests == 0
    assert result.is_error
    assert "byte limit" in result.content[0].text
    assert not info.is_error


@pytest.mark.asyncio
async def test_robots_trickle_timeout_fails_closed_and_remains_usable(
    tmp_path: Path,
) -> None:
    page_requests = 0

    async def trickle_robots(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/plain"})
        await response.prepare(request)
        try:
            await response.write(b"x")
            await asyncio.sleep(0.5)
        except (ConnectionError, RuntimeError, asyncio.CancelledError):
            pass
        return response

    async def root(_request: web.Request) -> web.Response:
        nonlocal page_requests
        page_requests += 1
        return web.Response(text="<p>must not be fetched</p>")

    app = web.Application()
    app.router.add_get("/robots.txt", trickle_robots)
    app.router.add_get("/", root)

    async with serve(app) as site:
        config = MCPServerConfig(
            database=tmp_path / "robots-timeout.db",
            allow_private_networks=True,
            requests_per_second=1000,
            request_timeout_seconds=0.03,
            crawl_timeout_seconds=0.12,
        )
        async with Client(create_server(config)) as client:
            result = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/")),
                    "max_pages": 1,
                    "max_depth": 0,
                },
            )
            info = await client.call_tool("get_index_info", {})

    assert page_requests == 0
    assert result.is_error
    assert "timed out" in result.content[0].text
    assert not info.is_error


@pytest.mark.asyncio
async def test_total_request_and_crawl_timeouts_stop_trickle_response(
    tmp_path: Path,
) -> None:
    async def robots(_request: web.Request) -> web.Response:
        return web.Response(text="User-agent: *\nAllow: /\n")

    async def trickle(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/html"})
        await response.prepare(request)
        try:
            for _ in range(20):
                await response.write(b"x")
                await asyncio.sleep(0.02)
        except (ConnectionError, RuntimeError, asyncio.CancelledError):
            pass
        return response

    app = web.Application()
    app.router.add_get("/robots.txt", robots)
    app.router.add_get("/", trickle)

    async with serve(app) as site:
        config = MCPServerConfig(
            database=tmp_path / "timeout.db",
            allow_private_networks=True,
            requests_per_second=1000,
            request_timeout_seconds=0.08,
            crawl_timeout_seconds=0.3,
        )
        async with Client(create_server(config)) as client:
            result = await client.call_tool(
                "index_site",
                {
                    "url": str(site.make_url("/")),
                    "max_pages": 1,
                    "max_depth": 0,
                },
            )
            info = await client.call_tool("get_index_info", {})

    assert result.is_error
    assert "timed out" in result.content[0].text
    assert not info.is_error
