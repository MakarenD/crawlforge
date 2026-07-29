"""In-memory protocol, lifecycle, retrieval, and error tests for MCP."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcp import Client

from crawlforge.context_engine import ContextEngine
from crawlforge.context_index import FTS5UnavailableError
from crawlforge.context_models import IndexInfo, IndexingResult
from crawlforge.crawler import CrawledPage
from crawlforge.mcp.config import MCPServerConfig
from crawlforge.mcp.server import create_server
from crawlforge.mcp.tools import ServerRuntime, ToolExecutionError

EXPECTED_TOOLS = {
    "index_site",
    "search_index",
    "build_context",
    "get_index_info",
}


def _page(url: str, title: str, body: str) -> CrawledPage:
    return CrawledPage(
        url=url,
        final_url=url,
        html=f"<title>{title}</title><main><h1>{title}</h1><p>{body}</p></main>",
        status_code=200,
        content_type="text/html",
        fetched_at=datetime(2026, 7, 29, tzinfo=UTC),
        depth=0,
    )


async def _seed_database(path: Path) -> None:
    async with ContextEngine(path) as engine:
        await engine.index_pages(
            [
                _page(
                    "https://docs.example.test/retries",
                    "Retry guide",
                    (
                        "AsyncCrawler uses RetryStrategy and exponential backoff. "
                        "Retry-After sets a lower bound for the next request. "
                        + "bounded retry policy "
                        * 45
                    ),
                ),
                _page(
                    "https://docs.example.test/storage",
                    "Storage guide",
                    (
                        "SQLiteContextIndex stores complete chunks in SQLite FTS5. "
                        "BM25 ranks exact technical terms deterministically."
                    ),
                ),
            ]
        )


@pytest.mark.asyncio
async def test_lifecycle_opens_reuses_and_closes_one_engine(tmp_path: Path) -> None:
    entered = 0
    closed = 0

    class TrackingEngine:
        async def __aenter__(self) -> TrackingEngine:
            nonlocal entered
            entered += 1
            return self

        async def get_index_info(self) -> IndexInfo:
            return IndexInfo(
                schema_version=2,
                document_count=0,
                chunk_count=0,
                last_indexed_at=None,
                last_session_summary=None,
                database_ready=True,
                fts5_available=True,
            )

        async def close(self) -> None:
            nonlocal closed
            closed += 1

    engine = TrackingEngine()

    def factory(_path: Path, _policy: object) -> TrackingEngine:
        return engine

    server = create_server(
        MCPServerConfig(database=tmp_path / "index.db"),
        engine_factory=factory,  # type: ignore[arg-type]
    )
    async with Client(server) as client:
        first = await client.call_tool("get_index_info", {})
        second = await client.call_tool("get_index_info", {})

    assert not first.is_error
    assert not second.is_error
    assert entered == 1
    assert closed == 1


@pytest.mark.asyncio
async def test_server_discovery_exposes_exact_typed_tool_contract(
    tmp_path: Path,
) -> None:
    server = create_server(MCPServerConfig(database=tmp_path / "index.db"))

    async with Client(server) as client:
        discovered = await client.list_tools()

    tools = {tool.name: tool for tool in discovered.tools}
    assert set(tools) == EXPECTED_TOOLS
    assert all(tool.output_schema is not None for tool in tools.values())
    assert "database" not in tools["index_site"].input_schema["properties"]
    assert tools["index_site"].input_schema["required"] == ["url"]
    assert tools["index_site"].annotations is not None
    assert tools["index_site"].annotations.destructive_hint is True
    assert tools["index_site"].annotations.idempotent_hint is False
    assert tools["search_index"].input_schema["properties"]["limit"]["maximum"] == 100
    assert "exact technical terms" in tools["search_index"].description
    assert "Prefer build_context" in tools["search_index"].description
    assert "model-agnostic" in tools["build_context"].description
    assert "untrusted" in tools["build_context"].description
    assert "provenance" in tools["build_context"].description


@pytest.mark.asyncio
async def test_get_index_info_reports_empty_and_populated_indexes(
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "empty.db"
    populated_path = tmp_path / "populated.db"
    await _seed_database(populated_path)

    async with Client(
        create_server(MCPServerConfig(database=empty_path))
    ) as empty_client:
        empty = await empty_client.call_tool("get_index_info", {})
    async with Client(
        create_server(MCPServerConfig(database=populated_path))
    ) as populated_client:
        populated = await populated_client.call_tool("get_index_info", {})

    assert not empty.is_error
    assert empty.structured_content == {
        "schema_version": 3,
        "document_count": 0,
        "chunk_count": 0,
        "last_indexed_at": None,
        "last_session_summary": None,
        "database_ready": True,
        "fts5_available": True,
    }
    assert not populated.is_error
    assert populated.structured_content is not None
    assert populated.structured_content["document_count"] == 2
    assert populated.structured_content["chunk_count"] >= 2
    assert populated.structured_content["last_session_summary"] is not None


@pytest.mark.asyncio
async def test_get_index_info_stays_available_for_corrupt_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "corrupt.db"
    database.write_bytes(b"not a sqlite database")

    async with Client(create_server(MCPServerConfig(database=database))) as client:
        info = await client.call_tool("get_index_info", {})
        search = await client.call_tool("search_index", {"query": "anything"})

    assert not info.is_error
    assert info.structured_content is not None
    assert not info.structured_content["database_ready"]
    assert info.structured_content["fts5_available"]
    assert search.is_error
    assert "unavailable" in search.content[0].text
    assert str(tmp_path) not in search.content[0].text


@pytest.mark.asyncio
async def test_get_index_info_stays_available_when_database_parent_is_invalid(
    tmp_path: Path,
) -> None:
    regular_file = tmp_path / "not-a-directory"
    regular_file.write_text("occupied", encoding="utf-8")
    server = create_server(MCPServerConfig(database=regular_file / "index.db"))

    async with Client(server) as client:
        tools = await client.list_tools()
        info = await client.call_tool("get_index_info", {})
        search = await client.call_tool("search_index", {"query": "anything"})

    assert {tool.name for tool in tools.tools} == EXPECTED_TOOLS
    assert not info.is_error
    assert info.structured_content is not None
    assert not info.structured_content["database_ready"]
    assert search.is_error
    assert "unavailable" in search.content[0].text
    assert str(tmp_path) not in search.content[0].text


@pytest.mark.asyncio
async def test_get_index_info_reports_missing_fts5_without_crashing(
    tmp_path: Path,
) -> None:
    closed = False

    class MissingFTSEngine:
        async def __aenter__(self) -> MissingFTSEngine:
            raise FTS5UnavailableError("no such module: fts5")

        async def close(self) -> None:
            nonlocal closed
            closed = True

    def factory(_path: Path, _policy: object) -> MissingFTSEngine:
        return MissingFTSEngine()

    server = create_server(
        MCPServerConfig(database=tmp_path / "index.db"),
        engine_factory=factory,  # type: ignore[arg-type]
    )
    async with Client(server) as client:
        info = await client.call_tool("get_index_info", {})
        search = await client.call_tool("search_index", {"query": "RetryStrategy"})

    assert closed
    assert not info.is_error
    assert info.structured_content is not None
    assert not info.structured_content["database_ready"]
    assert not info.structured_content["fts5_available"]
    assert search.is_error
    assert "FTS5" in search.content[0].text


@pytest.mark.asyncio
async def test_search_index_returns_ranked_trusted_schema_and_respects_limit(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.db"
    await _seed_database(database)

    async with Client(create_server(MCPServerConfig(database=database))) as client:
        result = await client.call_tool(
            "search_index",
            {"query": "RetryStrategy backoff", "limit": 1},
        )
        special = await client.call_tool(
            "search_index",
            {"query": '"RetryStrategy" OR (backoff):', "limit": 5},
        )
        multiple = await client.call_tool(
            "search_index",
            {"query": "guide", "limit": 5},
        )

    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["returned_results"] == 1
    hit = result.structured_content["results"][0]
    assert hit["rank"] == 1
    assert isinstance(hit["bm25_score"], float)
    assert hit["title"] == "Retry guide"
    assert hit["heading_path"]
    assert hit["url"] == "https://docs.example.test/retries"
    assert hit["canonical_url"] == hit["url"]
    assert "RetryStrategy" in hit["chunk_text"]
    assert hit["content_trust"] == "untrusted_web_content"
    assert result.content[0].text == "Found 1 lexical chunk result(s)."
    assert not special.is_error
    assert multiple.structured_content is not None
    assert multiple.structured_content["returned_results"] >= 2


@pytest.mark.asyncio
async def test_search_validation_empty_index_and_result_size_cap(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.db"
    await _seed_database(database)
    limited_config = MCPServerConfig(
        database=database,
        max_result_bytes=1_024,
    )

    async with Client(create_server(limited_config)) as client:
        empty_query = await client.call_tool("search_index", {"query": "   "})
        too_many = await client.call_tool(
            "search_index",
            {"query": "RetryStrategy", "limit": 21},
        )
        limited = await client.call_tool(
            "search_index",
            {"query": "RetryStrategy", "limit": 5},
        )

    async with Client(
        create_server(MCPServerConfig(database=tmp_path / "new.db"))
    ) as empty_client:
        empty_index = await empty_client.call_tool(
            "search_index",
            {"query": "RetryStrategy"},
        )

    assert empty_query.is_error
    assert "non-whitespace" in empty_query.content[0].text
    assert too_many.is_error
    assert "server cap" in too_many.content[0].text
    assert not limited.is_error
    assert limited.structured_content is not None
    assert limited.structured_content["result_limited"]
    assert limited.structured_content["warnings"]
    assert empty_index.is_error
    assert "empty" in empty_index.content[0].text


@pytest.mark.asyncio
async def test_build_context_returns_complete_ordered_bounded_chunks(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.db"
    await _seed_database(database)

    async with Client(create_server(MCPServerConfig(database=database))) as client:
        context = await client.call_tool(
            "build_context",
            {
                "query": "RetryStrategy SQLiteContextIndex",
                "limit": 2,
                "token_budget": 500,
            },
        )
        no_match = await client.call_tool(
            "build_context",
            {"query": "term-not-present", "token_budget": 100},
        )

    assert not context.is_error
    assert context.structured_content is not None
    structured = context.structured_content
    assert structured["returned_chunks"] == 2
    assert [chunk["rank"] for chunk in structured["chunks"]] == [1, 2]
    assert len({chunk["chunk_text"] for chunk in structured["chunks"]}) == 2
    assert all(chunk["chunk_text"] for chunk in structured["chunks"])
    assert structured["estimated_tokens"] <= structured["token_budget"]
    assert structured["token_estimate"] == "model_agnostic_heuristic"
    assert structured["context_reduction_interpretation"] == (
        "approximate_ratio_not_model_specific_savings"
    )
    assert all(
        chunk["content_trust"] == "untrusted_web_content"
        for chunk in structured["chunks"]
    )
    assert "complete chunk(s)" in context.content[0].text
    assert not no_match.is_error
    assert no_match.structured_content is not None
    assert no_match.structured_content["chunks"] == []


class _ConcurrentEngine:
    def __init__(self) -> None:
        self.index_active = 0
        self.max_index_active = 0
        self.search_active = 0
        self.max_search_active = 0
        self.index_started = asyncio.Event()
        self.release_index = asyncio.Event()
        self.search_release = asyncio.Event()
        self.two_searches_started = asyncio.Event()

    async def __aenter__(self) -> _ConcurrentEngine:
        return self

    async def close(self) -> None:
        return None

    async def ingest_url(self, _url: str, **_kwargs: object) -> IndexingResult:
        self.index_active += 1
        self.max_index_active = max(self.max_index_active, self.index_active)
        self.index_started.set()
        try:
            await self.release_index.wait()
        finally:
            self.index_active -= 1
        return IndexingResult(
            session_id="session",
            documents_seen=1,
            documents_indexed=1,
            duplicate_documents=0,
            chunks_indexed=1,
            duplicate_chunks=0,
            source_size_bytes=10,
            cleaned_size_bytes=8,
            source_estimated_tokens=3,
            cleaned_estimated_tokens=2,
            cleaning_time_ms=1,
            indexing_time_ms=1,
        )

    async def get_index_info(self) -> IndexInfo:
        return IndexInfo(
            schema_version=2,
            document_count=1,
            chunk_count=1,
            last_indexed_at=None,
            last_session_summary=None,
            database_ready=True,
            fts5_available=True,
        )

    async def search(self, _query: str, *, limit: int) -> list[object]:
        del limit
        self.search_active += 1
        self.max_search_active = max(self.max_search_active, self.search_active)
        if self.search_active == 2:
            self.two_searches_started.set()
        try:
            await self.search_release.wait()
        finally:
            self.search_active -= 1
        return []


@pytest.mark.asyncio
async def test_runtime_serializes_writes_without_blocking_parallel_reads(
    tmp_path: Path,
) -> None:
    engine = _ConcurrentEngine()
    runtime = ServerRuntime(
        config=MCPServerConfig(database=tmp_path / "index.db"),
        engine=engine,  # type: ignore[arg-type]
    )

    first_write = asyncio.create_task(
        runtime.index_site("https://example.com/one", max_pages=1, max_depth=0)
    )
    await engine.index_started.wait()
    second_write = asyncio.create_task(
        runtime.index_site("https://example.com/two", max_pages=1, max_depth=0)
    )
    first_read = asyncio.create_task(runtime.search_index("alpha", limit=1))
    second_read = asyncio.create_task(runtime.search_index("beta", limit=1))
    await asyncio.wait_for(engine.two_searches_started.wait(), timeout=1)
    engine.search_release.set()
    await asyncio.gather(first_read, second_read)

    assert engine.max_search_active == 2
    assert not second_write.done()
    engine.release_index.set()
    await asyncio.gather(first_write, second_write)
    assert engine.max_index_active == 1


@pytest.mark.asyncio
async def test_runtime_crawl_timeout_includes_waiting_for_write_lock(
    tmp_path: Path,
) -> None:
    engine = _ConcurrentEngine()
    runtime = ServerRuntime(
        config=MCPServerConfig(
            database=tmp_path / "index.db",
            crawl_timeout_seconds=0.02,
        ),
        engine=engine,  # type: ignore[arg-type]
    )
    await runtime.write_lock.acquire()

    try:
        with pytest.raises(ToolExecutionError, match="timed out"):
            await runtime.index_site(
                "https://example.com/queued",
                max_pages=1,
                max_depth=0,
            )
        assert engine.index_active == 0
    finally:
        runtime.write_lock.release()

    engine.release_index.set()
    completed = await runtime.index_site(
        "https://example.com/after-timeout",
        max_pages=1,
        max_depth=0,
    )
    assert completed.indexed_documents == 1


@pytest.mark.asyncio
async def test_runtime_propagates_cancellation_and_releases_write_lock(
    tmp_path: Path,
) -> None:
    engine = _ConcurrentEngine()
    runtime = ServerRuntime(
        config=MCPServerConfig(database=tmp_path / "index.db"),
        engine=engine,  # type: ignore[arg-type]
    )

    cancelled = asyncio.create_task(
        runtime.index_site("https://example.com/one", max_pages=1, max_depth=0)
    )
    await engine.index_started.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    engine.release_index.set()
    completed = await asyncio.wait_for(
        runtime.index_site(
            "https://example.com/two",
            max_pages=1,
            max_depth=0,
        ),
        timeout=1,
    )
    assert completed.indexed_documents == 1


@pytest.mark.asyncio
async def test_lifecycle_cancels_active_call_before_closing_engine(
    tmp_path: Path,
) -> None:
    class ShutdownEngine(_ConcurrentEngine):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = False
            self.closed = False
            self.closed_while_active = False

        async def ingest_url(
            self,
            _url: str,
            **_kwargs: object,
        ) -> IndexingResult:
            self.index_active += 1
            self.index_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            finally:
                self.index_active -= 1

        async def close(self) -> None:
            self.closed_while_active = self.index_active > 0
            self.closed = True

    engine = ShutdownEngine()

    def factory(_path: Path, _policy: object) -> ShutdownEngine:
        return engine

    server = create_server(
        MCPServerConfig(database=tmp_path / "index.db"),
        engine_factory=factory,  # type: ignore[arg-type]
    )
    client = Client(server)
    await client.__aenter__()
    call = asyncio.create_task(
        client.call_tool(
            "index_site",
            {
                "url": "https://example.com/",
                "max_pages": 1,
                "max_depth": 0,
            },
        )
    )
    await engine.index_started.wait()

    await asyncio.wait_for(client.__aexit__(None, None, None), timeout=2)

    assert call.done()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert engine.cancelled
    assert engine.closed
    assert not engine.closed_while_active


@pytest.mark.asyncio
async def test_unexpected_tool_error_is_sanitized_and_server_remains_live(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingEngine(_ConcurrentEngine):
        async def search(self, _query: str, *, limit: int) -> list[object]:
            del limit
            raise RuntimeError(f"secret traceback detail at {tmp_path / 'private.db'}")

    engine = FailingEngine()

    def factory(_path: Path, _policy: object) -> FailingEngine:
        return engine

    server = create_server(
        MCPServerConfig(database=tmp_path / "index.db"),
        engine_factory=factory,  # type: ignore[arg-type]
    )

    async with Client(server) as client:
        failure = await client.call_tool(
            "search_index",
            {"query": "RetryStrategy", "limit": 1},
        )
        info = await client.call_tool("get_index_info", {})

    assert failure.is_error
    message = failure.content[0].text
    assert "failed unexpectedly" in message
    assert "private.db" not in message
    assert not info.is_error
    diagnostics = capsys.readouterr().err
    assert "unexpected_error" in diagnostics
    assert "private.db" not in diagnostics


@pytest.mark.asyncio
async def test_locked_storage_error_is_actionable_and_does_not_kill_server(
    tmp_path: Path,
) -> None:
    class LockedEngine(_ConcurrentEngine):
        async def ingest_url(
            self,
            _url: str,
            **_kwargs: object,
        ) -> IndexingResult:
            raise sqlite3.OperationalError(
                f"database is locked at {tmp_path / 'private.db'}"
            )

    engine = LockedEngine()

    def factory(_path: Path, _policy: object) -> LockedEngine:
        return engine

    server = create_server(
        MCPServerConfig(database=tmp_path / "index.db"),
        engine_factory=factory,  # type: ignore[arg-type]
    )
    async with Client(server) as client:
        failure = await client.call_tool(
            "index_site",
            {
                "url": "https://example.com/",
                "max_pages": 1,
                "max_depth": 0,
            },
        )
        info = await client.call_tool("get_index_info", {})

    assert failure.is_error
    assert "locked" in failure.content[0].text
    assert "private.db" not in failure.content[0].text
    assert not info.is_error
