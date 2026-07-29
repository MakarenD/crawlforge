"""Tests for the deduplicated SQLite full-text context index."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

from crawlforge.context_index import FTS5UnavailableError, SQLiteContextIndex
from crawlforge.context_models import IndexingResult, SourceDocument, TextChunk


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _document(
    name: str,
    *,
    canonical_url: str | None = None,
    title: str = "CrawlForge guide",
    text: str = "Configure bounded retries for AsyncCrawler.",
    markdown: str | None = None,
    content_hash: str | None = None,
    source_size_bytes: int | None = None,
    source_estimated_tokens: int | None = None,
) -> SourceDocument:
    url = canonical_url or f"https://example.com/{name}"
    rendered_markdown = markdown if markdown is not None else text
    return SourceDocument(
        id=_sha256(url),
        url=url,
        canonical_url=url,
        title=title,
        text=text,
        markdown=rendered_markdown,
        status_code=200,
        content_type="text/html; charset=utf-8",
        fetched_at=datetime(2026, 7, 28, 12, 0, tzinfo=UTC),
        content_hash=content_hash or _sha256(f"{title}\0{text}\0{rendered_markdown}"),
        metadata={"language": "en"},
        source_size_bytes=source_size_bytes or len(text.encode()),
        cleaned_size_bytes=len(text),
        source_estimated_tokens=source_estimated_tokens or max(1, len(text) // 4),
        cleaned_estimated_tokens=max(1, len(text) // 4),
        blocks=(),
    )


def _chunk(
    document: SourceDocument,
    ordinal: int,
    text: str,
    *,
    heading_path: tuple[str, ...] = ("Guide",),
) -> TextChunk:
    content_hash = _sha256("\0".join((document.title, *heading_path, text)))
    return TextChunk(
        id=_sha256(f"{document.id}\0{ordinal}\0{content_hash}"),
        document_id=document.id,
        ordinal=ordinal,
        source_url=document.url,
        document_title=document.title,
        heading_path=heading_path,
        text=text,
        size_chars=len(text),
        estimated_tokens=max(1, len(text) // 4),
        content_hash=content_hash,
    )


async def _counts(path: Path) -> dict[str, int]:
    async with aiosqlite.connect(path) as connection:
        tables = (
            "document_contents",
            "documents",
            "chunks",
            "content_chunks",
            "chunk_provenance",
            "chunk_fts",
        )
        return {
            table: int(
                (await connection.execute_fetchall(f"SELECT COUNT(*) FROM {table}"))[0][
                    0
                ]
            )
            for table in tables
        }


async def _downgrade_schema_to_v1(path: Path) -> None:
    async with aiosqlite.connect(path) as connection:
        await connection.executescript(
            """
            DROP TABLE embedding_sessions;
            DROP TABLE chunk_embeddings;
            DROP TABLE embedding_models;
            DROP TABLE chunk_provenance;
            DROP INDEX idx_documents_content_hash;
            CREATE INDEX idx_documents_content_hash
            ON documents(content_hash);
            DROP INDEX idx_content_chunks_chunk_id;
            CREATE INDEX idx_content_chunks_chunk_id
            ON content_chunks(chunk_id);
            UPDATE context_schema_version SET version = 1 WHERE singleton = 1;
            """
        )
        await connection.commit()


async def _downgrade_schema_to_v2(path: Path) -> None:
    async with aiosqlite.connect(path) as connection:
        await connection.executescript(
            """
            DROP TABLE embedding_sessions;
            DROP TABLE chunk_embeddings;
            DROP TABLE embedding_models;
            UPDATE context_schema_version SET version = 2 WHERE singleton = 1;
            """
        )
        await connection.commit()


@pytest.mark.asyncio
async def test_initialize_creates_versioned_normalized_schema_and_is_idempotent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "context.sqlite3"
    index = SQLiteContextIndex(output)

    await index.initialize()
    await index.initialize()
    await index.close()

    reopened = SQLiteContextIndex(output)
    await reopened.initialize()
    await reopened.close()

    async with aiosqlite.connect(output) as connection:
        table_rows = await connection.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
        version = await connection.execute_fetchall(
            "SELECT singleton, version FROM context_schema_version"
        )
        foreign_keys = await connection.execute_fetchall(
            "PRAGMA foreign_key_list(documents)"
        )
    table_names = {str(row[0]) for row in table_rows}
    assert {
        "context_schema_version",
        "document_contents",
        "documents",
        "chunks",
        "content_chunks",
        "chunk_provenance",
        "index_sessions",
        "embedding_models",
        "chunk_embeddings",
        "embedding_sessions",
        "chunk_fts",
    } <= table_names
    assert version == [(1, 3)]
    assert any(str(row[2]) == "document_contents" for row in foreign_keys)


@pytest.mark.asyncio
async def test_get_index_info_reports_empty_and_populated_bounded_summaries(
    tmp_path: Path,
) -> None:
    output = tmp_path / "info.sqlite3"
    index = SQLiteContextIndex(output)

    empty = await index.get_index_info()
    document = _document("info")
    result = await index.index_documents(
        [(document, [_chunk(document, 0, document.text)])]
    )
    populated = await index.get_index_info()
    await index.close()

    assert empty.schema_version == 3
    assert empty.document_count == 0
    assert empty.chunk_count == 0
    assert empty.last_indexed_at is None
    assert empty.last_session_summary is None
    assert empty.database_ready
    assert empty.fts5_available
    assert populated.document_count == 1
    assert populated.chunk_count == 1
    assert populated.last_indexed_at is not None
    assert populated.last_session_summary is not None
    assert populated.last_session_summary.session_id == result.session_id
    assert populated.last_session_summary.documents_seen == 1
    assert populated.last_session_summary.finished_at is not None


@pytest.mark.asyncio
async def test_initialize_migrates_v1_provenance_and_indexes_idempotently(
    tmp_path: Path,
) -> None:
    output = tmp_path / "migrate-v1.sqlite3"
    document = _document("migrated")
    index = SQLiteContextIndex(output)
    await index.index_documents([(document, [_chunk(document, 0, document.text)])])
    await index.close()
    await _downgrade_schema_to_v1(output)

    migrated = SQLiteContextIndex(output)
    await migrated.initialize()
    hits = await migrated.search("AsyncCrawler")
    await migrated.close()
    reopened = SQLiteContextIndex(output)
    await reopened.initialize()
    repeated_hits = await reopened.search("AsyncCrawler")
    await reopened.close()

    assert hits == repeated_hits
    assert hits[0].source.document_id == document.id
    async with aiosqlite.connect(output) as connection:
        version = await connection.execute_fetchall(
            "SELECT version FROM context_schema_version WHERE singleton = 1"
        )
        provenance = await connection.execute_fetchall(
            "SELECT COUNT(*) FROM chunk_provenance"
        )
        document_index = await connection.execute_fetchall(
            "PRAGMA index_info(idx_documents_content_hash)"
        )
        chunk_index = await connection.execute_fetchall(
            "PRAGMA index_info(idx_content_chunks_chunk_id)"
        )
    assert version == [(3,)]
    assert provenance == [(1,)]
    assert [str(row[2]) for row in document_index] == [
        "content_hash",
        "canonical_url",
        "document_id",
    ]
    assert [str(row[2]) for row in chunk_index] == [
        "chunk_id",
        "content_hash",
        "ordinal",
    ]


@pytest.mark.asyncio
async def test_initialize_migrates_v2_to_semantic_schema_idempotently(
    tmp_path: Path,
) -> None:
    output = tmp_path / "migrate-v2.sqlite3"
    document = _document("semantic-migration")
    initial = SQLiteContextIndex(output)
    await initial.index_documents([(document, [_chunk(document, 0, document.text)])])
    await initial.close()
    await _downgrade_schema_to_v2(output)

    migrated = SQLiteContextIndex(output)
    await migrated.initialize()
    await migrated.initialize()
    hits = await migrated.search("AsyncCrawler")
    await migrated.close()

    async with aiosqlite.connect(output) as connection:
        version = await connection.execute_fetchall(
            "SELECT version FROM context_schema_version WHERE singleton = 1"
        )
        tables = await connection.execute_fetchall(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name LIKE '%embedding%'
            ORDER BY name
            """
        )
        foreign_keys = await connection.execute_fetchall(
            "PRAGMA foreign_key_list(chunk_embeddings)"
        )

    assert version == [(3,)]
    assert [str(row[0]) for row in tables] == [
        "chunk_embeddings",
        "embedding_models",
        "embedding_sessions",
    ]
    assert {str(row[2]) for row in foreign_keys} == {
        "chunks",
        "embedding_models",
    }
    assert hits[0].source.document_id == document.id


@pytest.mark.asyncio
async def test_v1_migration_failure_rolls_back_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "migration-rollback.sqlite3"
    document = _document("migration-rollback")
    initial = SQLiteContextIndex(output)
    await initial.index_documents([(document, [_chunk(document, 0, document.text)])])
    await initial.close()
    await _downgrade_schema_to_v1(output)
    original_execute = aiosqlite.Connection.execute

    async def fail_version_update(
        connection: aiosqlite.Connection,
        sql: str,
        parameters: tuple[object, ...] | None = None,
    ) -> aiosqlite.Cursor:
        if "UPDATE context_schema_version" in sql:
            raise aiosqlite.OperationalError("migration commit failed")
        if parameters is None:
            return await original_execute(connection, sql)
        return await original_execute(connection, sql, parameters)

    monkeypatch.setattr(aiosqlite.Connection, "execute", fail_version_update)
    failing = SQLiteContextIndex(output)
    with pytest.raises(aiosqlite.OperationalError, match="migration commit failed"):
        await failing.initialize()
    assert failing._connection is None

    async with aiosqlite.connect(output) as connection:
        version = await connection.execute_fetchall(
            "SELECT version FROM context_schema_version WHERE singleton = 1"
        )
        tables = await connection.execute_fetchall(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'chunk_provenance'
            """
        )
        document_index = await connection.execute_fetchall(
            "PRAGMA index_info(idx_documents_content_hash)"
        )
    assert version == [(1,)]
    assert tables == []
    assert [str(row[2]) for row in document_index] == ["content_hash"]

    monkeypatch.setattr(aiosqlite.Connection, "execute", original_execute)
    retried = SQLiteContextIndex(output)
    await retried.initialize()
    hits = await retried.search("AsyncCrawler")
    await retried.close()
    assert hits[0].source.document_id == document.id


@pytest.mark.asyncio
async def test_batch_index_records_session_and_searchable_relations(
    tmp_path: Path,
) -> None:
    output = tmp_path / "batch.sqlite3"
    first = _document("first", text="AsyncCrawler uses bounded retries.")
    second = _document("second", text="SQLite FTS5 provides lexical search.")
    index = SQLiteContextIndex(output)
    session_id = await index.start_session()

    result = await index.index_documents(
        [
            (first, [_chunk(first, 0, first.text)]),
            (second, [_chunk(second, 0, second.text)]),
        ],
        session_id=session_id,
    )
    await index.finish_session(replace(result, cleaning_time_ms=1.25))
    await index.close()

    assert result.documents_seen == 2
    assert result.documents_indexed == 2
    assert result.chunks_indexed == 2
    assert await _counts(output) == {
        "document_contents": 2,
        "documents": 2,
        "chunks": 2,
        "content_chunks": 2,
        "chunk_provenance": 2,
        "chunk_fts": 2,
    }
    async with aiosqlite.connect(output) as connection:
        session = await connection.execute_fetchall(
            """
            SELECT documents_seen, chunks_indexed, cleaning_time_ms, finished_at
            FROM index_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )
    assert session[0][:3] == (2, 2, 1.25)
    assert session[0][3] is not None


@pytest.mark.asyncio
async def test_exact_documents_share_content_mappings_with_distinct_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "exact-documents.sqlite3"
    first = _document(
        "a",
        source_size_bytes=300,
        source_estimated_tokens=75,
    )
    second = _document(
        "b",
        title=first.title,
        text=first.text,
        markdown=first.markdown,
        content_hash=first.content_hash,
        source_size_bytes=900,
        source_estimated_tokens=225,
    )
    second = replace(
        second,
        cleaned_size_bytes=999,
        cleaned_estimated_tokens=999,
    )
    second_chunk = replace(
        _chunk(second, 0, second.text),
        estimated_tokens=999,
    )
    index = SQLiteContextIndex(output)

    result = await index.index_documents(
        [
            (first, [_chunk(first, 0, first.text)]),
            (second, [second_chunk]),
        ]
    )
    hits = await index.search("AsyncCrawler")
    await index.close()

    assert result.documents_indexed == 1
    assert result.duplicate_documents == 1
    assert result.chunks_indexed == 1
    assert result.duplicate_chunks == 1
    assert await _counts(output) == {
        "document_contents": 1,
        "documents": 2,
        "chunks": 1,
        "content_chunks": 1,
        "chunk_provenance": 1,
        "chunk_fts": 1,
    }
    assert hits[0].source.canonical_url == first.canonical_url
    assert hits[0].source.source_size_bytes == 300
    assert hits[0].chunk.estimated_tokens == 999


@pytest.mark.asyncio
async def test_partially_overlapping_documents_share_only_exact_contextual_chunks(
    tmp_path: Path,
) -> None:
    output = tmp_path / "partial.sqlite3"
    first = _document("first", text="Shared section. Unique alpha section.")
    second = _document("second", text="Shared section. Unique beta section.")
    shared_first = _chunk(first, 0, "Shared section.")
    shared_second = _chunk(second, 0, "Shared section.")
    index = SQLiteContextIndex(output)

    result = await index.index_documents(
        [
            (first, [shared_first, _chunk(first, 1, "Unique alpha section.")]),
            (second, [shared_second, _chunk(second, 1, "Unique beta section.")]),
        ]
    )
    await index.close()

    assert shared_first.content_hash == shared_second.content_hash
    assert result.documents_indexed == 2
    assert result.chunks_indexed == 3
    assert result.duplicate_chunks == 1
    assert await _counts(output) == {
        "document_contents": 2,
        "documents": 2,
        "chunks": 3,
        "content_chunks": 4,
        "chunk_provenance": 3,
        "chunk_fts": 3,
    }


@pytest.mark.asyncio
async def test_reindex_replaces_chunk_mapping_and_derived_token_estimates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reconfigured.sqlite3"
    document = _document(
        "reconfigured",
        text="Alpha configuration. Beta configuration.",
    )
    original_chunk = _chunk(document, 0, document.text)
    reestimated = replace(
        document,
        cleaned_estimated_tokens=999,
    )
    replacement_chunks = (
        replace(
            _chunk(reestimated, 0, "Alpha configuration."),
            estimated_tokens=20,
        ),
        replace(
            _chunk(reestimated, 1, "Beta configuration."),
            estimated_tokens=30,
        ),
    )
    index = SQLiteContextIndex(output)

    await index.index_documents([(document, [original_chunk])])
    result = await index.index_documents([(reestimated, replacement_chunks)])
    hits = await index.search("Beta")
    await index.close()

    assert result.documents_indexed == 0
    assert result.duplicate_documents == 1
    assert result.chunks_indexed == 2
    assert result.duplicate_chunks == 0
    assert len(hits) == 1
    assert hits[0].chunk.text == "Beta configuration."
    assert hits[0].chunk.estimated_tokens == 30
    assert await _counts(output) == {
        "document_contents": 1,
        "documents": 1,
        "chunks": 2,
        "content_chunks": 2,
        "chunk_provenance": 2,
        "chunk_fts": 2,
    }
    async with aiosqlite.connect(output) as connection:
        metrics = await connection.execute_fetchall(
            """
            SELECT cleaned_estimated_tokens
            FROM document_contents
            WHERE content_hash = ?
            """,
            (document.content_hash,),
        )
    assert metrics == [(999,)]


@pytest.mark.asyncio
async def test_document_update_removes_stale_content_chunks_and_fts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "update.sqlite3"
    original = _document("changing", text="LegacyRetrier is configured here.")
    updated = _document(
        "changing",
        canonical_url=original.canonical_url,
        text="AdaptiveBackoff is configured here.",
    )
    index = SQLiteContextIndex(output)

    await index.index_documents([(original, [_chunk(original, 0, original.text)])])
    await index.index_documents([(updated, [_chunk(updated, 0, updated.text)])])

    assert await index.search("LegacyRetrier") == []
    current = await index.search("AdaptiveBackoff")
    await index.close()

    assert len(current) == 1
    assert current[0].source.document_id == original.id
    assert await _counts(output) == {
        "document_contents": 1,
        "documents": 1,
        "chunks": 1,
        "content_chunks": 1,
        "chunk_provenance": 1,
        "chunk_fts": 1,
    }


@pytest.mark.asyncio
async def test_update_preserves_chunk_still_referenced_by_another_content(
    tmp_path: Path,
) -> None:
    output = tmp_path / "shared-update.sqlite3"
    first = _document("first", text="SharedKeyword and obsolete content.")
    second = _document("second", text="SharedKeyword and retained content.")
    shared_first = _chunk(first, 0, "SharedKeyword")
    shared_second = _chunk(second, 0, "SharedKeyword")
    updated = _document(
        "first",
        canonical_url=first.canonical_url,
        text="ReplacementKeyword",
    )
    index = SQLiteContextIndex(output)

    await index.index_documents([(first, [shared_first]), (second, [shared_second])])
    await index.index_documents([(updated, [_chunk(updated, 0, updated.text)])])
    shared_hits = await index.search("SharedKeyword")
    await index.close()

    assert len(shared_hits) == 1
    assert shared_hits[0].source.document_id == second.id
    assert await _counts(output) == {
        "document_contents": 2,
        "documents": 2,
        "chunks": 2,
        "content_chunks": 2,
        "chunk_provenance": 2,
        "chunk_fts": 2,
    }


@pytest.mark.asyncio
async def test_search_matches_technical_terms_and_ranks_title_hits_first(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ranking.sqlite3"
    title_hit = _document(
        "title",
        title="AsyncCrawler reference",
        text="Transport configuration.",
    )
    text_hit = _document(
        "text",
        title="Transport reference",
        text="AsyncCrawler transport configuration.",
    )
    index = SQLiteContextIndex(output)
    await index.index_documents(
        [
            (text_hit, [_chunk(text_hit, 0, text_hit.text)]),
            (title_hit, [_chunk(title_hit, 0, title_hit.text)]),
        ]
    )

    hits = await index.search("AsyncCrawler", limit=2)
    await index.close()

    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].source.document_id == title_hit.id
    assert hits[0].bm25_score <= hits[1].bm25_score


@pytest.mark.asyncio
async def test_search_uses_literal_lexical_tokens_not_only_an_exact_phrase(
    tmp_path: Path,
) -> None:
    document = _document(
        "natural-language",
        text="Retries use exponential delay and a strict attempt budget.",
    )
    index = SQLiteContextIndex(tmp_path / "natural-language.sqlite3")
    await index.index_documents([(document, [_chunk(document, 0, document.text)])])

    hits = await index.search("How are retries configured?")
    await index.close()

    assert len(hits) == 1
    assert hits[0].source.document_id == document.id


@pytest.mark.asyncio
async def test_empty_index_query_and_arbitrary_special_query_are_safe(
    tmp_path: Path,
) -> None:
    index = SQLiteContextIndex(tmp_path / "empty.sqlite3")

    assert await index.search("anything") == []
    assert await index.search("") == []
    assert await index.search('  "*:-()[]{}  ') == []
    with pytest.raises(ValueError, match="limit"):
        await index.search("anything", limit=0)
    await index.close()


@pytest.mark.asyncio
async def test_missing_fts5_is_reported_with_specific_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_executescript = aiosqlite.Connection.executescript

    async def without_fts5(
        connection: aiosqlite.Connection,
        sql_script: str,
    ) -> aiosqlite.Cursor:
        if "CREATE VIRTUAL TABLE" in sql_script:
            raise aiosqlite.OperationalError("no such module: fts5")
        return await original_executescript(connection, sql_script)

    monkeypatch.setattr(aiosqlite.Connection, "executescript", without_fts5)
    index = SQLiteContextIndex(tmp_path / "no-fts.sqlite3")

    with pytest.raises(FTS5UnavailableError, match="FTS5 support is required"):
        await index.initialize()

    assert index._connection is None
    await index.close()


@pytest.mark.asyncio
async def test_batch_failure_rolls_back_documents_chunks_fts_and_counters(
    tmp_path: Path,
) -> None:
    output = tmp_path / "rollback.sqlite3"
    first = _document("first", canonical_url="https://example.com/collision")
    second = _document(
        "second",
        canonical_url="https://example.com/collision",
        text="Different content with the same canonical URL.",
    )
    second = replace(second, id=_sha256("different-stable-id"))
    index = SQLiteContextIndex(output)
    session_id = await index.start_session()

    with pytest.raises(aiosqlite.IntegrityError):
        await index.index_documents(
            [
                (first, [_chunk(first, 0, first.text)]),
                (second, [_chunk(second, 0, second.text)]),
            ],
            session_id=session_id,
        )
    await index.close()

    assert await _counts(output) == {
        "document_contents": 0,
        "documents": 0,
        "chunks": 0,
        "content_chunks": 0,
        "chunk_provenance": 0,
        "chunk_fts": 0,
    }
    async with aiosqlite.connect(output) as connection:
        counters = await connection.execute_fetchall(
            """
            SELECT documents_seen, documents_indexed, chunks_indexed
            FROM index_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )
    assert counters == [(0, 0, 0)]


@pytest.mark.asyncio
async def test_cancelled_committed_batch_cannot_be_erased_by_stale_session_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "cancelled-session.sqlite3"
    document = _document("cancelled")
    index = SQLiteContextIndex(output)
    session_id = await index.start_session()
    transaction_committed = asyncio.Event()
    release_result = asyncio.Event()
    original_transaction = index._index_batch_transaction

    async def hold_result_after_commit(
        connection: aiosqlite.Connection,
        documents: tuple[tuple[SourceDocument, tuple[TextChunk, ...]], ...],
        *,
        session_id: str,
    ) -> IndexingResult:
        result = await original_transaction(
            connection,
            documents,
            session_id=session_id,
        )
        transaction_committed.set()
        await release_result.wait()
        return result

    monkeypatch.setattr(index, "_index_batch_transaction", hold_result_after_commit)
    indexing = asyncio.create_task(
        index.index_documents(
            [(document, [_chunk(document, 0, document.text)])],
            session_id=session_id,
        )
    )
    await asyncio.wait_for(transaction_committed.wait(), timeout=2)
    indexing.cancel()
    release_result.set()

    with pytest.raises(asyncio.CancelledError):
        await indexing

    stale_result = IndexingResult(
        session_id=session_id,
        documents_seen=0,
        documents_indexed=0,
        duplicate_documents=0,
        chunks_indexed=0,
        duplicate_chunks=0,
        source_size_bytes=0,
        cleaned_size_bytes=0,
        source_estimated_tokens=0,
        cleaned_estimated_tokens=0,
        cleaning_time_ms=0.0,
        indexing_time_ms=0.0,
    )
    await index.finish_session(stale_result)
    await index.close()

    assert await _counts(output) == {
        "document_contents": 1,
        "documents": 1,
        "chunks": 1,
        "content_chunks": 1,
        "chunk_provenance": 1,
        "chunk_fts": 1,
    }
    async with aiosqlite.connect(output) as connection:
        counters = await connection.execute_fetchall(
            """
            SELECT documents_seen, documents_indexed, chunks_indexed, finished_at
            FROM index_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )
    assert counters[0][:3] == (1, 1, 1)
    assert counters[0][3] is not None


@pytest.mark.asyncio
async def test_close_is_idempotent_and_all_operations_reject_closed_index(
    tmp_path: Path,
) -> None:
    index = SQLiteContextIndex(tmp_path / "closed.sqlite3")
    await index.initialize()
    await index.close()
    await index.close()
    empty_result = IndexingResult(
        session_id="missing",
        documents_seen=0,
        documents_indexed=0,
        duplicate_documents=0,
        chunks_indexed=0,
        duplicate_chunks=0,
        source_size_bytes=0,
        cleaned_size_bytes=0,
        source_estimated_tokens=0,
        cleaned_estimated_tokens=0,
        cleaning_time_ms=0.0,
        indexing_time_ms=0.0,
    )

    with pytest.raises(RuntimeError, match="closed"):
        await index.search("")
    with pytest.raises(RuntimeError, match="closed"):
        await index.start_session()
    with pytest.raises(RuntimeError, match="closed"):
        await index.index_documents([])
    with pytest.raises(RuntimeError, match="closed"):
        await index.finish_session(empty_result)


@pytest.mark.asyncio
async def test_shared_chunk_uses_deterministic_provenance_and_chunk_identity(
    tmp_path: Path,
) -> None:
    later = _document(
        "later",
        canonical_url="https://example.com/z-source",
    )
    earlier = _document(
        "earlier",
        canonical_url="https://example.com/a-source",
        title=later.title,
        text=later.text,
        markdown=later.markdown,
        content_hash=later.content_hash,
    )
    index = SQLiteContextIndex(tmp_path / "provenance.sqlite3")
    await index.index_documents(
        [
            (later, [_chunk(later, 0, later.text)]),
            (earlier, [_chunk(earlier, 0, earlier.text)]),
        ]
    )

    first_search = await index.search("AsyncCrawler")
    second_search = await index.search("AsyncCrawler")
    await index.close()

    assert first_search == second_search
    assert first_search[0].source.document_id == earlier.id
    assert first_search[0].chunk.document_id == earlier.id
    assert first_search[0].chunk.id == _sha256(
        f"{earlier.id}\0{first_search[0].chunk.ordinal}\0"
        f"{first_search[0].chunk.content_hash}"
    )


@pytest.mark.asyncio
async def test_provenance_hydration_is_direct_under_large_duplicate_fanout(
    tmp_path: Path,
) -> None:
    output = tmp_path / "provenance-fanout.sqlite3"
    canonical = _document(
        "canonical",
        canonical_url="https://example.com/000-canonical",
    )
    documents = [(canonical, [_chunk(canonical, 0, canonical.text)])]
    for number in range(1, 101):
        duplicate = _document(
            f"duplicate-{number}",
            title=canonical.title,
            text=canonical.text,
            markdown=canonical.markdown,
            content_hash=canonical.content_hash,
        )
        documents.append((duplicate, [_chunk(duplicate, 0, duplicate.text)]))
    index = SQLiteContextIndex(output)
    await index.index_documents(documents)

    hits = await index.search("AsyncCrawler", limit=1)
    await index.close()

    assert len(hits) == 1
    assert hits[0].source.document_id == canonical.id
    async with aiosqlite.connect(output) as connection:
        plan_rows = await connection.execute_fetchall(
            """
            EXPLAIN QUERY PLAN
            SELECT chunk_provenance.document_id
            FROM chunk_provenance
            JOIN documents
                ON documents.document_id = chunk_provenance.document_id
            WHERE chunk_provenance.chunk_id IN (?)
            """,
            (1,),
        )
    query_plan = "\n".join(str(row[3]) for row in plan_rows)
    assert "SEARCH chunk_provenance USING INTEGER PRIMARY KEY" in query_plan
    assert "SCAN content_chunks" not in query_plan
    assert "USE TEMP B-TREE" not in query_plan
