"""SQLite-backed full-text index for reusable crawl context."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TypeVar

import aiosqlite

from crawlforge.context_models import (
    IndexInfo,
    IndexingResult,
    IndexSessionSummary,
    SearchHit,
    SourceDocument,
    SourceReference,
    TextChunk,
)
from crawlforge.semantic_models import (
    EmbeddingModelInfo,
    EmbeddingVector,
    SemanticChunkRecord,
    SemanticEmbeddingSnapshot,
    SemanticIndexIncompatibleError,
    SemanticIndexInfo,
    SemanticIndexingResult,
    SemanticIndexNotReadyError,
    SemanticIndexPlan,
    StoredChunkEmbedding,
    deserialize_embedding_vector,
    serialize_embedding_vector,
)

_SCHEMA_VERSION = 3
_T = TypeVar("_T")


class FTS5UnavailableError(RuntimeError):
    """Raised when the active SQLite build does not provide FTS5."""


class SQLiteContextIndex:
    """Store deduplicated documents and search their contextual chunks."""

    def __init__(self, path: str | Path) -> None:
        """Configure a SQLite database path without opening it eagerly."""
        self.path: str | Path = ":memory:" if str(path) == ":memory:" else Path(path)
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def __aenter__(self) -> SQLiteContextIndex:
        """Initialize and return the index."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the index when leaving its asynchronous context."""
        await self.close()

    async def initialize(self) -> None:
        """Create and validate the versioned schema."""
        async with self._lock:
            await self._ensure_connection_locked()

    async def start_session(self) -> str:
        """Create a running indexing session and return its identifier."""
        session_id = uuid.uuid4().hex
        async with self._lock:
            connection = await self._ensure_connection_locked()

            async def insert_session(active: aiosqlite.Connection) -> None:
                await active.execute(
                    """
                    INSERT INTO index_sessions (session_id, started_at)
                    VALUES (?, ?)
                    """,
                    (session_id, _utc_now()),
                )

            await self._run_transaction_locked(connection, insert_session)
        return session_id

    async def index_documents(
        self,
        documents: Sequence[tuple[SourceDocument, Sequence[TextChunk]]],
        *,
        session_id: str | None = None,
    ) -> IndexingResult:
        """Atomically index a batch and return its session-counter delta."""
        prepared = tuple(
            (
                document,
                tuple(sorted(chunks, key=lambda chunk: (chunk.ordinal, chunk.id))),
            )
            for document, chunks in documents
        )
        self._validate_batch(prepared)

        owned_session = session_id is None
        active_session = session_id
        if active_session is None:
            active_session = await self.start_session()

        async with self._lock:
            connection = await self._ensure_connection_locked()
            index_task = asyncio.create_task(
                self._index_batch_transaction(
                    connection,
                    prepared,
                    session_id=active_session,
                )
            )
            try:
                result = await asyncio.shield(index_task)
            except asyncio.CancelledError as cancelled:
                try:
                    result = await index_task
                except Exception as index_error:
                    raise cancelled from index_error
                if owned_session:
                    await self._finish_session_locked(connection, result)
                raise

            if owned_session:
                await self._finish_session_locked(connection, result)
            return result

    async def finish_session(self, result: IndexingResult) -> None:
        """Record aggregate counters and completion time for one session."""
        async with self._lock:
            connection = await self._ensure_connection_locked()
            await self._finish_session_locked(connection, result)

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        """Return the most relevant contextual chunks for a literal query."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        async with self._lock:
            connection = await self._ensure_connection_locked()
            if not query.strip():
                return []

            match_query = _literal_match_query(query)
            if match_query is None:
                return []
            candidate_rows = list(
                await connection.execute_fetchall(
                    """
                    SELECT
                        chunks.id,
                        chunks.content_hash,
                        chunks.document_title,
                        chunks.heading_path,
                        chunks.text,
                        chunks.size_chars,
                        chunks.estimated_tokens,
                        bm25(chunk_fts, 1.0, 5.0, 3.0) AS bm25_score
                    FROM chunk_fts
                    JOIN chunks ON chunks.id = chunk_fts.rowid
                    WHERE chunk_fts MATCH ?
                    ORDER BY bm25_score ASC, chunks.content_hash ASC
                    LIMIT ?
                    """,
                    (match_query, limit),
                )
            )
            if not candidate_rows:
                return []

            chunk_ids = tuple(_database_int(row[0]) for row in candidate_rows)
            placeholders = ", ".join("?" for _ in chunk_ids)
            source_rows = list(
                await connection.execute_fetchall(
                    f"""
                    SELECT
                        chunk_provenance.chunk_id,
                        chunk_provenance.ordinal,
                        documents.document_id,
                        documents.url,
                        documents.canonical_url,
                        document_contents.title,
                        documents.status_code,
                        documents.content_type,
                        documents.fetched_at,
                        documents.source_size_bytes,
                        documents.source_estimated_tokens
                    FROM chunk_provenance
                    JOIN documents
                        ON documents.document_id =
                            chunk_provenance.document_id
                    JOIN document_contents
                        ON document_contents.content_hash = documents.content_hash
                    WHERE chunk_provenance.chunk_id IN ({placeholders})
                    ORDER BY chunk_provenance.chunk_id ASC
                    """,
                    chunk_ids,
                )
            )

        sources: dict[int, aiosqlite.Row] = {}
        for row in source_rows:
            chunk_id = _database_int(row[0])
            sources.setdefault(chunk_id, row)

        hits: list[SearchHit] = []
        for rank, row in enumerate(candidate_rows, start=1):
            chunk_id = _database_int(row[0])
            source_row = sources.get(chunk_id)
            if source_row is None:
                continue
            ordinal = _database_int(source_row[1])
            document_id = _database_text(source_row[2])
            source = SourceReference(
                document_id=document_id,
                url=_database_text(source_row[3]),
                canonical_url=_database_text(source_row[4]),
                title=_database_text(source_row[5]),
                status_code=_database_int(source_row[6]),
                content_type=_database_text(source_row[7]),
                fetched_at=datetime.fromisoformat(_database_text(source_row[8])),
                source_size_bytes=_database_int(source_row[9]),
                source_estimated_tokens=_database_int(source_row[10]),
            )
            content_hash = _database_text(row[1])
            chunk = TextChunk(
                id=_document_chunk_id(document_id, ordinal, content_hash),
                document_id=document_id,
                ordinal=ordinal,
                source_url=source.url,
                document_title=_database_text(row[2]),
                heading_path=_decode_heading_path(_database_text(row[3])),
                text=_database_text(row[4]),
                size_chars=_database_int(row[5]),
                estimated_tokens=_database_int(row[6]),
                content_hash=content_hash,
            )
            hits.append(
                SearchHit(
                    chunk=chunk,
                    rank=rank,
                    bm25_score=_database_float(row[7]),
                    source=source,
                )
            )
        return hits

    async def prepare_semantic_index(
        self,
        model: EmbeddingModelInfo,
    ) -> SemanticIndexPlan:
        """Register a model and remove stale vectors before inference."""
        async with self._lock:
            connection = await self._ensure_connection_locked()

            async def prepare(active: aiosqlite.Connection) -> SemanticIndexPlan:
                await self._ensure_embedding_model(active, model)
                invalidation_rows = list(
                    await active.execute_fetchall(
                        """
                        SELECT pending_invalidations
                        FROM embedding_models
                        WHERE model_fingerprint = ?
                        """,
                        (model.fingerprint,),
                    )
                )
                if len(invalidation_rows) != 1:
                    raise RuntimeError("embedding invalidation state is unavailable")
                pending_invalidations = _database_int(invalidation_rows[0][0])
                stale_cursor = await active.execute(
                    """
                    DELETE FROM chunk_embeddings
                    WHERE model_fingerprint = ?
                      AND (
                        dimension != ?
                        OR dtype != ?
                        OR NOT EXISTS (
                            SELECT 1
                            FROM chunks
                            WHERE chunks.id = chunk_embeddings.chunk_id
                              AND chunks.content_hash =
                                  chunk_embeddings.chunk_content_hash
                        )
                      )
                    """,
                    (model.fingerprint, model.dimension, model.dtype),
                )
                await active.execute(
                    """
                    UPDATE embedding_models
                    SET pending_invalidations = 0
                    WHERE model_fingerprint = ?
                    """,
                    (model.fingerprint,),
                )
                counts = list(
                    await active.execute_fetchall(
                        """
                        SELECT
                            COUNT(chunks.id),
                            COUNT(chunk_embeddings.chunk_id)
                        FROM chunks
                        JOIN chunk_provenance
                            ON chunk_provenance.chunk_id = chunks.id
                        LEFT JOIN chunk_embeddings
                            ON chunk_embeddings.chunk_id = chunks.id
                           AND chunk_embeddings.model_fingerprint = ?
                        """,
                        (model.fingerprint,),
                    )
                )
                if len(counts) != 1:
                    raise RuntimeError("semantic index plan is unavailable")
                return SemanticIndexPlan(
                    considered_chunks=_database_int(counts[0][0]),
                    cache_hits=_database_int(counts[0][1]),
                    invalidated_embeddings=(
                        pending_invalidations + max(0, stale_cursor.rowcount)
                    ),
                    missing_chunks=(),
                )

            return await self._run_transaction_locked(connection, prepare)

    async def start_embedding_session(self, model: EmbeddingModelInfo) -> str:
        """Create a running embedding session for one compatible model."""
        session_id = uuid.uuid4().hex
        async with self._lock:
            connection = await self._ensure_connection_locked()

            async def start(active: aiosqlite.Connection) -> None:
                await self._ensure_embedding_model(active, model)
                await active.execute(
                    """
                    INSERT INTO embedding_sessions (
                        session_id,
                        model_fingerprint,
                        started_at
                    ) VALUES (?, ?, ?)
                    """,
                    (session_id, model.fingerprint, _utc_now()),
                )

            await self._run_transaction_locked(connection, start)
        return session_id

    async def list_chunks_missing_embedding(
        self,
        model_fingerprint: str,
        *,
        after_storage_id: int = 0,
        limit: int = 32,
    ) -> tuple[SemanticChunkRecord, ...]:
        """Return one bounded page of globally deduplicated missing chunks."""
        if not model_fingerprint:
            raise ValueError("model_fingerprint must not be empty")
        if after_storage_id < 0:
            raise ValueError("after_storage_id must not be negative")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        async with self._lock:
            connection = await self._ensure_connection_locked()
            rows = list(
                await connection.execute_fetchall(
                    """
                    SELECT
                        chunks.id,
                        chunks.content_hash,
                        chunks.document_title,
                        chunks.heading_path,
                        chunks.text,
                        chunks.size_chars,
                        chunks.estimated_tokens,
                        chunk_provenance.ordinal,
                        documents.document_id,
                        documents.url,
                        documents.canonical_url,
                        document_contents.title,
                        documents.status_code,
                        documents.content_type,
                        documents.fetched_at,
                        documents.source_size_bytes,
                        documents.source_estimated_tokens
                    FROM chunks
                    JOIN chunk_provenance
                        ON chunk_provenance.chunk_id = chunks.id
                    JOIN documents
                        ON documents.document_id = chunk_provenance.document_id
                    JOIN document_contents
                        ON document_contents.content_hash = documents.content_hash
                    LEFT JOIN chunk_embeddings
                        ON chunk_embeddings.chunk_id = chunks.id
                       AND chunk_embeddings.model_fingerprint = ?
                    WHERE chunks.id > ?
                      AND chunk_embeddings.chunk_id IS NULL
                    ORDER BY chunks.id ASC
                    LIMIT ?
                    """,
                    (model_fingerprint, after_storage_id, limit),
                )
            )
        return tuple(_semantic_chunk_record_from_row(row) for row in rows)

    async def store_chunk_embeddings(
        self,
        model: EmbeddingModelInfo,
        embeddings: Sequence[tuple[SemanticChunkRecord, EmbeddingVector]],
    ) -> int:
        """Atomically store one bounded vector batch as float32 blobs."""
        prepared = tuple(
            (
                record,
                serialize_embedding_vector(vector),
            )
            for record, vector in embeddings
        )
        for (record, vector), (_prepared_record, blob) in zip(
            embeddings,
            prepared,
            strict=True,
        ):
            if record.storage_id <= 0:
                raise ValueError("semantic chunk storage_id must be positive")
            if vector.dimension != model.dimension:
                raise ValueError(
                    "embedding vector dimension mismatch: "
                    f"expected {model.dimension}, received {vector.dimension}"
                )
            if len(blob) != model.dimension * 4:
                raise ValueError("serialized embedding size is invalid")
        if not prepared:
            return 0

        async with self._lock:
            connection = await self._ensure_connection_locked()
            store_task = asyncio.create_task(
                self._store_embeddings_transaction(
                    connection,
                    model,
                    prepared,
                )
            )
            try:
                return await asyncio.shield(store_task)
            except asyncio.CancelledError as cancelled:
                try:
                    await store_task
                except Exception as store_error:
                    raise cancelled from store_error
                raise

    async def load_semantic_snapshot(
        self,
        model: EmbeddingModelInfo,
    ) -> SemanticEmbeddingSnapshot:
        """Load one coherent ready-index snapshot for exact semantic search."""
        snapshot_started = time.perf_counter()
        async with self._lock:
            connection = await self._ensure_connection_locked()
            rows = list(
                await connection.execute_fetchall(
                    """
                    WITH summary AS (
                        SELECT
                            (SELECT COUNT(*) FROM chunks) AS total_chunks,
                            (
                                SELECT COUNT(*)
                                FROM chunk_embeddings
                                WHERE model_fingerprint = ?
                            ) AS embedded_chunks,
                            (
                                SELECT COALESCE(SUM(LENGTH(vector)), 0)
                                FROM chunk_embeddings
                                WHERE model_fingerprint = ?
                            ) AS stored_vector_bytes,
                            (
                                SELECT COUNT(*)
                                FROM embedding_models
                                WHERE model_fingerprint = ?
                            ) AS compatible_model_count,
                            (
                                SELECT COUNT(*)
                                FROM embedding_models
                                WHERE model_fingerprint != ?
                            ) AS other_model_count
                    ),
                    compatible AS (
                        SELECT
                            chunks.id,
                            chunks.content_hash,
                            chunks.document_title,
                            chunks.heading_path,
                            chunks.text,
                            chunks.size_chars,
                            chunks.estimated_tokens,
                            chunk_provenance.ordinal,
                            documents.document_id,
                            documents.url,
                            documents.canonical_url,
                            document_contents.title,
                            documents.status_code,
                            documents.content_type,
                            documents.fetched_at,
                            documents.source_size_bytes,
                            documents.source_estimated_tokens,
                            chunk_embeddings.vector,
                            chunk_embeddings.dimension,
                            chunk_embeddings.dtype
                        FROM chunk_embeddings
                        JOIN chunks ON chunks.id = chunk_embeddings.chunk_id
                        JOIN chunk_provenance
                            ON chunk_provenance.chunk_id = chunks.id
                        JOIN documents
                            ON documents.document_id = chunk_provenance.document_id
                        JOIN document_contents
                            ON document_contents.content_hash = documents.content_hash
                        WHERE chunk_embeddings.model_fingerprint = ?
                    )
                    SELECT
                        summary.total_chunks,
                        summary.embedded_chunks,
                        summary.stored_vector_bytes,
                        summary.compatible_model_count,
                        summary.other_model_count,
                        compatible.*
                    FROM summary
                    LEFT JOIN compatible ON TRUE
                    ORDER BY compatible.id ASC
                    """,
                    (
                        model.fingerprint,
                        model.fingerprint,
                        model.fingerprint,
                        model.fingerprint,
                        model.fingerprint,
                    ),
                )
            )
        snapshot_fetch_time_ms = (time.perf_counter() - snapshot_started) * 1000

        if len(rows) < 1:
            raise RuntimeError("semantic index snapshot is unavailable")
        total_chunks = _database_int(rows[0][0])
        embedded_chunks = _database_int(rows[0][1])
        stored_vector_bytes = _database_int(rows[0][2])
        registered = _database_int(rows[0][3]) == 1
        other_model_count = _database_int(rows[0][4])
        if total_chunks == 0:
            return SemanticEmbeddingSnapshot(
                embeddings=(),
                sqlite_snapshot_fetch_time_ms=snapshot_fetch_time_ms,
                vector_decode_time_ms=0.0,
                provenance_materialization_time_ms=0.0,
                stored_vector_bytes=0,
            )
        if not registered:
            if other_model_count:
                raise SemanticIndexIncompatibleError(
                    "Semantic embeddings exist, but not for the requested model "
                    "fingerprint. Build a separate compatible embedding index first."
                )
            raise SemanticIndexNotReadyError(
                "Semantic index is not ready. Run 'crawlforge embed' first."
            )
        if embedded_chunks != total_chunks:
            raise SemanticIndexNotReadyError(
                "Semantic index is incomplete for this model. "
                "Run 'crawlforge embed' first."
            )

        snapshot_rows = [row for row in rows if row[5] is not None]
        if len(snapshot_rows) != total_chunks:
            raise SemanticIndexNotReadyError(
                "Semantic index changed during search. Run 'crawlforge embed' first."
            )

        vector_decode_started = time.perf_counter()
        vectors: list[EmbeddingVector] = []
        for row in snapshot_rows:
            dimension = _database_int(row[23])
            dtype = _database_text(row[24])
            if dimension != model.dimension or dtype != model.dtype:
                raise RuntimeError("stored embedding metadata is incompatible")
            vectors.append(
                deserialize_embedding_vector(
                    _database_blob(row[22]),
                    dimension=dimension,
                )
            )
        vector_decode_time_ms = (time.perf_counter() - vector_decode_started) * 1000

        provenance_started = time.perf_counter()
        embeddings = tuple(
            StoredChunkEmbedding(
                record=_semantic_chunk_record_from_row(row[5:]),
                vector=vector,
            )
            for row, vector in zip(snapshot_rows, vectors, strict=True)
        )
        return SemanticEmbeddingSnapshot(
            embeddings=embeddings,
            sqlite_snapshot_fetch_time_ms=snapshot_fetch_time_ms,
            vector_decode_time_ms=vector_decode_time_ms,
            provenance_materialization_time_ms=(
                time.perf_counter() - provenance_started
            )
            * 1000,
            stored_vector_bytes=stored_vector_bytes,
        )

    async def finish_embedding_session(
        self,
        result: SemanticIndexingResult,
    ) -> None:
        """Persist final embedding counters and timing for one session."""
        async with self._lock:
            connection = await self._ensure_connection_locked()

            async def finish(active: aiosqlite.Connection) -> None:
                cursor = await active.execute(
                    """
                    UPDATE embedding_sessions
                    SET
                        finished_at = ?,
                        considered_chunks = ?,
                        embedded_chunks = ?,
                        cache_hits = ?,
                        invalidated_embeddings = ?,
                        failed_chunks = ?,
                        elapsed_time_ms = ?,
                        vector_bytes = ?,
                        warnings = ?
                    WHERE session_id = ?
                      AND model_fingerprint = ?
                    """,
                    (
                        _utc_now(),
                        result.considered_chunks,
                        result.embedded_chunks,
                        result.cache_hits,
                        result.invalidated_embeddings,
                        result.failed_chunks,
                        result.elapsed_time_ms,
                        result.stored_vector_bytes,
                        _json_text(result.warnings),
                        result.session_id,
                        result.model.fingerprint,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError(f"unknown embedding session: {result.session_id}")

            await self._run_transaction_locked(connection, finish)

    async def get_semantic_index_info(
        self,
        model: EmbeddingModelInfo,
    ) -> SemanticIndexInfo:
        """Return readiness and storage for one exact model fingerprint."""
        async with self._lock:
            connection = await self._ensure_connection_locked()
            rows = list(
                await connection.execute_fetchall(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM chunks),
                        (
                            SELECT COUNT(*)
                            FROM chunk_embeddings
                            WHERE model_fingerprint = ?
                        ),
                        (
                            SELECT COALESCE(SUM(LENGTH(vector)), 0)
                            FROM chunk_embeddings
                            WHERE model_fingerprint = ?
                        ),
                        (
                            SELECT MAX(finished_at)
                            FROM embedding_sessions
                            WHERE model_fingerprint = ?
                        ),
                        (
                            SELECT COUNT(*)
                            FROM embedding_models
                            WHERE model_fingerprint = ?
                        ),
                        (
                            SELECT COUNT(*)
                            FROM embedding_models
                            WHERE model_fingerprint != ?
                        )
                    """,
                    (
                        model.fingerprint,
                        model.fingerprint,
                        model.fingerprint,
                        model.fingerprint,
                        model.fingerprint,
                    ),
                )
            )
        if len(rows) != 1:
            raise RuntimeError("semantic index summary is unavailable")
        total = _database_int(rows[0][0])
        embedded = _database_int(rows[0][1])
        registered = _database_int(rows[0][4]) == 1
        return SemanticIndexInfo(
            model=model,
            total_chunks=total,
            embedded_chunks=embedded,
            missing_chunks=max(0, total - embedded),
            stored_vector_bytes=_database_int(rows[0][2]),
            last_indexed_at=(
                _database_text(rows[0][3]) if rows[0][3] is not None else None
            ),
            ready=registered and (total == 0 or embedded == total),
            compatible_model_registered=registered,
            other_model_count=_database_int(rows[0][5]),
        )

    async def get_index_info(self) -> IndexInfo:
        """Return a bounded readiness, size, and latest-session summary."""
        async with self._lock:
            connection = await self._ensure_connection_locked()
            rows = list(
                await connection.execute_fetchall(
                    """
                    SELECT
                        (SELECT version
                         FROM context_schema_version
                         WHERE singleton = 1),
                        (SELECT COUNT(*) FROM documents),
                        (SELECT COUNT(*) FROM chunks),
                        (SELECT MAX(finished_at) FROM index_sessions)
                    """
                )
            )
            if len(rows) != 1:
                raise RuntimeError("context index summary is unavailable")
            session_rows = list(
                await connection.execute_fetchall(
                    """
                    SELECT
                        session_id,
                        started_at,
                        finished_at,
                        documents_seen,
                        documents_indexed,
                        duplicate_documents,
                        chunks_indexed,
                        duplicate_chunks,
                        source_size_bytes,
                        cleaned_size_bytes,
                        source_estimated_tokens,
                        cleaned_estimated_tokens,
                        cleaning_time_ms,
                        indexing_time_ms
                    FROM index_sessions
                    ORDER BY started_at DESC, session_id DESC
                    LIMIT 1
                    """
                )
            )

        row = rows[0]
        last_indexed_at = (
            datetime.fromisoformat(_database_text(row[3]))
            if row[3] is not None
            else None
        )
        last_session = (
            _session_summary_from_row(session_rows[0]) if session_rows else None
        )
        return IndexInfo(
            schema_version=_database_int(row[0]),
            document_count=_database_int(row[1]),
            chunk_count=_database_int(row[2]),
            last_indexed_at=last_indexed_at,
            last_session_summary=last_session,
            database_ready=True,
            fts5_available=True,
        )

    async def close(self) -> None:
        """Close the database connection; repeated calls are safe."""
        close_task = asyncio.create_task(self._close_impl())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await close_task
            except Exception as close_error:
                raise cancelled from close_error
            raise

    async def _ensure_connection_locked(self) -> aiosqlite.Connection:
        if self._closed:
            raise RuntimeError("SQLiteContextIndex is closed")
        if self._connection is not None:
            return self._connection

        connection = await aiosqlite.connect(self.path)
        connection.row_factory = aiosqlite.Row
        initialize_task = asyncio.create_task(self._initialize_connection(connection))
        try:
            await asyncio.shield(initialize_task)
        except asyncio.CancelledError as cancelled:
            initialization_error: Exception | None = None
            try:
                await initialize_task
            except Exception as caught:
                initialization_error = caught
            try:
                await self._close_connection(connection)
            except Exception as close_error:
                raise cancelled from close_error
            if initialization_error is not None:
                raise cancelled from initialization_error
            raise
        except Exception:
            await self._close_connection(connection)
            raise

        self._connection = connection
        return connection

    async def _initialize_connection(
        self,
        connection: aiosqlite.Connection,
    ) -> None:
        await connection.execute("PRAGMA foreign_keys = ON")
        existing_version = list(
            await connection.execute_fetchall(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name = 'context_schema_version'
                """
            )
        )
        detected_version: int | None = None
        if existing_version:
            version_rows = list(
                await connection.execute_fetchall(
                    "SELECT version FROM context_schema_version WHERE singleton = 1"
                )
            )
            if version_rows:
                detected_version = _database_int(version_rows[0][0])
            if detected_version == 1:
                await self._migrate_v1_to_v2(connection)
                detected_version = 2
            if detected_version == 2:
                await self._migrate_v2_to_v3(connection)
                detected_version = _SCHEMA_VERSION
            elif detected_version is not None and detected_version != _SCHEMA_VERSION:
                raise RuntimeError(
                    "unsupported context index schema version "
                    f"{detected_version}; expected {_SCHEMA_VERSION}"
                )

        try:
            await connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS context_schema_version (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_contents (
                    content_hash TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    markdown TEXT,
                    cleaned_size_bytes INTEGER NOT NULL
                        CHECK (cleaned_size_bytes >= 0),
                    cleaned_estimated_tokens INTEGER NOT NULL
                        CHECK (cleaned_estimated_tokens >= 0)
                );

                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    canonical_url TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL
                        REFERENCES document_contents(content_hash),
                    status_code INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    source_size_bytes INTEGER NOT NULL CHECK (source_size_bytes >= 0),
                    source_estimated_tokens INTEGER NOT NULL
                        CHECK (source_estimated_tokens >= 0),
                    metadata TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_content_hash
                ON documents(content_hash, canonical_url, document_id);

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL UNIQUE,
                    document_title TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    text TEXT NOT NULL,
                    size_chars INTEGER NOT NULL CHECK (size_chars >= 0),
                    estimated_tokens INTEGER NOT NULL CHECK (estimated_tokens >= 0)
                );

                CREATE TABLE IF NOT EXISTS content_chunks (
                    content_hash TEXT NOT NULL
                        REFERENCES document_contents(content_hash) ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id),
                    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                    PRIMARY KEY (content_hash, ordinal)
                );

                CREATE INDEX IF NOT EXISTS idx_content_chunks_chunk_id
                ON content_chunks(chunk_id, content_hash, ordinal);

                CREATE TABLE IF NOT EXISTS chunk_provenance (
                    chunk_id INTEGER PRIMARY KEY
                        REFERENCES chunks(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL
                        REFERENCES documents(document_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL CHECK (ordinal >= 0)
                );

                CREATE TABLE IF NOT EXISTS index_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    documents_seen INTEGER NOT NULL DEFAULT 0,
                    documents_indexed INTEGER NOT NULL DEFAULT 0,
                    duplicate_documents INTEGER NOT NULL DEFAULT 0,
                    chunks_indexed INTEGER NOT NULL DEFAULT 0,
                    duplicate_chunks INTEGER NOT NULL DEFAULT 0,
                    source_size_bytes INTEGER NOT NULL DEFAULT 0,
                    cleaned_size_bytes INTEGER NOT NULL DEFAULT 0,
                    source_estimated_tokens INTEGER NOT NULL DEFAULT 0,
                    cleaned_estimated_tokens INTEGER NOT NULL DEFAULT 0,
                    cleaning_time_ms REAL NOT NULL DEFAULT 0.0,
                    indexing_time_ms REAL NOT NULL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS embedding_models (
                    model_fingerprint TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_revision TEXT,
                    dimension INTEGER NOT NULL CHECK (dimension > 0),
                    dtype TEXT NOT NULL,
                    normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
                    precision TEXT NOT NULL,
                    document_format_version TEXT NOT NULL,
                    query_format_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    pending_invalidations INTEGER NOT NULL DEFAULT 0
                        CHECK (pending_invalidations >= 0)
                );

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    model_fingerprint TEXT NOT NULL
                        REFERENCES embedding_models(model_fingerprint)
                        ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL
                        REFERENCES chunks(id) ON DELETE CASCADE,
                    chunk_content_hash TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL CHECK (dimension > 0),
                    dtype TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (model_fingerprint, chunk_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model_chunk
                ON chunk_embeddings(model_fingerprint, chunk_id);

                CREATE TABLE IF NOT EXISTS embedding_sessions (
                    session_id TEXT PRIMARY KEY,
                    model_fingerprint TEXT NOT NULL
                        REFERENCES embedding_models(model_fingerprint),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    considered_chunks INTEGER NOT NULL DEFAULT 0,
                    embedded_chunks INTEGER NOT NULL DEFAULT 0,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    invalidated_embeddings INTEGER NOT NULL DEFAULT 0,
                    failed_chunks INTEGER NOT NULL DEFAULT 0,
                    elapsed_time_ms REAL NOT NULL DEFAULT 0.0,
                    vector_bytes INTEGER NOT NULL DEFAULT 0,
                    warnings TEXT NOT NULL DEFAULT '[]'
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_text,
                    document_title,
                    heading_path
                );

                INSERT OR IGNORE INTO context_schema_version (singleton, version)
                VALUES (1, 3);
                """
            )
        except aiosqlite.OperationalError as error:
            if _is_fts5_unavailable(error):
                raise FTS5UnavailableError(
                    "SQLite FTS5 support is required for SQLiteContextIndex"
                ) from error
            raise

        version_rows = list(
            await connection.execute_fetchall(
                "SELECT version FROM context_schema_version WHERE singleton = 1"
            )
        )
        if (
            len(version_rows) != 1
            or _database_int(version_rows[0][0]) != _SCHEMA_VERSION
        ):
            raise RuntimeError("context index schema version is missing or invalid")
        await connection.commit()

    async def _migrate_v1_to_v2(
        self,
        connection: aiosqlite.Connection,
    ) -> None:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_provenance (
                    chunk_id INTEGER PRIMARY KEY
                        REFERENCES chunks(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL
                        REFERENCES documents(document_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL CHECK (ordinal >= 0)
                )
                """
            )
            await connection.execute("DROP INDEX IF EXISTS idx_documents_content_hash")
            await connection.execute(
                """
                CREATE INDEX idx_documents_content_hash
                ON documents(content_hash, canonical_url, document_id)
                """
            )
            await connection.execute("DROP INDEX IF EXISTS idx_content_chunks_chunk_id")
            await connection.execute(
                """
                CREATE INDEX idx_content_chunks_chunk_id
                ON content_chunks(chunk_id, content_hash, ordinal)
                """
            )
            await connection.execute("DELETE FROM chunk_provenance")
            await connection.execute(
                """
                INSERT INTO chunk_provenance (chunk_id, document_id, ordinal)
                SELECT chunk_id, document_id, ordinal
                FROM (
                    SELECT
                        content_chunks.chunk_id,
                        documents.document_id,
                        content_chunks.ordinal,
                        ROW_NUMBER() OVER (
                            PARTITION BY content_chunks.chunk_id
                            ORDER BY
                                documents.canonical_url ASC,
                                documents.document_id ASC,
                                content_chunks.ordinal ASC
                        ) AS provenance_rank
                    FROM content_chunks
                    JOIN documents
                        ON documents.content_hash =
                            content_chunks.content_hash
                )
                WHERE provenance_rank = 1
                """
            )
            await connection.execute(
                """
                UPDATE context_schema_version
                SET version = ?
                WHERE singleton = 1
                """,
                (2,),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

    async def _migrate_v2_to_v3(
        self,
        connection: aiosqlite.Connection,
    ) -> None:
        try:
            await connection.executescript(
                """
                BEGIN IMMEDIATE;

                CREATE TABLE IF NOT EXISTS embedding_models (
                    model_fingerprint TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_revision TEXT,
                    dimension INTEGER NOT NULL CHECK (dimension > 0),
                    dtype TEXT NOT NULL,
                    normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
                    precision TEXT NOT NULL,
                    document_format_version TEXT NOT NULL,
                    query_format_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    pending_invalidations INTEGER NOT NULL DEFAULT 0
                        CHECK (pending_invalidations >= 0)
                );

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    model_fingerprint TEXT NOT NULL
                        REFERENCES embedding_models(model_fingerprint)
                        ON DELETE CASCADE,
                    chunk_id INTEGER NOT NULL
                        REFERENCES chunks(id) ON DELETE CASCADE,
                    chunk_content_hash TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL CHECK (dimension > 0),
                    dtype TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (model_fingerprint, chunk_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model_chunk
                ON chunk_embeddings(model_fingerprint, chunk_id);

                CREATE TABLE IF NOT EXISTS embedding_sessions (
                    session_id TEXT PRIMARY KEY,
                    model_fingerprint TEXT NOT NULL
                        REFERENCES embedding_models(model_fingerprint),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    considered_chunks INTEGER NOT NULL DEFAULT 0,
                    embedded_chunks INTEGER NOT NULL DEFAULT 0,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    invalidated_embeddings INTEGER NOT NULL DEFAULT 0,
                    failed_chunks INTEGER NOT NULL DEFAULT 0,
                    elapsed_time_ms REAL NOT NULL DEFAULT 0.0,
                    vector_bytes INTEGER NOT NULL DEFAULT 0,
                    warnings TEXT NOT NULL DEFAULT '[]'
                );

                UPDATE context_schema_version
                SET version = 3
                WHERE singleton = 1;

                COMMIT;
                """
            )
        except BaseException:
            await connection.rollback()
            raise

    async def _index_batch_transaction(
        self,
        connection: aiosqlite.Connection,
        documents: tuple[tuple[SourceDocument, tuple[TextChunk, ...]], ...],
        *,
        session_id: str,
    ) -> IndexingResult:
        started = time.perf_counter()

        async def index_batch(active: aiosqlite.Connection) -> IndexingResult:
            session_rows = list(
                await active.execute_fetchall(
                    """
                    SELECT finished_at
                    FROM index_sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
            )
            if not session_rows:
                raise ValueError(f"unknown indexing session: {session_id}")
            if session_rows[0][0] is not None:
                raise ValueError(f"indexing session is already finished: {session_id}")

            documents_indexed = 0
            duplicate_documents = 0
            chunks_indexed = 0
            duplicate_chunks = 0
            affected_chunk_ids: set[int] = set()

            for document, chunks in documents:
                previous_chunk_rows = list(
                    await active.execute_fetchall(
                        """
                        SELECT content_chunks.chunk_id
                        FROM documents
                        JOIN content_chunks
                            ON content_chunks.content_hash =
                                documents.content_hash
                        WHERE documents.document_id = ?
                        """,
                        (document.id,),
                    )
                )
                affected_chunk_ids.update(
                    _database_int(row[0]) for row in previous_chunk_rows
                )
                content_is_new = await self._ensure_content(active, document)
                if content_is_new:
                    documents_indexed += 1
                else:
                    duplicate_documents += 1

                inserted, duplicates, affected = await self._replace_content_chunks(
                    active,
                    document,
                    chunks,
                )
                chunks_indexed += inserted
                duplicate_chunks += duplicates
                affected_chunk_ids.update(affected)
                await self._upsert_document(active, document)

            await self._delete_orphans(active)
            await self._refresh_provenance(active, affected_chunk_ids)
            indexing_time_ms = (time.perf_counter() - started) * 1000.0
            result = IndexingResult(
                session_id=session_id,
                documents_seen=len(documents),
                documents_indexed=documents_indexed,
                duplicate_documents=duplicate_documents,
                chunks_indexed=chunks_indexed,
                duplicate_chunks=duplicate_chunks,
                source_size_bytes=sum(
                    document.source_size_bytes for document, _chunks in documents
                ),
                cleaned_size_bytes=sum(
                    document.cleaned_size_bytes for document, _chunks in documents
                ),
                source_estimated_tokens=sum(
                    document.source_estimated_tokens for document, _chunks in documents
                ),
                cleaned_estimated_tokens=sum(
                    document.cleaned_estimated_tokens for document, _chunks in documents
                ),
                cleaning_time_ms=0.0,
                indexing_time_ms=indexing_time_ms,
            )
            await active.execute(
                """
                UPDATE index_sessions
                SET
                    documents_seen = documents_seen + ?,
                    documents_indexed = documents_indexed + ?,
                    duplicate_documents = duplicate_documents + ?,
                    chunks_indexed = chunks_indexed + ?,
                    duplicate_chunks = duplicate_chunks + ?,
                    source_size_bytes = source_size_bytes + ?,
                    cleaned_size_bytes = cleaned_size_bytes + ?,
                    source_estimated_tokens = source_estimated_tokens + ?,
                    cleaned_estimated_tokens = cleaned_estimated_tokens + ?,
                    cleaning_time_ms = cleaning_time_ms + ?,
                    indexing_time_ms = indexing_time_ms + ?
                WHERE session_id = ? AND finished_at IS NULL
                """,
                (
                    result.documents_seen,
                    result.documents_indexed,
                    result.duplicate_documents,
                    result.chunks_indexed,
                    result.duplicate_chunks,
                    result.source_size_bytes,
                    result.cleaned_size_bytes,
                    result.source_estimated_tokens,
                    result.cleaned_estimated_tokens,
                    result.cleaning_time_ms,
                    result.indexing_time_ms,
                    session_id,
                ),
            )
            return result

        return await self._run_transaction_locked(connection, index_batch)

    async def _ensure_content(
        self,
        connection: aiosqlite.Connection,
        document: SourceDocument,
    ) -> bool:
        existing = list(
            await connection.execute_fetchall(
                """
            SELECT
                title,
                text,
                markdown
            FROM document_contents
                WHERE content_hash = ?
                """,
                (document.content_hash,),
            )
        )
        semantic_values = (
            document.title,
            document.text,
            document.markdown,
        )
        if existing:
            if tuple(existing[0]) != semantic_values:
                raise ValueError(
                    f"document content hash collision: {document.content_hash}"
                )
            await connection.execute(
                """
                UPDATE document_contents
                SET
                    cleaned_size_bytes = ?,
                    cleaned_estimated_tokens = ?
                WHERE content_hash = ?
                """,
                (
                    document.cleaned_size_bytes,
                    document.cleaned_estimated_tokens,
                    document.content_hash,
                ),
            )
            return False

        await connection.execute(
            """
            INSERT INTO document_contents (
                content_hash,
                title,
                text,
                markdown,
                cleaned_size_bytes,
                cleaned_estimated_tokens
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document.content_hash,
                *semantic_values,
                document.cleaned_size_bytes,
                document.cleaned_estimated_tokens,
            ),
        )
        return True

    async def _replace_content_chunks(
        self,
        connection: aiosqlite.Connection,
        document: SourceDocument,
        chunks: tuple[TextChunk, ...],
    ) -> tuple[int, int, set[int]]:
        inserted = 0
        duplicates = 0
        relations: list[tuple[str, int, int]] = []
        existing_relations = list(
            await connection.execute_fetchall(
                """
                SELECT chunk_id
                FROM content_chunks
                WHERE content_hash = ?
                """,
                (document.content_hash,),
            )
        )
        affected_chunk_ids = {_database_int(row[0]) for row in existing_relations}

        for chunk in chunks:
            existing = list(
                await connection.execute_fetchall(
                    """
                    SELECT
                        id,
                        document_title,
                        heading_path,
                        text
                    FROM chunks
                    WHERE content_hash = ?
                    """,
                    (chunk.content_hash,),
                )
            )
            heading_path = _json_text(chunk.heading_path)
            chunk_values = (
                chunk.document_title,
                heading_path,
                chunk.text,
            )
            if existing:
                chunk_id = _database_int(existing[0][0])
                if tuple(existing[0][1:]) != chunk_values:
                    raise ValueError(
                        f"chunk content hash collision: {chunk.content_hash}"
                    )
                await connection.execute(
                    """
                    UPDATE chunks
                    SET size_chars = ?, estimated_tokens = ?
                    WHERE id = ?
                    """,
                    (chunk.size_chars, chunk.estimated_tokens, chunk_id),
                )
                duplicates += 1
            else:
                cursor = await connection.execute(
                    """
                    INSERT INTO chunks (
                        content_hash,
                        document_title,
                        heading_path,
                        text,
                        size_chars,
                        estimated_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.content_hash,
                        *chunk_values,
                        chunk.size_chars,
                        chunk.estimated_tokens,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return an inserted chunk id")
                chunk_id = cursor.lastrowid
                await connection.execute(
                    """
                    INSERT INTO chunk_fts (
                        rowid,
                        chunk_text,
                        document_title,
                        heading_path
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        chunk.text,
                        chunk.document_title,
                        " / ".join(chunk.heading_path),
                    ),
                )
                inserted += 1
            relations.append((document.content_hash, chunk_id, chunk.ordinal))
            affected_chunk_ids.add(chunk_id)

        await connection.execute(
            "DELETE FROM content_chunks WHERE content_hash = ?",
            (document.content_hash,),
        )
        if relations:
            await connection.executemany(
                """
                INSERT INTO content_chunks (content_hash, chunk_id, ordinal)
                VALUES (?, ?, ?)
                """,
                relations,
            )
        return inserted, duplicates, affected_chunk_ids

    async def _upsert_document(
        self,
        connection: aiosqlite.Connection,
        document: SourceDocument,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO documents (
                document_id,
                url,
                canonical_url,
                content_hash,
                status_code,
                content_type,
                fetched_at,
                source_size_bytes,
                source_estimated_tokens,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                url = excluded.url,
                canonical_url = excluded.canonical_url,
                content_hash = excluded.content_hash,
                status_code = excluded.status_code,
                content_type = excluded.content_type,
                fetched_at = excluded.fetched_at,
                source_size_bytes = excluded.source_size_bytes,
                source_estimated_tokens = excluded.source_estimated_tokens,
                metadata = excluded.metadata
            """,
            (
                document.id,
                document.url,
                document.canonical_url,
                document.content_hash,
                document.status_code,
                document.content_type,
                document.fetched_at.isoformat(),
                document.source_size_bytes,
                document.source_estimated_tokens,
                _json_object(document.metadata),
            ),
        )

    async def _delete_orphans(self, connection: aiosqlite.Connection) -> None:
        await connection.execute(
            """
            DELETE FROM document_contents
            WHERE NOT EXISTS (
                SELECT 1
                FROM documents
                WHERE documents.content_hash = document_contents.content_hash
            )
            """
        )
        await connection.execute(
            """
            UPDATE embedding_models
            SET pending_invalidations = pending_invalidations + (
                SELECT COUNT(*)
                FROM chunk_embeddings
                JOIN chunks ON chunks.id = chunk_embeddings.chunk_id
                WHERE chunk_embeddings.model_fingerprint =
                        embedding_models.model_fingerprint
                  AND NOT EXISTS (
                      SELECT 1
                      FROM content_chunks
                      WHERE content_chunks.chunk_id = chunks.id
                  )
            )
            """
        )
        await connection.execute(
            """
            DELETE FROM chunk_fts
            WHERE rowid IN (
                SELECT chunk_fts.rowid
                FROM chunk_fts
                LEFT JOIN chunks ON chunks.id = chunk_fts.rowid
                LEFT JOIN content_chunks ON content_chunks.chunk_id = chunks.id
                WHERE content_chunks.chunk_id IS NULL
            )
            """
        )
        await connection.execute(
            """
            DELETE FROM chunks
            WHERE NOT EXISTS (
                SELECT 1
                FROM content_chunks
                WHERE content_chunks.chunk_id = chunks.id
            )
            """
        )

    async def _refresh_provenance(
        self,
        connection: aiosqlite.Connection,
        chunk_ids: set[int],
    ) -> None:
        for chunk_id in sorted(chunk_ids):
            await connection.execute(
                "DELETE FROM chunk_provenance WHERE chunk_id = ?",
                (chunk_id,),
            )
            await connection.execute(
                """
                INSERT INTO chunk_provenance (chunk_id, document_id, ordinal)
                SELECT
                    content_chunks.chunk_id,
                    documents.document_id,
                    content_chunks.ordinal
                FROM content_chunks
                JOIN documents
                    ON documents.content_hash = content_chunks.content_hash
                WHERE content_chunks.chunk_id = ?
                ORDER BY
                    documents.canonical_url ASC,
                    documents.document_id ASC,
                    content_chunks.ordinal ASC
                LIMIT 1
                """,
                (chunk_id,),
            )

    async def _store_embeddings_transaction(
        self,
        connection: aiosqlite.Connection,
        model: EmbeddingModelInfo,
        embeddings: tuple[tuple[SemanticChunkRecord, bytes], ...],
    ) -> int:
        async def store(active: aiosqlite.Connection) -> int:
            await self._ensure_embedding_model(active, model)
            rows: list[tuple[str, int, str, bytes, int, str, str]] = []
            for record, blob in embeddings:
                chunk_rows = list(
                    await active.execute_fetchall(
                        """
                        SELECT content_hash
                        FROM chunks
                        WHERE id = ?
                        """,
                        (record.storage_id,),
                    )
                )
                if len(chunk_rows) != 1:
                    raise ValueError(
                        f"unknown semantic chunk storage id: {record.storage_id}"
                    )
                stored_hash = _database_text(chunk_rows[0][0])
                if stored_hash != record.chunk.content_hash:
                    raise ValueError(
                        f"stale semantic chunk: {record.chunk.content_hash}"
                    )
                rows.append(
                    (
                        model.fingerprint,
                        record.storage_id,
                        record.chunk.content_hash,
                        blob,
                        model.dimension,
                        model.dtype,
                        _utc_now(),
                    )
                )
            await active.executemany(
                """
                INSERT INTO chunk_embeddings (
                    model_fingerprint,
                    chunk_id,
                    chunk_content_hash,
                    vector,
                    dimension,
                    dtype,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_fingerprint, chunk_id) DO UPDATE SET
                    chunk_content_hash = excluded.chunk_content_hash,
                    vector = excluded.vector,
                    dimension = excluded.dimension,
                    dtype = excluded.dtype,
                    created_at = excluded.created_at
                """,
                rows,
            )
            return sum(len(row[3]) for row in rows)

        return await self._run_transaction_locked(connection, store)

    async def _ensure_embedding_model(
        self,
        connection: aiosqlite.Connection,
        model: EmbeddingModelInfo,
    ) -> None:
        existing = list(
            await connection.execute_fetchall(
                """
                SELECT
                    provider,
                    model_id,
                    model_revision,
                    dimension,
                    dtype,
                    normalized,
                    precision,
                    document_format_version,
                    query_format_version
                FROM embedding_models
                WHERE model_fingerprint = ?
                """,
                (model.fingerprint,),
            )
        )
        stable_values = (
            model.provider,
            model.model_id,
            model.model_revision,
            model.dimension,
            model.dtype,
            int(model.normalized),
            model.precision,
            model.document_format_version,
            model.query_format_version,
        )
        metadata = _json_object(model.metadata)
        if existing:
            if tuple(existing[0]) != stable_values:
                raise ValueError(
                    f"embedding model fingerprint collision: {model.fingerprint}"
                )
            await connection.execute(
                """
                UPDATE embedding_models
                SET metadata = ?
                WHERE model_fingerprint = ?
                """,
                (metadata, model.fingerprint),
            )
            return
        await connection.execute(
            """
            INSERT INTO embedding_models (
                model_fingerprint,
                provider,
                model_id,
                model_revision,
                dimension,
                dtype,
                normalized,
                precision,
                document_format_version,
                query_format_version,
                created_at,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model.fingerprint,
                *stable_values,
                _utc_now(),
                metadata,
            ),
        )

    async def _finish_session_locked(
        self,
        connection: aiosqlite.Connection,
        result: IndexingResult,
    ) -> None:
        async def finish(active: aiosqlite.Connection) -> None:
            cursor = await active.execute(
                """
                UPDATE index_sessions
                SET
                    finished_at = ?,
                    cleaning_time_ms = MAX(cleaning_time_ms, ?)
                WHERE session_id = ?
                """,
                (
                    _utc_now(),
                    result.cleaning_time_ms,
                    result.session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"unknown indexing session: {result.session_id}")

        await self._run_transaction_locked(connection, finish)

    async def _run_transaction_locked(
        self,
        connection: aiosqlite.Connection,
        operation: Callable[[aiosqlite.Connection], Awaitable[_T]],
    ) -> _T:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            result = await operation(connection)
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
        return result

    async def _close_impl(self) -> None:
        async with self._lock:
            if self._closed:
                return
            connection = self._connection
            if connection is not None:
                await self._close_connection(connection)
            self._connection = None
            self._closed = True

    async def _close_connection(self, connection: aiosqlite.Connection) -> None:
        close_task = asyncio.create_task(connection.close())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await close_task
            except Exception as close_error:
                raise cancelled from close_error
            raise

    @staticmethod
    def _validate_batch(
        documents: tuple[tuple[SourceDocument, tuple[TextChunk, ...]], ...],
    ) -> None:
        for document, chunks in documents:
            ordinals: set[int] = set()
            for chunk in chunks:
                if chunk.document_id != document.id:
                    raise ValueError(
                        f"chunk {chunk.id} belongs to a different document"
                    )
                if chunk.ordinal in ordinals:
                    raise ValueError(
                        f"document {document.id} has duplicate chunk ordinal "
                        f"{chunk.ordinal}"
                    )
                ordinals.add(chunk.ordinal)


def _literal_match_query(query: str) -> str | None:
    tokens = re.findall(r"\w+(?:(?:[.+#-]+\w+)|(?:\+\+)|#)*", query)
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        folded = token.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        unique_tokens.append(token)
    if not unique_tokens:
        return None
    return " OR ".join(f'"{token.replace('"', '""')}"' for token in unique_tokens)


def _document_chunk_id(
    document_id: str,
    ordinal: int,
    content_hash: str,
) -> str:
    value = f"{document_id}\0{ordinal}\0{content_hash}"
    return hashlib.sha256(value.encode()).hexdigest()


def _json_text(values: Sequence[str]) -> str:
    return json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _json_object(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_heading_path(value: str) -> tuple[str, ...]:
    decoded: object = json.loads(value)
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise RuntimeError("stored chunk heading path is invalid")
    return tuple(decoded)


def _semantic_chunk_record_from_row(row: Sequence[object]) -> SemanticChunkRecord:
    storage_id = _database_int(row[0])
    content_hash = _database_text(row[1])
    ordinal = _database_int(row[7])
    document_id = _database_text(row[8])
    source = SourceReference(
        document_id=document_id,
        url=_database_text(row[9]),
        canonical_url=_database_text(row[10]),
        title=_database_text(row[11]),
        status_code=_database_int(row[12]),
        content_type=_database_text(row[13]),
        fetched_at=datetime.fromisoformat(_database_text(row[14])),
        source_size_bytes=_database_int(row[15]),
        source_estimated_tokens=_database_int(row[16]),
    )
    return SemanticChunkRecord(
        storage_id=storage_id,
        chunk=TextChunk(
            id=_document_chunk_id(document_id, ordinal, content_hash),
            document_id=document_id,
            ordinal=ordinal,
            source_url=source.url,
            document_title=_database_text(row[2]),
            heading_path=_decode_heading_path(_database_text(row[3])),
            text=_database_text(row[4]),
            size_chars=_database_int(row[5]),
            estimated_tokens=_database_int(row[6]),
            content_hash=content_hash,
        ),
        source=source,
    )


def _database_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("stored SQLite integer is invalid")
    return value


def _database_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise RuntimeError("stored SQLite number is invalid")
    return float(value)


def _database_text(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("stored SQLite text is invalid")
    return value


def _database_blob(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise RuntimeError("stored SQLite blob is invalid")
    return value


def _session_summary_from_row(row: aiosqlite.Row) -> IndexSessionSummary:
    finished_at = (
        datetime.fromisoformat(_database_text(row[2])) if row[2] is not None else None
    )
    return IndexSessionSummary(
        session_id=_database_text(row[0]),
        started_at=datetime.fromisoformat(_database_text(row[1])),
        finished_at=finished_at,
        documents_seen=_database_int(row[3]),
        documents_indexed=_database_int(row[4]),
        duplicate_documents=_database_int(row[5]),
        chunks_indexed=_database_int(row[6]),
        duplicate_chunks=_database_int(row[7]),
        source_size_bytes=_database_int(row[8]),
        cleaned_size_bytes=_database_int(row[9]),
        source_estimated_tokens=_database_int(row[10]),
        cleaned_estimated_tokens=_database_int(row[11]),
        cleaning_time_ms=_database_float(row[12]),
        indexing_time_ms=_database_float(row[13]),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_fts5_unavailable(error: aiosqlite.OperationalError) -> bool:
    message = str(error).casefold()
    return "fts5" in message and (
        "no such module" in message or "not authorized" in message
    )
