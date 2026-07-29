from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest
from semantic_fakes import DeterministicEmbeddingProvider

import crawlforge.semantic as semantic_module
from crawlforge import ContextEngine
from crawlforge.context_models import DocumentBlock, SourceDocument
from crawlforge.semantic_models import (
    EmbeddingModelInfo,
    EmbeddingVector,
    SemanticEmbeddingSnapshot,
    SemanticIndexIncompatibleError,
    SemanticIndexInfo,
    SemanticIndexingResult,
    SemanticIndexNotReadyError,
    StoredChunkEmbedding,
    embedding_model_info,
    format_semantic_document,
)


def _document(name: str, text: str | None = None) -> SourceDocument:
    body = text or f"{name} controls bounded crawler behavior."
    content_hash = hashlib.sha256(f"{name}\0{body}".encode()).hexdigest()
    heading = f"{name} section"
    return SourceDocument(
        id=f"document-{name}",
        url=f"https://example.test/{name}",
        canonical_url=f"https://example.test/{name}",
        title=f"{name} guide",
        text=f"{heading}\n\n{body}",
        markdown=None,
        status_code=200,
        content_type="text/html",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash=content_hash,
        metadata={},
        source_size_bytes=len(body.encode()),
        cleaned_size_bytes=len(body.encode()),
        source_estimated_tokens=max(1, len(body) // 4),
        cleaned_estimated_tokens=max(1, len(body) // 4),
        blocks=(
            DocumentBlock(
                kind="heading",
                text=heading,
                markdown=f"# {heading}",
                heading_path=(heading,),
            ),
            DocumentBlock(
                kind="paragraph",
                text=body,
                markdown=body,
                heading_path=(heading,),
            ),
        ),
    )


def _provider_for_documents(
    engine: ContextEngine,
    documents: tuple[SourceDocument, ...],
    vectors: tuple[tuple[float, ...], ...],
    *,
    query_vectors: dict[str, tuple[float, ...]] | None = None,
    revision: str = "revision-a",
    gate: asyncio.Event | None = None,
    query_gate: asyncio.Event | None = None,
) -> DeterministicEmbeddingProvider:
    chunks = tuple(
        chunk for document in documents for chunk in engine.chunker.chunk(document)
    )
    return DeterministicEmbeddingProvider(
        document_vectors={
            format_semantic_document(chunk): vector
            for chunk, vector in zip(chunks, vectors, strict=True)
        },
        query_vectors=query_vectors or {},
        revision=revision,
        dimension=len(vectors[0]) if vectors else 3,
        document_gate=gate,
        query_gate=query_gate,
    )


@pytest.mark.asyncio
async def test_incremental_index_reuses_cache_and_batches_without_reembedding(
    tmp_path: Path,
) -> None:
    documents = (_document("alpha"), _document("beta"), _document("gamma"))
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages(documents)
    provider = _provider_for_documents(
        engine,
        documents,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )

    first = await engine.index_embeddings(provider, batch_size=2)
    second = await engine.index_embeddings(provider, batch_size=2)
    info = await engine.get_semantic_index_info(provider)
    await engine.close()

    assert first.considered_chunks == 3
    assert first.embedded_chunks == 3
    assert first.cache_hits == 0
    assert first.failed_chunks == 0
    assert first.stored_vector_bytes == 36
    assert [len(call) for call in provider.document_calls] == [2, 1]
    assert second.embedded_chunks == 0
    assert second.cache_hits == 3
    assert second.total_stored_vector_bytes == 36
    assert len(provider.document_calls) == 2
    assert info.ready
    assert info.embedded_chunks == 3
    assert info.stored_vector_bytes == 36


@pytest.mark.asyncio
async def test_identical_chunk_content_reuses_one_embedding_across_sources(
    tmp_path: Path,
) -> None:
    first = _document("shared")
    second = replace(
        first,
        id="document-shared-copy",
        url="https://example.test/shared-copy",
        canonical_url="https://example.test/shared-copy",
    )
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((first, second))
    provider = _provider_for_documents(
        engine,
        (first,),
        ((1.0, 0.0, 0.0),),
    )

    result = await engine.index_embeddings(provider)
    info = await engine.get_semantic_index_info(provider)
    await engine.close()

    assert result.considered_chunks == 1
    assert result.embedded_chunks == 1
    assert len(provider.document_calls) == 1
    assert info.total_chunks == 1
    assert info.embedded_chunks == 1


@pytest.mark.asyncio
async def test_exact_cosine_search_preserves_rank_ties_limit_and_provenance(
    tmp_path: Path,
) -> None:
    documents = (_document("alpha"), _document("beta"), _document("gamma"))
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages(documents)
    provider = _provider_for_documents(
        engine,
        documents,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        query_vectors={"bounded behavior": (1.0, 0.0, 0.0)},
    )
    await engine.index_embeddings(provider)

    result = await engine.semantic_search_with_metrics(
        "bounded behavior",
        provider=provider,
        limit=3,
    )
    limited = await engine.semantic_search(
        "bounded behavior",
        provider=provider,
        limit=1,
    )
    await engine.close()

    assert [hit.rank for hit in result.hits] == [1, 2, 3]
    assert result.hits[0].source.document_id == "document-alpha"
    assert result.hits[0].cosine_similarity == pytest.approx(1.0)
    tied_hashes = [hit.chunk.content_hash for hit in result.hits[1:]]
    assert tied_hashes == sorted(tied_hashes)
    assert all(hit.source.url and hit.source.canonical_url for hit in result.hits)
    assert len(limited) == 1
    assert provider.query_calls == [
        ("bounded behavior",),
        ("bounded behavior",),
    ]
    assert result.loaded_vector_bytes == 36


@pytest.mark.asyncio
async def test_semantic_context_reuses_complete_chunk_budget_selection(
    tmp_path: Path,
) -> None:
    documents = (_document("alpha"), _document("beta"))
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages(documents)
    chunks = tuple(engine.chunker.chunk(document)[0] for document in documents)
    provider = _provider_for_documents(
        engine,
        documents,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        query_vectors={"query": (1.0, 0.0, 0.0)},
    )
    await engine.index_embeddings(provider)

    context = await engine.build_semantic_context(
        "query",
        provider=provider,
        limit=2,
        token_budget=chunks[0].estimated_tokens,
    )
    await engine.close()

    assert len(context.hits) == 1
    assert context.estimated_tokens <= context.token_budget
    assert context.hits[0].chunk.text == chunks[0].text
    assert context.retrieval_strategy == "semantic"
    assert context.score_type == "cosine_similarity"


@pytest.mark.asyncio
async def test_empty_missing_and_incompatible_semantic_indexes_are_explicit(
    tmp_path: Path,
) -> None:
    empty = ContextEngine(tmp_path / "empty.sqlite3")
    empty_provider = DeterministicEmbeddingProvider(
        document_vectors={},
        query_vectors={},
    )
    assert await empty.semantic_search("query", provider=empty_provider) == []
    await empty.close()

    document = _document("alpha")
    engine = ContextEngine(tmp_path / "missing.sqlite3")
    await engine.index_pages((document,))
    provider = _provider_for_documents(
        engine,
        (document,),
        ((1.0, 0.0, 0.0),),
        query_vectors={"query": (1.0, 0.0, 0.0)},
    )
    with pytest.raises(SemanticIndexNotReadyError, match="crawlforge embed"):
        await engine.semantic_search("query", provider=provider)
    await engine.index_embeddings(provider)

    incompatible = _provider_for_documents(
        engine,
        (document,),
        ((1.0, 0.0, 0.0),),
        query_vectors={"query": (1.0, 0.0, 0.0)},
        revision="revision-b",
    )
    with pytest.raises(SemanticIndexIncompatibleError, match="separate compatible"):
        await engine.semantic_search("query", provider=incompatible)
    await engine.close()


@pytest.mark.asyncio
async def test_multiple_model_fingerprints_remain_separate_and_searchable(
    tmp_path: Path,
) -> None:
    document = _document("alpha")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((document,))
    first = _provider_for_documents(
        engine,
        (document,),
        ((1.0, 0.0, 0.0),),
        query_vectors={"query": (1.0, 0.0, 0.0)},
        revision="revision-a",
    )
    second = _provider_for_documents(
        engine,
        (document,),
        ((0.0, 1.0, 0.0),),
        query_vectors={"query": (0.0, 1.0, 0.0)},
        revision="revision-b",
    )

    await engine.index_embeddings(first)
    await engine.index_embeddings(second)
    first_info = await engine.get_semantic_index_info(first)
    second_info = await engine.get_semantic_index_info(second)
    first_hits = await engine.semantic_search("query", provider=first)
    second_hits = await engine.semantic_search("query", provider=second)
    await engine.close()

    assert first_info.ready and second_info.ready
    assert first_info.other_model_count == second_info.other_model_count == 1
    assert first_hits[0].model_fingerprint != second_hits[0].model_fingerprint
    assert first_hits[0].cosine_similarity == pytest.approx(1.0)
    assert second_hits[0].cosine_similarity == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_changed_chunk_invalidates_old_vector_and_resumes_incrementally(
    tmp_path: Path,
) -> None:
    original = _document("alpha", "Original bounded behavior.")
    updated = _document("alpha", "Updated cancellation behavior.")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((original,))
    first_provider = _provider_for_documents(
        engine,
        (original,),
        ((1.0, 0.0, 0.0),),
    )
    await engine.index_embeddings(first_provider)
    await engine.index_pages((updated,))
    second_provider = _provider_for_documents(
        engine,
        (updated,),
        ((0.0, 1.0, 0.0),),
    )

    result = await engine.index_embeddings(second_provider)
    info = await engine.get_semantic_index_info(second_provider)
    await engine.close()

    assert result.invalidated_embeddings == 1
    assert result.embedded_chunks == 1
    assert result.cache_hits == 0
    assert info.embedded_chunks == 1
    assert info.stored_vector_bytes == 12


@pytest.mark.asyncio
async def test_embedding_batch_transaction_rolls_back_on_stale_chunk(
    tmp_path: Path,
) -> None:
    document = _document("alpha")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((document,))
    provider = _provider_for_documents(
        engine,
        (document,),
        ((1.0, 0.0, 0.0),),
    )
    model = embedding_model_info(provider)
    await engine.index.prepare_semantic_index(model)
    records = await engine.index.list_chunks_missing_embedding(model.fingerprint)
    stale = replace(records[0], storage_id=records[0].storage_id + 10_000)

    with pytest.raises(ValueError, match="unknown semantic chunk"):
        await engine.index.store_chunk_embeddings(
            model,
            (
                (records[0], EmbeddingVector((1.0, 0.0, 0.0))),
                (stale, EmbeddingVector((0.0, 1.0, 0.0))),
            ),
        )
    info = await engine.index.get_semantic_index_info(model)
    await engine.close()

    assert info.embedded_chunks == 0


@pytest.mark.asyncio
async def test_cancelled_embedding_session_is_finished_and_retryable(
    tmp_path: Path,
) -> None:
    document = _document("alpha")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((document,))
    gate = asyncio.Event()
    provider = _provider_for_documents(
        engine,
        (document,),
        ((1.0, 0.0, 0.0),),
        gate=gate,
    )
    task = asyncio.create_task(engine.index_embeddings(provider))
    while not provider.document_calls:
        await asyncio.sleep(0)
    task.cancel()
    gate.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    retry = await engine.index_embeddings(provider)
    await engine.close()

    assert retry.embedded_chunks == 1
    async with aiosqlite.connect(tmp_path / "semantic.sqlite3") as connection:
        unfinished = await connection.execute_fetchall(
            "SELECT COUNT(*) FROM embedding_sessions WHERE finished_at IS NULL"
        )
    assert [tuple(row) for row in unfinished] == [(0,)]


@pytest.mark.asyncio
async def test_success_session_finish_survives_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document("alpha")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((document,))
    provider = _provider_for_documents(
        engine,
        (document,),
        ((1.0, 0.0, 0.0),),
    )
    finish_started = asyncio.Event()
    allow_finish = asyncio.Event()
    original_finish = engine.index.finish_embedding_session

    async def delayed_finish(result: SemanticIndexingResult) -> None:
        finish_started.set()
        await allow_finish.wait()
        await original_finish(result)

    monkeypatch.setattr(engine.index, "finish_embedding_session", delayed_finish)
    task = asyncio.create_task(engine.index_embeddings(provider))
    await finish_started.wait()
    task.cancel()
    allow_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    await engine.close()
    async with aiosqlite.connect(tmp_path / "semantic.sqlite3") as connection:
        unfinished = await connection.execute_fetchall(
            "SELECT COUNT(*) FROM embedding_sessions WHERE finished_at IS NULL"
        )
    assert [tuple(row) for row in unfinished] == [(0,)]


@pytest.mark.asyncio
async def test_search_uses_coherent_snapshot_during_chunk_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _document("alpha", "Original bounded behavior.")
    updated = _document("alpha", "Updated cancellation behavior.")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((original,))
    provider = _provider_for_documents(
        engine,
        (original,),
        ((1.0, 0.0, 0.0),),
        query_vectors={"query": (1.0, 0.0, 0.0)},
    )
    await engine.index_embeddings(provider)
    scan_started = threading.Event()
    allow_scan = threading.Event()
    original_rank = semantic_module._rank_vectors

    def delayed_rank(
        query: EmbeddingVector,
        stored: Sequence[StoredChunkEmbedding],
        model: EmbeddingModelInfo,
        limit: int,
    ) -> tuple[semantic_module._RankedVector, ...]:
        scan_started.set()
        if not allow_scan.wait(timeout=5):
            raise TimeoutError("semantic scan gate timed out")
        return original_rank(query, stored, model, limit)

    monkeypatch.setattr(semantic_module, "_rank_vectors", delayed_rank)
    search_task = asyncio.create_task(
        engine.semantic_search("query", provider=provider)
    )
    assert await asyncio.to_thread(scan_started.wait, 5)
    await engine.index_pages((updated,))
    allow_scan.set()

    hits = await search_task
    await engine.close()

    assert len(hits) == 1
    assert hits[0].chunk.text.endswith("Original bounded behavior.")
    assert hits[0].source.document_id == original.id


@pytest.mark.asyncio
async def test_search_rechecks_snapshot_readiness_after_concurrent_addition(
    tmp_path: Path,
) -> None:
    original = _document("alpha")
    added = _document("beta")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((original,))
    query_gate = asyncio.Event()
    provider = _provider_for_documents(
        engine,
        (original,),
        ((1.0, 0.0, 0.0),),
        query_vectors={"query": (1.0, 0.0, 0.0)},
        query_gate=query_gate,
    )
    await engine.index_embeddings(provider)
    search_task = asyncio.create_task(
        engine.semantic_search("query", provider=provider)
    )
    while not provider.query_calls:
        await asyncio.sleep(0)
    await engine.index_pages((added,))
    query_gate.set()

    with pytest.raises(SemanticIndexNotReadyError, match="incomplete"):
        await search_task
    await engine.close()


@pytest.mark.asyncio
async def test_search_metrics_include_readiness_and_snapshot_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document("alpha")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((document,))
    provider = _provider_for_documents(
        engine,
        (document,),
        ((1.0, 0.0, 0.0),),
        query_vectors={"query": (1.0, 0.0, 0.0)},
    )
    await engine.index_embeddings(provider)
    original_info = engine.index.get_semantic_index_info
    original_snapshot = engine.index.load_semantic_snapshot

    async def delayed_info(model: EmbeddingModelInfo) -> SemanticIndexInfo:
        await asyncio.sleep(0.01)
        return await original_info(model)

    async def delayed_snapshot(
        model: EmbeddingModelInfo,
    ) -> SemanticEmbeddingSnapshot:
        await asyncio.sleep(0.01)
        snapshot = await original_snapshot(model)
        return replace(
            snapshot,
            sqlite_snapshot_fetch_time_ms=11.0,
            vector_decode_time_ms=12.0,
            provenance_materialization_time_ms=13.0,
        )

    monkeypatch.setattr(engine.index, "get_semantic_index_info", delayed_info)
    monkeypatch.setattr(engine.index, "load_semantic_snapshot", delayed_snapshot)

    result = await engine.semantic_search_with_metrics(
        "query",
        provider=provider,
    )
    await engine.close()

    assert result.readiness_check_time_ms >= 5.0
    assert result.sqlite_snapshot_fetch_time_ms == 11.0
    assert result.vector_decode_time_ms == 12.0
    assert result.provenance_materialization_time_ms == 13.0
    assert result.total_retrieval_time_ms >= 15.0


@pytest.mark.asyncio
async def test_dimension_mismatch_fails_batch_without_corrupting_index(
    tmp_path: Path,
) -> None:
    document = _document("alpha")
    engine = ContextEngine(tmp_path / "semantic.sqlite3")
    await engine.index_pages((document,))
    chunk = engine.chunker.chunk(document)[0]
    provider = DeterministicEmbeddingProvider(
        document_vectors={
            format_semantic_document(chunk): (1.0, 0.0),
        },
        query_vectors={},
        dimension=3,
    )

    result = await engine.index_embeddings(provider)
    info = await engine.get_semantic_index_info(provider)
    await engine.close()

    assert result.embedded_chunks == 0
    assert result.failed_chunks == 1
    assert result.warnings == (
        "Embedding batch ending at chunk 1 failed with ValueError.",
    )
    assert not info.ready
