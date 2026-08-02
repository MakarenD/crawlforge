from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from semantic_fakes import DeterministicEmbeddingProvider

from crawlforge import ContextEngine
from crawlforge.context_models import DocumentBlock, SourceDocument
from crawlforge.hybrid import HybridRetriever
from crawlforge.hybrid_models import HybridRetrievalError, HybridSearchConfig
from crawlforge.semantic_models import format_semantic_document


def _document(name: str, body: str) -> SourceDocument:
    heading = f"{name.title()} guide"
    return SourceDocument(
        id=f"document-{name}",
        url=f"https://example.test/{name}",
        canonical_url=f"https://example.test/{name}",
        title=heading,
        text=f"{heading}\n\n{body}",
        markdown=None,
        status_code=200,
        content_type="text/html",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_hash=hashlib.sha256(f"{name}\0{body}".encode()).hexdigest(),
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


def _provider(
    engine: ContextEngine,
    documents: tuple[SourceDocument, ...],
    *,
    query_gate: asyncio.Event | None = None,
) -> DeterministicEmbeddingProvider:
    chunks = tuple(
        chunk for document in documents for chunk in engine.chunker.chunk(document)
    )
    vectors = (
        (0.8, 0.6, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    assert len(chunks) == len(vectors)
    return DeterministicEmbeddingProvider(
        document_vectors={
            format_semantic_document(chunk): vector
            for chunk, vector in zip(chunks, vectors, strict=True)
        },
        query_vectors={"host politeness": (1.0, 0.0, 0.0)},
        query_gate=query_gate,
    )


def _documents() -> tuple[SourceDocument, ...]:
    return (
        _document(
            "alpha",
            "A polite crawler limits requests to each host and applies delays.",
        ),
        _document(
            "beta",
            "Backpressure prevents an origin from receiving too much work.",
        ),
        _document(
            "gamma",
            "The queue has a deterministic bounded capacity.",
        ),
    )


@pytest.mark.asyncio
async def test_hybrid_retrieval_fuses_production_search_and_bounded_context(
    tmp_path: Path,
) -> None:
    documents = _documents()
    engine = ContextEngine(tmp_path / "hybrid.sqlite3")
    await engine.index_pages(documents)
    provider = _provider(engine, documents)
    retriever = HybridRetriever(context_engine=engine, embedding_provider=provider)
    before = await retriever.get_index_readiness()
    await engine.index_embeddings(provider)

    readiness = await retriever.get_index_readiness()
    result = await retriever.search_with_metrics("host politeness", limit=3)
    repeated = await retriever.search("host politeness", limit=3)
    context = await retriever.build_context(
        "host politeness",
        limit=3,
        token_budget=result.hits[0].chunk.estimated_tokens,
    )
    await engine.close()

    assert not before.ready
    assert before.lexical_ready and not before.semantic_ready
    assert readiness.ready
    assert [hit.source.document_id for hit in result.hits] == [
        "document-alpha",
        "document-beta",
        "document-gamma",
    ]
    assert result.hits[0].bm25_rank == 1
    assert result.hits[0].semantic_rank == 2
    assert result.hits[1].bm25_rank is None
    assert result.hits[1].semantic_rank == 1
    assert result.hits[0].rrf_score == pytest.approx(
        0.01639344262295082 + 0.016129032258064516
    )
    assert result.hits[0].identity == result.hits[0].chunk.content_hash
    assert len({hit.chunk.content_hash for hit in result.hits}) == 3
    assert [hit.chunk.content_hash for hit in repeated] == [
        hit.chunk.content_hash for hit in result.hits
    ]
    assert result.hits[0].source.url == "https://example.test/alpha"
    assert result.hits[0].chunk.heading_path == ("Alpha guide",)
    assert result.hits[0].model_fingerprint == readiness.model_fingerprint
    assert result.metrics.execution_mode == "sequential"
    assert result.metrics.bm25_candidate_count == 1
    assert result.metrics.semantic_candidate_count == 3
    assert result.metrics.overlapping_candidate_count == 1
    assert result.metrics.unique_candidate_count == 3
    assert result.metrics.fusion_time_ms >= 0
    assert result.metrics.total_retrieval_time_ms >= result.metrics.fusion_time_ms
    assert len(context.hits) == 1
    assert context.estimated_tokens <= context.token_budget
    assert context.hits[0].chunk.text == result.hits[0].chunk.text
    assert context.retrieval_strategy == "hybrid-rrf"
    assert context.fusion_strategy == "reciprocal-rank-fusion"
    assert provider.close_calls == 0


@pytest.mark.asyncio
async def test_hybrid_missing_semantic_index_is_typed_and_actionable(
    tmp_path: Path,
) -> None:
    documents = _documents()
    engine = ContextEngine(tmp_path / "missing.sqlite3")
    await engine.index_pages(documents)
    provider = _provider(engine, documents)
    retriever = HybridRetriever(context_engine=engine, embedding_provider=provider)

    with pytest.raises(HybridRetrievalError) as raised:
        await retriever.search("host politeness")
    await engine.close()

    assert raised.value.strategy == "semantic"
    assert raised.value.cause_type == "SemanticIndexNotReadyError"
    assert "crawlforge embed" in str(raised.value)
    assert provider.document_calls == []
    assert provider.close_calls == 0


@pytest.mark.asyncio
async def test_hybrid_wraps_bm25_failure_without_running_semantic(
    tmp_path: Path,
) -> None:
    documents = _documents()
    engine = ContextEngine(tmp_path / "closed.sqlite3")
    await engine.index_pages(documents)
    provider = _provider(engine, documents)
    await engine.index_embeddings(provider)
    retriever = HybridRetriever(context_engine=engine, embedding_provider=provider)
    await engine.close()

    with pytest.raises(HybridRetrievalError) as raised:
        await retriever.search("host politeness")

    assert raised.value.strategy == "bm25"
    assert raised.value.cause_type == "RuntimeError"
    assert provider.query_calls == []


@pytest.mark.asyncio
async def test_hybrid_search_propagates_cancellation_and_keeps_provider_open(
    tmp_path: Path,
) -> None:
    documents = _documents()
    engine = ContextEngine(tmp_path / "cancel.sqlite3")
    indexing_provider = _provider(engine, documents)
    await engine.index_pages(documents)
    await engine.index_embeddings(indexing_provider)
    query_gate = asyncio.Event()
    provider = _provider(engine, documents, query_gate=query_gate)
    retriever = HybridRetriever(context_engine=engine, embedding_provider=provider)
    task = asyncio.create_task(retriever.search("host politeness"))
    while not provider.query_calls:
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    query_gate.set()
    await engine.close()

    assert provider.close_calls == 0


@pytest.mark.asyncio
async def test_hybrid_concurrent_callers_repeat_the_same_ranking(
    tmp_path: Path,
) -> None:
    documents = _documents()
    engine = ContextEngine(tmp_path / "concurrent.sqlite3")
    await engine.index_pages(documents)
    provider = _provider(engine, documents)
    await engine.index_embeddings(provider)
    retriever = HybridRetriever(context_engine=engine, embedding_provider=provider)

    results = await asyncio.gather(
        *(retriever.search("host politeness", limit=3) for _ in range(5))
    )
    await engine.close()

    identities = [tuple(hit.chunk.content_hash for hit in result) for result in results]
    assert identities[1:] == [identities[0]] * 4
    assert provider.query_calls == [("host politeness",)] * 5
    assert provider.close_calls == 0


@pytest.mark.asyncio
async def test_hybrid_rechecks_semantic_readiness_during_concurrent_reindex(
    tmp_path: Path,
) -> None:
    original = _documents()[0]
    added = _documents()[1]
    engine = ContextEngine(tmp_path / "race.sqlite3")
    await engine.index_pages((original,))
    initial_provider = DeterministicEmbeddingProvider(
        document_vectors={
            format_semantic_document(engine.chunker.chunk(original)[0]): (
                1.0,
                0.0,
                0.0,
            )
        },
        query_vectors={"host politeness": (1.0, 0.0, 0.0)},
    )
    await engine.index_embeddings(initial_provider)
    query_gate = asyncio.Event()
    provider = DeterministicEmbeddingProvider(
        document_vectors={},
        query_vectors={"host politeness": (1.0, 0.0, 0.0)},
        query_gate=query_gate,
    )
    retriever = HybridRetriever(context_engine=engine, embedding_provider=provider)
    task = asyncio.create_task(retriever.search("host politeness"))
    while not provider.query_calls:
        await asyncio.sleep(0)
    await engine.index_pages((added,))
    query_gate.set()

    with pytest.raises(HybridRetrievalError) as raised:
        await task
    await engine.close()

    assert raised.value.strategy == "semantic"
    assert raised.value.cause_type == "SemanticIndexNotReadyError"
    assert "incomplete" in str(raised.value)


def test_hybrid_rejects_final_limit_above_either_candidate_depth(
    tmp_path: Path,
) -> None:
    engine = ContextEngine(tmp_path / "limits.sqlite3")
    provider = DeterministicEmbeddingProvider(
        document_vectors={},
        query_vectors={},
    )
    retriever = HybridRetriever(
        context_engine=engine,
        embedding_provider=provider,
        config=HybridSearchConfig(
            bm25_candidate_limit=2,
            semantic_candidate_limit=3,
        ),
    )

    with pytest.raises(ValueError, match="bm25 candidate limit"):
        asyncio.run(retriever.search("query", limit=3))
