"""Hybrid evaluator adapter tests over deterministic production retrieval."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from semantic_fakes import DeterministicEmbeddingProvider

from crawlforge.context_engine import ContextEngine
from crawlforge.context_models import DocumentBlock, SourceDocument
from crawlforge.evaluation.hybrid_strategy import HybridContextEngineStrategy
from crawlforge.evaluation.models import (
    EvaluationDataset,
    EvaluationDocument,
    EvaluationSection,
)
from crawlforge.evaluation.runner import RetrievalStrategy
from crawlforge.hybrid_models import HybridRetrievalError, HybridSearchConfig
from crawlforge.semantic_models import format_semantic_document


@pytest.mark.asyncio
async def test_hybrid_strategy_preserves_final_rrf_ranking_and_evidence(
    tmp_path: Path,
) -> None:
    documents = (
        _document("alpha", "alpha lexical token controls the crawler."),
        _document("beta", "vector meaning explains bounded requests."),
    )
    dataset = _dataset(documents, root=tmp_path)
    engine = ContextEngine(tmp_path / "hybrid-evaluation.sqlite3")
    await engine.index_pages(documents)
    provider = _provider(
        engine,
        documents,
        document_vectors=((0.0, 1.0), (1.0, 0.0)),
        query_vectors={"alpha": (1.0, 0.0)},
    )
    indexing = await engine.index_embeddings(provider, batch_size=2)
    index_info = await engine.get_semantic_index_info(provider)
    config = HybridSearchConfig(
        rrf_k=60,
        bm25_candidate_limit=5,
        semantic_candidate_limit=5,
    )
    strategy = HybridContextEngineStrategy(
        engine,
        provider,
        dataset,
        config=config,
        indexing_result=indexing,
        index_info=index_info,
    )

    first = tuple(await strategy.search("alpha", limit=2))
    repeated = tuple(await strategy.search("alpha", limit=2))
    metadata = first[0].strategy_metadata
    performance = strategy.performance_metadata()
    await engine.close()

    assert isinstance(strategy, RetrievalStrategy)
    assert strategy.name == "hybrid-rrf"
    assert strategy.retrieval_strategy == strategy.name
    assert len(first) == 2
    assert [(item.rank, item.document_id) for item in first] == [
        (1, "benchmark-alpha"),
        (2, "benchmark-beta"),
    ]
    assert [item.content_hash for item in repeated] == [
        item.content_hash for item in first
    ]
    assert first[0].score == pytest.approx((1.0 / 61.0) + (1.0 / 62.0))
    assert metadata["retrieval_strategy"] == "hybrid-rrf"
    assert metadata["fusion_strategy"] == "reciprocal-rank-fusion"
    assert metadata["score_type"] == "rrf_score"
    assert metadata["bm25_rank"] == 1
    assert metadata["semantic_rank"] == 2
    assert metadata["bm25_contribution"] == pytest.approx(1.0 / 61.0)
    assert metadata["semantic_contribution"] == pytest.approx(1.0 / 62.0)
    assert isinstance(metadata["bm25_score"], float)
    assert metadata["cosine_similarity"] == pytest.approx(0.0)
    assert metadata["rrf_k"] == 60
    assert metadata["bm25_weight"] == 1.0
    assert metadata["semantic_weight"] == 1.0
    assert metadata["bm25_candidate_limit"] == 5
    assert metadata["semantic_candidate_limit"] == 5
    assert metadata["model_id"] == "test/controlled"
    assert metadata["model_revision"] == "test-revision"
    assert isinstance(metadata["model_fingerprint"], str)
    assert first[1].strategy_metadata["bm25_rank"] is None
    assert first[1].strategy_metadata["semantic_rank"] == 1
    assert first[1].strategy_metadata["bm25_score"] is None
    assert performance["ranking"] == "reciprocal_rank_fusion"
    assert performance["execution_mode"] == "sequential"
    assert performance["strict_component_success"] is True
    assert performance["rrf_k"] == 60
    assert performance["bm25_candidate_count_max"] == 1
    assert performance["semantic_candidate_count_max"] == 2
    assert performance["overlapping_candidate_count_max"] == 1
    assert performance["unique_candidate_count_max"] == 2
    assert performance["embedded_chunks"] == 2
    assert performance["stored_vector_bytes"] == 16
    assert performance["device"] == "cpu"
    assert provider.query_calls == [("alpha",), ("alpha",)]
    assert provider.close_calls == 0


@pytest.mark.asyncio
async def test_hybrid_strategy_keeps_provider_caller_owned_on_strict_failure(
    tmp_path: Path,
) -> None:
    documents = (_document("alpha", "alpha lexical token."),)
    dataset = _dataset(documents, root=tmp_path)
    engine = ContextEngine(tmp_path / "missing-semantic-index.sqlite3")
    await engine.index_pages(documents)
    provider = _provider(
        engine,
        documents,
        document_vectors=((1.0, 0.0),),
        query_vectors={"alpha": (1.0, 0.0)},
    )
    strategy = HybridContextEngineStrategy(engine, provider, dataset)

    with pytest.raises(HybridRetrievalError) as failure:
        await strategy.search("alpha", limit=1)
    await engine.close()

    assert failure.value.strategy == "semantic"
    assert provider.close_calls == 0
    assert any("does not silently fall back" in item for item in strategy.warnings)


def _document(name: str, body: str) -> SourceDocument:
    heading = f"{name.title()} section"
    content_hash = hashlib.sha256(f"{name}\0{body}".encode()).hexdigest()
    return SourceDocument(
        id=f"source-{name}",
        url=f"https://benchmark.invalid/{name}",
        canonical_url=f"https://benchmark.invalid/{name}",
        title=f"{name.title()} guide",
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


def _dataset(
    documents: tuple[SourceDocument, ...],
    *,
    root: Path,
) -> EvaluationDataset:
    return EvaluationDataset(
        schema_version=1,
        name="hybrid-strategy-fixture",
        version="1.0.0",
        description="Deterministic hybrid strategy fixture.",
        documents=tuple(
            EvaluationDocument(
                document_id=f"benchmark-{document.id.removeprefix('source-')}",
                path=f"{document.id}.html",
                url=document.url,
                title=document.title,
                sections=(
                    EvaluationSection(
                        section_id=f"section-{document.id}",
                        heading_path=document.blocks[0].heading_path,
                    ),
                ),
                content=document.text,
            )
            for document in documents
        ),
        queries=(),
        root=root,
    )


def _provider(
    engine: ContextEngine,
    documents: tuple[SourceDocument, ...],
    *,
    document_vectors: tuple[tuple[float, ...], ...],
    query_vectors: dict[str, tuple[float, ...]],
) -> DeterministicEmbeddingProvider:
    chunks = tuple(
        chunk for document in documents for chunk in engine.chunker.chunk(document)
    )
    return DeterministicEmbeddingProvider(
        document_vectors={
            format_semantic_document(chunk): vector
            for chunk, vector in zip(chunks, document_vectors, strict=True)
        },
        query_vectors=query_vectors,
        dimension=len(document_vectors[0]),
    )
