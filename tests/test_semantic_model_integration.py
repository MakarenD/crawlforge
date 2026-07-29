"""Explicit opt-in smoke test for the pinned real embedding model."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from crawlforge.semantic_models import (
    DEFAULT_SEMANTIC_DIMENSION,
    DEFAULT_SEMANTIC_MODEL_ID,
    DEFAULT_SEMANTIC_MODEL_REVISION,
    exact_cosine_similarity,
)
from crawlforge.semantic_provider import SentenceTransformerEmbeddingProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("CRAWLFORGE_RUN_SEMANTIC_MODEL_TESTS") != "1",
    reason="set CRAWLFORGE_RUN_SEMANTIC_MODEL_TESTS=1 to load the pinned model",
)


@pytest.mark.asyncio
async def test_pinned_model_produces_normalized_semantic_ranking() -> None:
    """Load the real model, validate its vectors, and rank controlled documents."""
    cache_value = os.environ.get("CRAWLFORGE_SEMANTIC_CACHE")
    provider = SentenceTransformerEmbeddingProvider(
        model_id=DEFAULT_SEMANTIC_MODEL_ID,
        revision=DEFAULT_SEMANTIC_MODEL_REVISION,
        dimension=DEFAULT_SEMANTIC_DIMENSION,
        device="cpu",
        cache_directory=Path(cache_value) if cache_value else None,
        local_files_only=os.environ.get("CRAWLFORGE_SEMANTIC_OFFLINE") == "1",
    )
    documents = (
        "A puppy runs and plays outside in a grassy park.",
        "SQLite uses indexes to accelerate database queries.",
    )
    try:
        document_vectors = await provider.embed_documents(documents)
        query_vectors = await provider.embed_queries(("a dog playing outdoors",))
    finally:
        await provider.close()

    assert len(document_vectors) == 2
    assert len(query_vectors) == 1
    assert all(
        vector.dimension == DEFAULT_SEMANTIC_DIMENSION
        for vector in (*document_vectors, *query_vectors)
    )
    for vector in (*document_vectors, *query_vectors):
        assert all(math.isfinite(value) for value in vector.values)
        norm = math.sqrt(sum(value * value for value in vector.values))
        assert norm == pytest.approx(1.0, abs=1e-5)

    query = query_vectors[0]
    scores = tuple(
        exact_cosine_similarity(query, document, normalized=True)
        for document in document_vectors
    )
    assert scores[0] > scores[1]
