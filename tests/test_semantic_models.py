from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import pytest
from semantic_fakes import DeterministicEmbeddingProvider

import crawlforge.semantic_models as semantic_models
from crawlforge.context_models import TextChunk
from crawlforge.semantic_models import (
    DOCUMENT_FORMAT_VERSION,
    QUERY_FORMAT_VERSION,
    EmbeddingVector,
    deserialize_embedding_vector,
    embedding_model_info,
    exact_cosine_similarity,
    format_semantic_document,
    format_semantic_query,
    serialize_embedding_vector,
    validate_embedding_batch,
)


def _chunk() -> TextChunk:
    return TextChunk(
        id="chunk",
        document_id="document",
        ordinal=0,
        source_url="https://example.test/path?ignored=1",
        document_title="Руководство `AsyncCrawler`",
        heading_path=("Ошибки", "Повторные попытки"),
        text="Use `RetryStrategy`:\n\n```python\nretry()\n```",
        size_chars=52,
        estimated_tokens=13,
        content_hash="content-hash",
    )


def _provider(
    *,
    revision: str | None = "revision-a",
    dimension: int = 3,
    normalized: bool = True,
) -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider(
        document_vectors={},
        query_vectors={},
        revision=revision,
        dimension=dimension,
        normalized=normalized,
    )


def test_document_formatter_is_exact_versioned_and_excludes_unstable_metadata() -> None:
    formatted = format_semantic_document(_chunk())

    assert formatted == (
        "Title: Руководство `AsyncCrawler`\n"
        "Section: Ошибки > Повторные попытки\n\n"
        "Use `RetryStrategy`:\n\n```python\nretry()\n```"
    )
    assert DOCUMENT_FORMAT_VERSION == "crawlforge-semantic-document-v1"
    assert "ignored=1" not in formatted
    assert "content-hash" not in formatted


def test_query_formatter_preserves_user_text_without_answer_prefix() -> None:
    assert format_semantic_query("  How are retries bounded?  ") == (
        "How are retries bounded?"
    )
    assert QUERY_FORMAT_VERSION == "crawlforge-semantic-query-v1"
    with pytest.raises(ValueError, match="must not be empty"):
        format_semantic_query(" \n ")


@pytest.mark.parametrize(
    "values,error",
    [
        ((), "must not be empty"),
        ((math.nan,), "must be finite"),
        ((math.inf,), "must be finite"),
    ],
)
def test_embedding_vector_rejects_empty_or_non_finite_values(
    values: tuple[float, ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        EmbeddingVector(values)


def test_embedding_vector_rejects_non_numeric_and_boolean_values() -> None:
    with pytest.raises(TypeError, match="must be numbers"):
        EmbeddingVector.from_sequence([True])
    with pytest.raises(TypeError, match="must be numbers"):
        EmbeddingVector.from_sequence(cast(Sequence[float], ["1"]))


def test_float32_serialization_round_trip_and_length_validation() -> None:
    vector = EmbeddingVector((0.1, -0.25, 0.75))
    blob = serialize_embedding_vector(vector)

    restored = deserialize_embedding_vector(blob, dimension=3)

    assert len(blob) == 12
    assert restored.values == pytest.approx(vector.values)
    with pytest.raises(ValueError, match="blob length mismatch"):
        deserialize_embedding_vector(blob[:-1], dimension=3)


def test_batch_validation_checks_count_dimension_and_normalization() -> None:
    unit = EmbeddingVector((1.0, 0.0, 0.0))
    assert validate_embedding_batch(
        (unit,),
        expected_count=1,
        dimension=3,
        normalized=True,
    ) == (unit,)
    with pytest.raises(ValueError, match="for 2 inputs"):
        validate_embedding_batch(
            (unit,),
            expected_count=2,
            dimension=3,
            normalized=True,
        )
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_embedding_batch(
            (unit,),
            expected_count=1,
            dimension=2,
            normalized=True,
        )
    with pytest.raises(ValueError, match="non-normalized"):
        validate_embedding_batch(
            (EmbeddingVector((0.5, 0.0, 0.0)),),
            expected_count=1,
            dimension=3,
            normalized=True,
        )


def test_model_fingerprint_is_stable_and_covers_compatibility_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = embedding_model_info(_provider())
    repeated = embedding_model_info(_provider())
    revision = embedding_model_info(_provider(revision="revision-b"))
    dimension = embedding_model_info(_provider(dimension=4))
    normalization = embedding_model_info(_provider(normalized=False))

    assert baseline.fingerprint == repeated.fingerprint
    assert len(baseline.fingerprint) == 64
    assert baseline.fingerprint != revision.fingerprint
    assert baseline.fingerprint != dimension.fingerprint
    assert baseline.fingerprint != normalization.fingerprint
    monkeypatch.setattr(
        semantic_models,
        "DOCUMENT_FORMAT_VERSION",
        "crawlforge-semantic-document-v2",
    )
    formatter = embedding_model_info(_provider())

    assert baseline.fingerprint != formatter.fingerprint


def test_exact_cosine_similarity_has_higher_is_better_semantics() -> None:
    query = EmbeddingVector((1.0, 0.0, 0.0))

    assert exact_cosine_similarity(
        query,
        EmbeddingVector((1.0, 0.0, 0.0)),
        normalized=True,
    ) == pytest.approx(1.0)
    assert exact_cosine_similarity(
        query,
        EmbeddingVector((0.0, 1.0, 0.0)),
        normalized=True,
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="different dimensions"):
        exact_cosine_similarity(
            query,
            EmbeddingVector((1.0, 0.0)),
            normalized=True,
        )
