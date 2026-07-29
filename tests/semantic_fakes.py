"""Deterministic controlled embedding provider used only by tests."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from crawlforge.semantic_models import (
    EmbeddingInputStatistics,
    EmbeddingRuntimeInfo,
    EmbeddingVector,
    JSONValue,
)


class DeterministicEmbeddingProvider:
    """Return explicitly configured document and query vectors."""

    implementation = "test-controlled-vectors"

    def __init__(
        self,
        *,
        document_vectors: Mapping[str, Sequence[float]],
        query_vectors: Mapping[str, Sequence[float]],
        model_id: str = "test/controlled",
        revision: str | None = "test-revision",
        dimension: int = 3,
        normalized: bool = True,
        document_gate: asyncio.Event | None = None,
        query_gate: asyncio.Event | None = None,
    ) -> None:
        self._document_vectors = {
            key: EmbeddingVector.from_sequence(value)
            for key, value in document_vectors.items()
        }
        self._query_vectors = {
            key: EmbeddingVector.from_sequence(value)
            for key, value in query_vectors.items()
        }
        self._model_id = model_id
        self._revision = revision
        self._dimension = dimension
        self._normalized = normalized
        self._document_gate = document_gate
        self._query_gate = query_gate
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[tuple[str, ...]] = []
        self.analysis_calls: list[tuple[str, ...]] = []
        self.close_calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str | None:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalized(self) -> bool:
        return self._normalized

    @property
    def precision(self) -> str:
        return "float32"

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        return {"fixture": "controlled", "closed": self.close_calls > 0}

    @property
    def runtime_info(self) -> EmbeddingRuntimeInfo:
        return EmbeddingRuntimeInfo(
            device="cpu",
            sentence_transformers_version="test",
            transformers_version="test",
            torch_version="test",
            model_load_time_ms=0.0,
            max_sequence_length=256,
        )

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        copied = tuple(texts)
        self.document_calls.append(copied)
        if self._document_gate is not None:
            await self._document_gate.wait()
        return tuple(self._document_vectors[text] for text in copied)

    async def embed_queries(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        copied = tuple(texts)
        self.query_calls.append(copied)
        if self._query_gate is not None:
            await self._query_gate.wait()
        return tuple(self._query_vectors[text] for text in copied)

    async def analyze_document_inputs(
        self,
        texts: Sequence[str],
    ) -> EmbeddingInputStatistics:
        copied = tuple(texts)
        self.analysis_calls.append(copied)
        lengths = tuple(len(text.split()) + 2 for text in copied)
        return EmbeddingInputStatistics(
            configured_max_sequence_length=256,
            input_count=len(copied),
            truncated_input_count=sum(length > 256 for length in lengths),
            maximum_tokenized_length=max(lengths, default=0),
            average_tokenized_length=(sum(lengths) / len(lengths) if lengths else 0.0),
        )

    async def close(self) -> None:
        self.close_calls += 1


class ConstantEmbeddingProvider:
    """Return one explicit unit vector for every controlled test input."""

    implementation = "test-constant-vector"

    def __init__(
        self,
        *,
        vector: Sequence[float] = (1.0, 0.0, 0.0),
    ) -> None:
        self._vector = EmbeddingVector.from_sequence(vector)
        self.document_calls = 0
        self.query_calls = 0
        self.close_calls = 0

    @property
    def model_id(self) -> str:
        return "test/constant"

    @property
    def model_revision(self) -> str:
        return "test-revision"

    @property
    def dimension(self) -> int:
        return self._vector.dimension

    @property
    def normalized(self) -> bool:
        return True

    @property
    def precision(self) -> str:
        return "float32"

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        return {"fixture": "constant"}

    @property
    def runtime_info(self) -> EmbeddingRuntimeInfo:
        return EmbeddingRuntimeInfo(
            device="cpu",
            sentence_transformers_version="test",
            transformers_version="test",
            torch_version="test",
            model_load_time_ms=0.0,
            max_sequence_length=256,
        )

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        self.document_calls += 1
        return tuple(self._vector for _ in texts)

    async def embed_queries(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        self.query_calls += 1
        return tuple(self._vector for _ in texts)

    async def analyze_document_inputs(
        self,
        texts: Sequence[str],
    ) -> EmbeddingInputStatistics:
        lengths = tuple(len(text.split()) + 2 for text in texts)
        return EmbeddingInputStatistics(
            configured_max_sequence_length=256,
            input_count=len(lengths),
            truncated_input_count=0,
            maximum_tokenized_length=max(lengths, default=0),
            average_tokenized_length=(sum(lengths) / len(lengths) if lengths else 0.0),
        )

    async def close(self) -> None:
        self.close_calls += 1
