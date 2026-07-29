"""Typed models and validation for optional local semantic retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from crawlforge.context_models import SourceReference, TextChunk

DEFAULT_SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_SEMANTIC_DIMENSION = 384
DOCUMENT_FORMAT_VERSION = "crawlforge-semantic-document-v1"
QUERY_FORMAT_VERSION = "crawlforge-semantic-query-v1"
EMBEDDING_DTYPE = "float32"
EMBEDDING_PRECISION = "float32"

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]
type DeviceName = Literal["auto", "cpu", "mps", "cuda"]


class SemanticDependencyError(RuntimeError):
    """Raised when the optional semantic runtime is unavailable."""


class SemanticIndexNotReadyError(RuntimeError):
    """Raised when semantic search has no compatible indexed vectors."""


class SemanticIndexIncompatibleError(RuntimeError):
    """Raised when only embeddings from another fingerprint are available."""


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """One finite, non-empty embedding vector."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("embedding vector must not be empty")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("embedding vector values must be finite")

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> EmbeddingVector:
        """Copy and validate a numeric vector."""
        converted: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("embedding vector values must be numbers")
            converted.append(float(value))
        return cls(tuple(converted))

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return len(self.values)


@dataclass(frozen=True, slots=True)
class EmbeddingRuntimeInfo:
    """Safe runtime metadata exposed after a provider loads its model."""

    device: str
    sentence_transformers_version: str
    transformers_version: str
    torch_version: str
    model_load_time_ms: float
    max_sequence_length: int
    model_cache_size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingInputStatistics:
    """Aggregate tokenizer lengths without retaining tokenizer output."""

    configured_max_sequence_length: int
    input_count: int
    truncated_input_count: int
    maximum_tokenized_length: int
    average_tokenized_length: float

    @property
    def truncated_fraction(self) -> float:
        """Return the fraction of inputs exceeding the model limit."""
        if self.input_count == 0:
            return 0.0
        return self.truncated_input_count / self.input_count


@dataclass(frozen=True, slots=True)
class EmbeddingModelInfo:
    """Stable model configuration used to isolate compatible vectors."""

    fingerprint: str
    provider: str
    model_id: str
    model_revision: str | None
    dimension: int
    normalized: bool
    dtype: str
    precision: str
    document_format_version: str
    query_format_version: str
    metadata: dict[str, JSONValue]

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raise ValueError("model fingerprint must not be empty")
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be greater than zero")
        _validated_json_metadata(self.metadata)


@dataclass(frozen=True, slots=True)
class SemanticChunkRecord:
    """One indexed chunk plus its stable SQLite identity and provenance."""

    storage_id: int
    chunk: TextChunk
    source: SourceReference


@dataclass(frozen=True, slots=True)
class StoredChunkEmbedding:
    """One hydrated chunk and its compatible stored vector."""

    record: SemanticChunkRecord
    vector: EmbeddingVector


@dataclass(frozen=True, slots=True)
class SemanticEmbeddingSnapshot:
    """One coherent ready-index snapshot with separated load timings."""

    embeddings: tuple[StoredChunkEmbedding, ...]
    sqlite_snapshot_fetch_time_ms: float
    vector_decode_time_ms: float
    provenance_materialization_time_ms: float
    stored_vector_bytes: int


@dataclass(frozen=True, slots=True)
class SemanticIndexPlan:
    """Missing/stale work determined before model inference starts."""

    considered_chunks: int
    cache_hits: int
    invalidated_embeddings: int
    missing_chunks: tuple[SemanticChunkRecord, ...]


@dataclass(frozen=True, slots=True)
class SemanticIndexingResult:
    """Aggregate outcome and timings for one embedding indexing session."""

    session_id: str
    model: EmbeddingModelInfo
    considered_chunks: int
    embedded_chunks: int
    cache_hits: int
    invalidated_embeddings: int
    failed_chunks: int
    elapsed_time_ms: float
    model_load_time_ms: float
    document_encoding_time_ms: float
    sqlite_write_time_ms: float
    stored_vector_bytes: int
    total_stored_vector_bytes: int
    batch_size: int
    input_statistics: EmbeddingInputStatistics | None
    warnings: tuple[str, ...] = ()

    @property
    def embeddings_per_second(self) -> float:
        """Return document encoding throughput."""
        seconds = self.document_encoding_time_ms / 1000
        return self.embedded_chunks / seconds if seconds > 0 else 0.0

    @property
    def average_bytes_per_chunk(self) -> float:
        """Return mean stored vector bytes per newly embedded chunk."""
        if self.embedded_chunks == 0:
            return 0.0
        return self.stored_vector_bytes / self.embedded_chunks


@dataclass(frozen=True, slots=True)
class SemanticSearchHit:
    """A cosine-ranked text chunk and its complete provenance."""

    chunk: TextChunk
    rank: int
    cosine_similarity: float
    source: SourceReference
    model_id: str
    model_revision: str | None
    model_fingerprint: str
    retrieval_strategy: str = "semantic-exact-cosine"
    score_type: str = "cosine_similarity"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("semantic rank must be greater than zero")
        if not math.isfinite(self.cosine_similarity):
            raise ValueError("cosine similarity must be finite")


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    """Semantic hits plus separated retrieval timing measurements."""

    query: str
    hits: tuple[SemanticSearchHit, ...]
    readiness_check_time_ms: float
    query_encoding_time_ms: float
    sqlite_snapshot_fetch_time_ms: float
    vector_decode_time_ms: float
    exact_scan_time_ms: float
    provenance_materialization_time_ms: float
    total_retrieval_time_ms: float
    loaded_vector_bytes: int
    loaded_vector_memory_estimate_bytes: int


@dataclass(frozen=True, slots=True)
class SemanticContextResult:
    """Bounded semantic context with model and provenance metadata."""

    query: str
    hits: tuple[SemanticSearchHit, ...]
    total_size_chars: int
    estimated_tokens: int
    candidates_considered: int
    search_time_ms: float
    limit: int
    token_budget: int
    source_estimated_tokens: int
    estimated_context_reduction: float
    index_hit: bool
    model_id: str
    model_revision: str | None
    model_fingerprint: str
    retrieval_strategy: str = "semantic"
    score_type: str = "cosine_similarity"


@dataclass(frozen=True, slots=True)
class SemanticIndexInfo:
    """Readiness and storage summary for one model fingerprint."""

    model: EmbeddingModelInfo
    total_chunks: int
    embedded_chunks: int
    missing_chunks: int
    stored_vector_bytes: int
    last_indexed_at: str | None
    ready: bool
    compatible_model_registered: bool
    other_model_count: int


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Typed asynchronous boundary for document and query embeddings."""

    @property
    def implementation(self) -> str:
        """Return the stable provider implementation name."""

    @property
    def model_id(self) -> str:
        """Return the configured model identifier."""

    @property
    def model_revision(self) -> str | None:
        """Return the configured immutable model revision, when provided."""

    @property
    def dimension(self) -> int:
        """Return the configured and subsequently verified dimension."""

    @property
    def normalized(self) -> bool:
        """Return whether the provider produces unit-normalized vectors."""

    @property
    def precision(self) -> str:
        """Return the provider output precision."""

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        """Return JSON-safe configuration and runtime metadata."""

    @property
    def runtime_info(self) -> EmbeddingRuntimeInfo | None:
        """Return safe runtime details after model initialization."""

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        """Embed formatted document inputs."""

    async def embed_queries(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        """Embed raw query inputs."""

    async def analyze_document_inputs(
        self,
        texts: Sequence[str],
    ) -> EmbeddingInputStatistics:
        """Measure tokenizer lengths without retaining tokenized inputs."""

    async def close(self) -> None:
        """Release provider-owned model resources."""


def format_semantic_document(chunk: TextChunk) -> str:
    """Format stable document context for embedding."""
    heading = " > ".join(chunk.heading_path)
    return f"Title: {chunk.document_title}\nSection: {heading}\n\n{chunk.text}"


def format_semantic_query(query: str) -> str:
    """Normalize only outer whitespace for query embedding."""
    normalized = query.strip()
    if not normalized:
        raise ValueError("semantic query must not be empty")
    return normalized


def embedding_model_info(provider: EmbeddingProvider) -> EmbeddingModelInfo:
    """Build a validated, stable model fingerprint from provider configuration."""
    metadata = dict(provider.metadata)
    validated_metadata = _validated_json_metadata(metadata)
    fingerprint_payload: dict[str, JSONValue] = {
        "provider": provider.implementation,
        "model_id": provider.model_id,
        "model_revision": provider.model_revision,
        "dimension": provider.dimension,
        "normalized": provider.normalized,
        "dtype": EMBEDDING_DTYPE,
        "precision": provider.precision,
        "document_format_version": DOCUMENT_FORMAT_VERSION,
        "query_format_version": QUERY_FORMAT_VERSION,
    }
    encoded = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return EmbeddingModelInfo(
        fingerprint=hashlib.sha256(encoded).hexdigest(),
        provider=provider.implementation,
        model_id=provider.model_id,
        model_revision=provider.model_revision,
        dimension=provider.dimension,
        normalized=provider.normalized,
        dtype=EMBEDDING_DTYPE,
        precision=provider.precision,
        document_format_version=DOCUMENT_FORMAT_VERSION,
        query_format_version=QUERY_FORMAT_VERSION,
        metadata=validated_metadata,
    )


def validate_embedding_batch(
    vectors: Sequence[EmbeddingVector],
    *,
    expected_count: int,
    dimension: int,
    normalized: bool,
) -> tuple[EmbeddingVector, ...]:
    """Validate provider count, dimension, finiteness, and normalization."""
    copied = tuple(vectors)
    if len(copied) != expected_count:
        raise ValueError(
            "embedding provider returned "
            f"{len(copied)} vectors for {expected_count} inputs"
        )
    for vector in copied:
        if vector.dimension != dimension:
            raise ValueError(
                "embedding vector dimension mismatch: "
                f"expected {dimension}, received {vector.dimension}"
            )
        if normalized:
            norm = math.sqrt(math.fsum(value * value for value in vector.values))
            if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("embedding provider returned a non-normalized vector")
    return copied


def serialize_embedding_vector(vector: EmbeddingVector) -> bytes:
    """Serialize a vector as stable little-endian float32 bytes."""
    return struct.pack(f"<{vector.dimension}f", *vector.values)


def deserialize_embedding_vector(blob: bytes, *, dimension: int) -> EmbeddingVector:
    """Decode and validate a little-endian float32 vector."""
    if dimension <= 0:
        raise ValueError("embedding dimension must be greater than zero")
    expected_bytes = dimension * 4
    if len(blob) != expected_bytes:
        raise ValueError(
            "stored embedding blob length mismatch: "
            f"expected {expected_bytes}, received {len(blob)}"
        )
    return EmbeddingVector(tuple(struct.unpack(f"<{dimension}f", blob)))


def exact_cosine_similarity(
    left: EmbeddingVector,
    right: EmbeddingVector,
    *,
    normalized: bool,
) -> float:
    """Return exact cosine similarity for equal-dimension finite vectors."""
    if left.dimension != right.dimension:
        raise ValueError("cannot compare embedding vectors with different dimensions")
    dot = math.fsum(a * b for a, b in zip(left.values, right.values, strict=True))
    if normalized:
        score = dot
    else:
        left_norm = math.sqrt(math.fsum(value * value for value in left.values))
        right_norm = math.sqrt(math.fsum(value * value for value in right.values))
        if left_norm == 0.0 or right_norm == 0.0:
            raise ValueError("cosine similarity is undefined for a zero vector")
        score = dot / (left_norm * right_norm)
    if not math.isfinite(score):
        raise ValueError("cosine similarity is non-finite")
    return max(-1.0, min(1.0, score))


def _validated_json_metadata(
    metadata: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    encoded = json.dumps(
        dict(metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    decoded: object = json.loads(encoded)
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("embedding metadata must be a JSON object")
    return _copy_json_object(decoded)


def _copy_json_object(value: dict[object, object]) -> dict[str, JSONValue]:
    copied: dict[str, JSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("embedding metadata keys must be strings")
        copied[key] = _copy_json_value(item)
    return copied


def _copy_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("embedding metadata numbers must be finite")
        return value
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    if isinstance(value, dict):
        return _copy_json_object(value)
    raise ValueError("embedding metadata must contain only JSON-safe values")
