"""Typed contracts for deterministic rank-fused hybrid retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from crawlforge.context_models import IndexInfo, SourceReference, TextChunk
from crawlforge.semantic_models import SemanticIndexInfo

type RetrievalSource = Literal["bm25", "semantic"]
type HybridFailureSource = Literal["bm25", "semantic", "fusion"]
type HybridExecutionMode = Literal["sequential"]


@dataclass(frozen=True, slots=True)
class HybridSearchConfig:
    """Fixed RRF and candidate-depth configuration for one retriever."""

    rrf_k: int = 60
    bm25_weight: float = 1.0
    semantic_weight: float = 1.0
    bm25_candidate_limit: int = 50
    semantic_candidate_limit: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.rrf_k, int) or isinstance(self.rrf_k, bool):
            raise ValueError("rrf_k must be an integer")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        for name, value in (
            ("bm25_weight", self.bm25_weight),
            ("semantic_weight", self.semantic_weight),
        ):
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.bm25_weight == 0 and self.semantic_weight == 0:
            raise ValueError("at least one retrieval weight must be greater than zero")
        for name, value in (
            ("bm25_candidate_limit", self.bm25_candidate_limit),
            ("semantic_candidate_limit", self.semantic_candidate_limit),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One strategy-ranked candidate supplied to a fusion implementation."""

    identity: str
    rank: int
    chunk: TextChunk
    source: SourceReference
    raw_score: float
    score_type: str

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("candidate identity must not be empty")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError("candidate rank must be an integer")
        if self.rank <= 0:
            raise ValueError("candidate rank must be greater than zero")
        if not math.isfinite(self.raw_score):
            raise ValueError("candidate raw score must be finite")
        if not self.score_type:
            raise ValueError("candidate score type must not be empty")


@dataclass(frozen=True, slots=True)
class RankedCandidateList:
    """One named, weighted, already ordered strategy ranking."""

    strategy: str
    weight: float
    candidates: tuple[RankedCandidate, ...]

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("ranking strategy must not be empty")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight):
            raise ValueError("ranking weight must be finite")
        if self.weight < 0:
            raise ValueError("ranking weight must not be negative")


@dataclass(frozen=True, slots=True)
class RankContribution:
    """Auditable contribution from one component rank."""

    strategy: str
    rank: int
    weight: float
    raw_score: float
    score_type: str
    contribution: float

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("contribution strategy must not be empty")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError("contribution rank must be an integer")
        if self.rank <= 0:
            raise ValueError("contribution rank must be greater than zero")
        if self.weight < 0:
            raise ValueError("contribution weight must not be negative")
        for name, value in (
            ("weight", self.weight),
            ("raw_score", self.raw_score),
            ("contribution", self.contribution),
        ):
            if not math.isfinite(value):
                raise ValueError(f"contribution {name} must be finite")


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """One final candidate produced solely from component ranks."""

    identity: str
    rank: int
    chunk: TextChunk
    source: SourceReference
    rrf_score: float
    contributions: tuple[RankContribution, ...]
    bm25_rank: int | None
    semantic_rank: int | None
    bm25_score: float | None
    cosine_similarity: float | None

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("fused identity must not be empty")
        if self.rank <= 0:
            raise ValueError("fused rank must be greater than zero")
        if not math.isfinite(self.rrf_score):
            raise ValueError("RRF score must be finite")


@runtime_checkable
class RankFusionStrategy(Protocol):
    """Combine existing rankings without interpreting their raw scores."""

    @property
    def name(self) -> str:
        """Return the stable fusion strategy name."""

    def fuse(
        self,
        *,
        rankings: Sequence[RankedCandidateList],
        limit: int,
    ) -> Sequence[FusedCandidate]:
        """Return a deterministic final ranking."""


@dataclass(frozen=True, slots=True)
class HybridSearchHit:
    """One fused chunk with component evidence and complete provenance."""

    identity: str
    chunk: TextChunk
    rank: int
    rrf_score: float
    source: SourceReference
    bm25_rank: int | None
    semantic_rank: int | None
    bm25_contribution: float
    semantic_contribution: float
    bm25_score: float | None
    cosine_similarity: float | None
    contributions: tuple[RankContribution, ...]
    model_id: str
    model_revision: str | None
    model_fingerprint: str
    fusion_configuration: HybridSearchConfig
    retrieval_strategy: str = "hybrid-rrf"
    fusion_strategy: str = "reciprocal-rank-fusion"
    score_type: str = "rrf_score"

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("hybrid identity must not be empty")
        if self.rank <= 0:
            raise ValueError("hybrid rank must be greater than zero")
        for name, value in (
            ("rrf_score", self.rrf_score),
            ("bm25_contribution", self.bm25_contribution),
            ("semantic_contribution", self.semantic_contribution),
        ):
            if not math.isfinite(value):
                raise ValueError(f"hybrid {name} must be finite")
        for name, optional_value in (
            ("bm25_score", self.bm25_score),
            ("cosine_similarity", self.cosine_similarity),
        ):
            if optional_value is not None and not math.isfinite(optional_value):
                raise ValueError(f"hybrid {name} must be finite when present")
        if not self.model_fingerprint:
            raise ValueError("model fingerprint must not be empty")


@dataclass(frozen=True, slots=True)
class HybridSearchMetrics:
    """Separated warm-search timings and candidate counts."""

    bm25_retrieval_time_ms: float
    semantic_readiness_check_time_ms: float
    semantic_query_encoding_time_ms: float
    semantic_snapshot_fetch_time_ms: float
    semantic_vector_decode_time_ms: float
    semantic_exact_scan_time_ms: float
    semantic_provenance_materialization_time_ms: float
    semantic_retrieval_time_ms: float
    fusion_time_ms: float
    provenance_context_hydration_time_ms: float
    total_retrieval_time_ms: float
    sum_component_durations_ms: float
    bm25_candidate_count: int
    semantic_candidate_count: int
    overlapping_candidate_count: int
    unique_candidate_count: int
    execution_mode: HybridExecutionMode = "sequential"
    parallel_wall_time_ms: float | None = None

    def __post_init__(self) -> None:
        timing_values = (
            self.bm25_retrieval_time_ms,
            self.semantic_readiness_check_time_ms,
            self.semantic_query_encoding_time_ms,
            self.semantic_snapshot_fetch_time_ms,
            self.semantic_vector_decode_time_ms,
            self.semantic_exact_scan_time_ms,
            self.semantic_provenance_materialization_time_ms,
            self.semantic_retrieval_time_ms,
            self.fusion_time_ms,
            self.provenance_context_hydration_time_ms,
            self.total_retrieval_time_ms,
            self.sum_component_durations_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in timing_values):
            raise ValueError("hybrid timing values must be finite and non-negative")
        counts = (
            self.bm25_candidate_count,
            self.semantic_candidate_count,
            self.overlapping_candidate_count,
            self.unique_candidate_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("hybrid candidate counts must be non-negative")
        if self.parallel_wall_time_ms is not None and (
            not math.isfinite(self.parallel_wall_time_ms)
            or self.parallel_wall_time_ms < 0
        ):
            raise ValueError("parallel wall time must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """Fused results plus reproducibility and latency metadata."""

    query: str
    hits: tuple[HybridSearchHit, ...]
    metrics: HybridSearchMetrics
    limit: int
    model_id: str
    model_revision: str | None
    model_fingerprint: str
    fusion_configuration: HybridSearchConfig
    retrieval_strategy: str = "hybrid-rrf"
    fusion_strategy: str = "reciprocal-rank-fusion"
    score_type: str = "rrf_score"


@dataclass(frozen=True, slots=True)
class HybridContextResult:
    """Complete fused chunks selected under the shared token budget."""

    query: str
    hits: tuple[HybridSearchHit, ...]
    total_size_chars: int
    estimated_tokens: int
    candidates_considered: int
    search_time_ms: float
    context_selection_time_ms: float
    limit: int
    token_budget: int
    source_estimated_tokens: int
    estimated_context_reduction: float
    index_hit: bool
    model_id: str
    model_revision: str | None
    model_fingerprint: str
    fusion_configuration: HybridSearchConfig
    metrics: HybridSearchMetrics
    retrieval_strategy: str = "hybrid-rrf"
    fusion_strategy: str = "reciprocal-rank-fusion"
    score_type: str = "rrf_score"


@dataclass(frozen=True, slots=True)
class HybridIndexReadiness:
    """Combined lexical and semantic readiness for strict hybrid search."""

    lexical: IndexInfo
    semantic: SemanticIndexInfo
    lexical_ready: bool
    semantic_ready: bool
    ready: bool
    model_id: str
    model_revision: str | None
    model_fingerprint: str


class HybridRetrievalError(RuntimeError):
    """Actionable strict failure from one required hybrid component."""

    def __init__(self, strategy: HybridFailureSource, error: Exception) -> None:
        self.strategy = strategy
        self.cause_type = type(error).__name__
        detail = str(error).strip() or self.cause_type
        super().__init__(
            f"Hybrid retrieval requires the {strategy} strategy, but it failed: "
            f"{detail}"
        )
