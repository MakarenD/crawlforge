"""Typed models for deterministic retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

QueryCategory = Literal[
    "exact_term",
    "code_symbol",
    "error_lookup",
    "paraphrase",
    "conceptual",
    "ambiguous",
    "multi_relevant",
    "negative",
]

QUERY_CATEGORIES: tuple[QueryCategory, ...] = (
    "exact_term",
    "code_symbol",
    "error_lookup",
    "paraphrase",
    "conceptual",
    "ambiguous",
    "multi_relevant",
    "negative",
)


@dataclass(frozen=True, slots=True)
class EvaluationSection:
    """Stable dataset section identity independent of generated chunks."""

    section_id: str
    heading_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """One versioned local corpus document."""

    document_id: str
    path: str
    url: str
    title: str
    sections: tuple[EvaluationSection, ...]
    content: str


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """A graded stable source or section expected for one query."""

    judgment_id: str
    document_id: str
    relevance: int
    canonical_source: str | None = None
    section_id: str | None = None
    heading_path: tuple[str, ...] = ()
    evidence: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """One benchmark query and its human-authored ground truth."""

    query_id: str
    query: str
    category: QueryCategory
    relevant_sources: tuple[RelevanceJudgment, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Loaded evaluation dataset plus its non-reportable local root."""

    schema_version: int
    name: str
    version: str
    description: str
    documents: tuple[EvaluationDocument, ...]
    queries: tuple[EvaluationQuery, ...]
    root: Path


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    """Strategy-neutral ranked retrieval item with stable provenance."""

    rank: int
    document_id: str
    url: str
    canonical_url: str
    title: str
    section_id: str | None
    heading_path: tuple[str, ...]
    text: str
    score: float
    estimated_tokens: int
    source_estimated_tokens: int
    content_hash: str
    relevance_grade: int = 0
    matched_judgment_id: str | None = None


@dataclass(frozen=True, slots=True)
class QueryMetricValues:
    """Standard IR metric values for one query."""

    hit_rate_at: dict[int, float]
    precision_at: dict[int, float]
    recall_at: dict[int, float]
    average_precision_at: dict[int, float]
    ndcg_at: dict[int, float]
    reciprocal_rank: float
    no_result_correct: bool | None


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Retrieval, relevance, metric, latency, and context evidence for one query."""

    query_id: str
    query: str
    category: QueryCategory
    expected_sources: tuple[RelevanceJudgment, ...]
    retrieved_items: tuple[RetrievedItem, ...]
    context_item_ranks: tuple[int, ...]
    context_relevant_chunk_count: int
    first_relevant_rank: int | None
    missed_judgment_ids: tuple[str, ...]
    metrics: QueryMetricValues
    candidate_count: int
    returned_estimated_tokens: int
    relevant_estimated_tokens: int
    irrelevant_estimated_tokens: int
    source_estimated_tokens: int
    estimated_context_reduction: float
    relevant_chunks_per_1000_estimated_tokens: float
    irrelevant_estimated_token_ratio: float
    relevant_source_coverage: float
    latency_samples_ms: tuple[float, ...]
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate standard IR metrics, excluding negatives from positive metrics."""

    query_count: int
    positive_query_count: int
    negative_query_count: int
    failed_query_count: int
    hit_rate_at: dict[int, float]
    precision_at: dict[int, float]
    recall_at: dict[int, float]
    map_at: dict[int, float]
    ndcg_at: dict[int, float]
    mrr: float
    no_result_accuracy: float | None


@dataclass(frozen=True, slots=True)
class ContextQualitySummary:
    """CrawlForge-specific context-efficiency measurements."""

    mean_candidate_count: float
    mean_returned_estimated_tokens: float
    relevant_chunks_per_1000_estimated_tokens: float
    irrelevant_estimated_token_ratio: float
    mean_relevant_source_coverage: float
    mean_estimated_context_reduction: float


@dataclass(frozen=True, slots=True)
class CategorySummary:
    """Metrics for one explicit benchmark query category."""

    category: QueryCategory
    metrics: MetricSummary
    context_quality: ContextQualitySummary


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Warm-index search latency measurements for the current machine."""

    sample_count: int
    repeat_count: int
    warmup_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    maximum_ms: float


@dataclass(frozen=True, slots=True)
class CorpusStatistics:
    """Offline corpus ingestion and resulting index size."""

    document_count: int
    section_count: int
    chunk_count: int
    source_size_bytes: int
    cleaned_size_bytes: int
    source_estimated_tokens: int
    cleaned_estimated_tokens: int
    indexing_time_ms: float


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Complete machine-readable output of one retrieval evaluation run."""

    dataset_name: str
    dataset_version: str
    retrieval_strategy: str
    retrieval_configuration: dict[str, object]
    chunking_configuration: dict[str, object]
    timestamp: str
    corpus_statistics: CorpusStatistics
    query_results: tuple[QueryEvaluation, ...]
    aggregate_metrics: MetricSummary
    category_metrics: tuple[CategorySummary, ...]
    latency: LatencySummary
    context_quality: ContextQualitySummary
    worst_queries: tuple[str, ...]
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
