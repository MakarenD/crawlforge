"""Deterministic retrieval evaluation over CrawlForge's public context API."""

from crawlforge.evaluation.dataset import (
    DatasetValidationError,
    filter_dataset,
    load_dataset,
    validate_dataset,
)
from crawlforge.evaluation.models import (
    QUERY_CATEGORIES,
    CategorySummary,
    ContextQualitySummary,
    CorpusStatistics,
    EvaluationDataset,
    EvaluationDocument,
    EvaluationQuery,
    EvaluationRun,
    EvaluationSection,
    LatencySummary,
    MetricSummary,
    QueryCategory,
    QueryEvaluation,
    QueryMetricValues,
    RelevanceJudgment,
    RetrievedItem,
)
from crawlforge.evaluation.runner import (
    BM25ContextEngineStrategy,
    RetrievalEvaluationRunner,
    RetrievalStrategy,
    ingest_evaluation_corpus,
)

__all__ = [
    "QUERY_CATEGORIES",
    "BM25ContextEngineStrategy",
    "CategorySummary",
    "ContextQualitySummary",
    "CorpusStatistics",
    "DatasetValidationError",
    "EvaluationDataset",
    "EvaluationDocument",
    "EvaluationQuery",
    "EvaluationRun",
    "EvaluationSection",
    "LatencySummary",
    "MetricSummary",
    "QueryCategory",
    "QueryEvaluation",
    "QueryMetricValues",
    "RelevanceJudgment",
    "RetrievalEvaluationRunner",
    "RetrievalStrategy",
    "RetrievedItem",
    "filter_dataset",
    "ingest_evaluation_corpus",
    "load_dataset",
    "validate_dataset",
]
