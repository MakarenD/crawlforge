"""Standard IR metrics and explicitly project-specific context measurements."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

from crawlforge.evaluation.models import (
    ContextQualitySummary,
    EvaluationQuery,
    LatencySummary,
    MetricSummary,
    QueryEvaluation,
    QueryMetricValues,
    RetrievedItem,
)


def query_metric_values(
    query: EvaluationQuery,
    retrieved: Sequence[RetrievedItem],
    limits: Sequence[int],
) -> QueryMetricValues:
    """Calculate standard metrics from independently authored judgments."""
    grades = [item.relevance_grade for item in retrieved]
    matched_ids = [item.matched_judgment_id for item in retrieved]
    positive_judgments = tuple(
        judgment for judgment in query.relevant_sources if judgment.relevance > 0
    )
    total_relevant = len(positive_judgments)
    ideal_grades = sorted(
        (judgment.relevance for judgment in positive_judgments),
        reverse=True,
    )
    is_negative = query.category == "negative"
    return QueryMetricValues(
        hit_rate_at={
            limit: float(any(grade > 0 for grade in grades[:limit])) for limit in limits
        },
        precision_at={limit: precision_at(grades, limit) for limit in limits},
        recall_at={
            limit: recall_at(matched_ids, total_relevant, limit) for limit in limits
        },
        average_precision_at={
            limit: average_precision_at(grades, total_relevant, limit)
            for limit in limits
        },
        ndcg_at={limit: ndcg_at(grades, ideal_grades, limit) for limit in limits},
        reciprocal_rank=reciprocal_rank(grades),
        no_result_correct=not retrieved if is_negative else None,
    )


def precision_at(grades: Sequence[int], limit: int) -> float:
    """Return relevant retrieved items divided by the requested cutoff K."""
    _validate_limit(limit)
    return sum(grade > 0 for grade in grades[:limit]) / limit


def recall_at(
    matched_judgment_ids: Sequence[str | None],
    total_relevant: int,
    limit: int,
) -> float:
    """Return unique known judgments found in the first K results."""
    _validate_limit(limit)
    if total_relevant <= 0:
        return 0.0
    matched = {
        judgment_id
        for judgment_id in matched_judgment_ids[:limit]
        if judgment_id is not None
    }
    return len(matched) / total_relevant


def reciprocal_rank(grades: Sequence[int]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    for rank, grade in enumerate(grades, start=1):
        if grade > 0:
            return 1.0 / rank
    return 0.0


def average_precision_at(
    grades: Sequence[int],
    total_relevant: int,
    limit: int,
) -> float:
    """Return AP@K with at most one credited item per stable judgment."""
    _validate_limit(limit)
    if total_relevant <= 0:
        return 0.0
    accumulated = 0.0
    relevant_seen = 0
    for rank, grade in enumerate(grades[:limit], start=1):
        if grade <= 0:
            continue
        relevant_seen += 1
        accumulated += relevant_seen / rank
    return accumulated / min(total_relevant, limit)


def dcg_at(grades: Sequence[int], limit: int) -> float:
    """Return graded discounted cumulative gain using exponential gains."""
    _validate_limit(limit)
    return math.fsum(
        float(2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:limit], start=1)
        if grade > 0
    )


def ndcg_at(
    grades: Sequence[int],
    ideal_grades: Sequence[int],
    limit: int,
) -> float:
    """Return normalized DCG@K, or zero when no positive judgment exists."""
    ideal = dcg_at(sorted(ideal_grades, reverse=True), limit)
    if ideal == 0:
        return 0.0
    return dcg_at(grades, limit) / ideal


def summarize_metrics(
    query_results: Sequence[QueryEvaluation],
    limits: Sequence[int],
) -> MetricSummary:
    """Average positive-query IR metrics and negative-query abstention separately."""
    successful = [result for result in query_results if result.failure is None]
    positives = [result for result in successful if result.category != "negative"]
    negatives = [result for result in successful if result.category == "negative"]
    return MetricSummary(
        query_count=len(query_results),
        positive_query_count=len(positives),
        negative_query_count=len(negatives),
        failed_query_count=len(query_results) - len(successful),
        hit_rate_at={
            limit: _mean(result.metrics.hit_rate_at[limit] for result in positives)
            for limit in limits
        },
        precision_at={
            limit: _mean(result.metrics.precision_at[limit] for result in positives)
            for limit in limits
        },
        recall_at={
            limit: _mean(result.metrics.recall_at[limit] for result in positives)
            for limit in limits
        },
        map_at={
            limit: _mean(
                result.metrics.average_precision_at[limit] for result in positives
            )
            for limit in limits
        },
        ndcg_at={
            limit: _mean(result.metrics.ndcg_at[limit] for result in positives)
            for limit in limits
        },
        mrr=_mean(result.metrics.reciprocal_rank for result in positives),
        no_result_accuracy=(
            _mean(
                float(result.metrics.no_result_correct is True) for result in negatives
            )
            if negatives
            else None
        ),
    )


def summarize_context_quality(
    query_results: Sequence[QueryEvaluation],
) -> ContextQualitySummary:
    """Aggregate CrawlForge-specific efficiency measures without IR branding."""
    successful = [result for result in query_results if result.failure is None]
    positives = [result for result in successful if result.category != "negative"]
    total_tokens = sum(result.returned_estimated_tokens for result in successful)
    relevant_chunks = sum(result.context_relevant_chunk_count for result in successful)
    irrelevant_tokens = sum(result.irrelevant_estimated_tokens for result in successful)
    return ContextQualitySummary(
        mean_candidate_count=_mean(
            float(result.candidate_count) for result in successful
        ),
        mean_returned_estimated_tokens=_mean(
            float(result.returned_estimated_tokens) for result in successful
        ),
        relevant_chunks_per_1000_estimated_tokens=(
            relevant_chunks * 1000 / total_tokens if total_tokens else 0.0
        ),
        irrelevant_estimated_token_ratio=(
            irrelevant_tokens / total_tokens if total_tokens else 0.0
        ),
        mean_relevant_source_coverage=_mean(
            result.relevant_source_coverage for result in positives
        ),
        mean_estimated_context_reduction=_mean(
            result.estimated_context_reduction for result in successful
        ),
    )


def summarize_latency(
    samples_ms: Iterable[float],
    *,
    repeat_count: int,
    warmup_count: int,
) -> LatencySummary:
    """Summarize warm-index timings using a deterministic nearest-rank p95."""
    samples = sorted(samples_ms)
    if not samples:
        return LatencySummary(
            sample_count=0,
            repeat_count=repeat_count,
            warmup_count=warmup_count,
            mean_ms=0.0,
            median_ms=0.0,
            p95_ms=0.0,
            maximum_ms=0.0,
        )
    p95_index = max(0, math.ceil(0.95 * len(samples)) - 1)
    return LatencySummary(
        sample_count=len(samples),
        repeat_count=repeat_count,
        warmup_count=warmup_count,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        p95_ms=samples[p95_index],
        maximum_ms=samples[-1],
    )


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("metric limit must be greater than zero")


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return statistics.fmean(collected) if collected else 0.0
