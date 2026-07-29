"""Deterministic paired BM25 versus semantic evaluation comparison."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal, cast

from crawlforge.evaluation.models import EvaluationRun, QueryEvaluation

ComparisonReportFormat = Literal["json", "markdown"]


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """One aggregate candidate-minus-baseline metric difference."""

    metric: str
    bm25: float | None
    semantic: float | None
    delta: float | None


@dataclass(frozen=True, slots=True)
class CategoryDelta:
    """One paired category MRR comparison."""

    category: str
    bm25_mrr: float
    semantic_mrr: float
    delta: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Exploratory paired bootstrap interval over query-level deltas."""

    metric: str
    mean_delta: float | None
    lower_95: float | None
    upper_95: float | None
    bootstrap_samples: int
    seed: int
    paired_query_count: int


@dataclass(frozen=True, slots=True)
class QueryComparison:
    """Paired quality and source evidence for one query."""

    query_id: str
    query: str
    category: str
    bm25_first_relevant_rank: int | None
    semantic_first_relevant_rank: int | None
    recall_at_5_delta: float
    ndcg_at_5_delta: float
    bm25_sources: tuple[str, ...]
    semantic_sources: tuple[str, ...]
    bm25_only_sources: tuple[str, ...]
    semantic_only_sources: tuple[str, ...]
    bm25_failure: str | None
    semantic_failure: str | None


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    """Complete paired BM25-versus-semantic comparison report."""

    dataset_name: str
    dataset_version: str
    dataset_signature: str
    focus_limit: int
    baseline_strategy: str
    candidate_strategy: str
    chunking_configuration: dict[str, object]
    bm25_retrieval_configuration: dict[str, object]
    semantic_retrieval_configuration: dict[str, object]
    metrics: tuple[MetricDelta, ...]
    category_metrics: tuple[CategoryDelta, ...]
    uncertainty: tuple[BootstrapInterval, ...]
    query_comparisons: tuple[QueryComparison, ...]
    semantic_wins: tuple[str, ...]
    bm25_wins: tuple[str, ...]
    both_succeed: tuple[str, ...]
    both_fail: tuple[str, ...]
    semantic_regressions: tuple[str, ...]
    warnings: tuple[str, ...]


def compare_evaluation_runs(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
    *,
    focus_limit: int = 5,
    bootstrap_samples: int = 5000,
    seed: int = 20260729,
) -> EvaluationComparison:
    """Validate and compare two paired runs without changing their rankings."""
    if focus_limit <= 0:
        raise ValueError("focus_limit must be greater than zero")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be greater than zero")
    _validate_pair(bm25, semantic, focus_limit)

    baseline_by_id = {result.query_id: result for result in bm25.query_results}
    candidate_by_id = {result.query_id: result for result in semantic.query_results}
    query_comparisons: list[QueryComparison] = []
    semantic_wins: list[str] = []
    bm25_wins: list[str] = []
    both_succeed: list[str] = []
    both_fail: list[str] = []
    semantic_regressions: list[str] = []
    for query_id in baseline_by_id:
        baseline = baseline_by_id[query_id]
        candidate = candidate_by_id[query_id]
        query_comparisons.append(_query_comparison(baseline, candidate, focus_limit))
        baseline_quality = _query_quality(baseline, focus_limit)
        candidate_quality = _query_quality(candidate, focus_limit)
        baseline_success = _query_succeeded(baseline, focus_limit)
        candidate_success = _query_succeeded(candidate, focus_limit)
        if baseline_success and candidate_success:
            both_succeed.append(query_id)
        if not baseline_success and not candidate_success:
            both_fail.append(query_id)
        if candidate_quality > baseline_quality:
            semantic_wins.append(query_id)
        elif baseline_quality > candidate_quality:
            bm25_wins.append(query_id)
        if baseline_success and (
            not candidate_success or candidate_quality < baseline_quality
        ):
            semantic_regressions.append(query_id)

    metrics = _aggregate_metric_deltas(bm25, semantic, focus_limit)
    positive_pairs = [
        (baseline_by_id[query_id], candidate_by_id[query_id])
        for query_id in baseline_by_id
        if baseline_by_id[query_id].category != "negative"
        and baseline_by_id[query_id].failure is None
        and candidate_by_id[query_id].failure is None
    ]
    bootstrap_metrics: tuple[
        tuple[str, Callable[[QueryEvaluation, int], float]],
        ...,
    ] = (
        ("Hit@5", _query_hit),
        ("Recall@5", _query_recall),
        ("MRR", _query_mrr),
        ("NDCG@5", _query_ndcg),
    )
    uncertainty = tuple(
        _paired_bootstrap(
            metric_name,
            [
                metric(candidate, focus_limit) - metric(baseline, focus_limit)
                for baseline, candidate in positive_pairs
            ],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for metric_name, metric in bootstrap_metrics
    )
    category_metrics = _category_deltas(bm25, semantic)
    warnings = [
        "Bootstrap intervals are exploratory estimates from a small synthetic dataset.",
        (
            "Intervals do not establish statistical significance or transferability "
            "to real websites."
        ),
        (
            "Semantic cosine scores are not calibrated confidence and no negative "
            "query threshold was fitted."
        ),
    ]
    if not positive_pairs:
        warnings.append(
            "No non-failed paired positive queries were available for bootstrap "
            "uncertainty; interval values are unavailable."
        )
    return EvaluationComparison(
        dataset_name=bm25.dataset_name,
        dataset_version=bm25.dataset_version,
        dataset_signature=bm25.dataset_signature,
        focus_limit=focus_limit,
        baseline_strategy=bm25.retrieval_strategy,
        candidate_strategy=semantic.retrieval_strategy,
        chunking_configuration=dict(bm25.chunking_configuration),
        bm25_retrieval_configuration=dict(bm25.retrieval_configuration),
        semantic_retrieval_configuration=dict(semantic.retrieval_configuration),
        metrics=metrics,
        category_metrics=category_metrics,
        uncertainty=uncertainty,
        query_comparisons=tuple(query_comparisons),
        semantic_wins=tuple(semantic_wins),
        bm25_wins=tuple(bm25_wins),
        both_succeed=tuple(both_succeed),
        both_fail=tuple(both_fail),
        semantic_regressions=tuple(semantic_regressions),
        warnings=tuple(warnings),
    )


def render_comparison_json(comparison: EvaluationComparison) -> str:
    """Render a stable machine-readable paired comparison."""
    payload = cast(dict[str, object], _json_compatible(asdict(comparison)))
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def render_comparison_markdown(comparison: EvaluationComparison) -> str:
    """Render aggregate, category, uncertainty, and query-level paired evidence."""
    semantic_configuration = comparison.semantic_retrieval_configuration
    lines = [
        "# CrawlForge BM25 vs Semantic Retrieval",
        "",
        "## Reproducibility",
        "",
        f"- Dataset: `{comparison.dataset_name}` `{comparison.dataset_version}`",
        f"- Dataset signature: `{comparison.dataset_signature}`",
        f"- Baseline: `{comparison.baseline_strategy}`",
        f"- Candidate: `{comparison.candidate_strategy}`",
        (
            "- Semantic model: "
            f"`{semantic_configuration.get('model_id', 'unknown')}` at "
            f"`{semantic_configuration.get('model_revision', 'unknown')}`"
        ),
        (
            "- Semantic vectors: "
            f"{semantic_configuration.get('dimension', 'unknown')} dimensions, "
            f"{semantic_configuration.get('precision', 'unknown')}, normalized="
            f"{str(semantic_configuration.get('normalized', 'unknown')).lower()}"
        ),
        (
            "- Semantic formatter: "
            f"`{semantic_configuration.get('document_format_version', 'unknown')}` "
            f"documents, "
            f"`{semantic_configuration.get('query_format_version', 'unknown')}` "
            "queries"
        ),
        f"- Semantic device: `{semantic_configuration.get('device', 'unknown')}`",
        (
            "- Both runs use the same corpus, chunks, judgments, K values, "
            "and token budget."
        ),
        "",
        "## Aggregate comparison",
        "",
        "| Metric | BM25 | Semantic | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in comparison.metrics:
        lines.append(
            f"| {metric.metric} | {_value(metric.bm25)} | "
            f"{_value(metric.semantic)} | {_signed(metric.delta)} |"
        )
    lines.extend(
        [
            "",
            "## Category comparison",
            "",
            "| Category | BM25 MRR | Semantic MRR | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for category in comparison.category_metrics:
        lines.append(
            f"| `{category.category}` | {category.bm25_mrr:.4f} | "
            f"{category.semantic_mrr:.4f} | {category.delta:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap uncertainty",
            "",
            (
                "These deterministic 95% percentile intervals are exploratory. "
                "The dataset is small and synthetic; the intervals do not prove "
                "generalization to real sites or automatic statistical significance."
            ),
            "",
            "| Metric | Mean delta | Lower 95% | Upper 95% | Samples | Seed |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for interval in comparison.uncertainty:
        lines.append(
            f"| {interval.metric} | {_signed(interval.mean_delta)} | "
            f"{_signed(interval.lower_95)} | {_signed(interval.upper_95)} | "
            f"{interval.bootstrap_samples} | {interval.seed} |"
        )
    lines.extend(
        [
            "",
            "## Failure analysis",
            "",
            f"- Semantic wins: {_query_ids(comparison.semantic_wins)}",
            f"- BM25 wins: {_query_ids(comparison.bm25_wins)}",
            f"- Both succeed: {_query_ids(comparison.both_succeed)}",
            f"- Both fail: {_query_ids(comparison.both_fail)}",
            (f"- Semantic regressions: {_query_ids(comparison.semantic_regressions)}"),
            "",
            "## Selected query analysis",
            "",
        ]
    )
    selected = {
        query.query_id: query
        for query in comparison.query_comparisons
        if query.query_id in {"q031", "q041", "q046", "q048"}
    }
    for query_id in ("q031", "q041", "q046", "q048"):
        query = selected.get(query_id)
        if query is None:
            continue
        lines.append(
            f"- `{query.query_id}` ({query.category}): first relevant rank "
            f"BM25 {_rank(query.bm25_first_relevant_rank)}, semantic "
            f"{_rank(query.semantic_first_relevant_rank)}; Recall@5 delta "
            f"{query.recall_at_5_delta:+.4f}; NDCG@5 delta "
            f"{query.ndcg_at_5_delta:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Query-level paired evidence",
            "",
            (
                "| Query | Category | BM25 first relevant | Semantic first relevant | "
                "Recall@5 delta | NDCG@5 delta | BM25-only sources | "
                "Semantic-only sources |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for query in comparison.query_comparisons:
        lines.append(
            f"| `{query.query_id}` {_escape(query.query)} | `{query.category}` | "
            f"{query.bm25_first_relevant_rank or 'none'} | "
            f"{query.semantic_first_relevant_rank or 'none'} | "
            f"{query.recall_at_5_delta:+.4f} | {query.ndcg_at_5_delta:+.4f} | "
            f"{_sources(query.bm25_only_sources)} | "
            f"{_sources(query.semantic_only_sources)} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The corpus and judgments are small, synthetic, and English-focused.",
            "- The semantic baseline uses an English lightweight embedding model.",
            "- Exact vector search is linear in chunk count and embedding dimension.",
            "- Model download is required unless the pinned revision is cached.",
            "- Token counts are deterministic approximations, not model-exact counts.",
            "- Semantic scores are cosine similarities, not calibrated confidence.",
            "- No hybrid retrieval, reranking, or negative-query threshold is used.",
        ]
    )
    lines.extend(f"- Warning: {_escape(warning)}" for warning in comparison.warnings)
    lines.append("")
    return "\n".join(lines)


def write_comparison_report(
    comparison: EvaluationComparison,
    path: str | Path,
    *,
    report_format: ComparisonReportFormat,
) -> None:
    """Atomically replace one comparison report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = (
        render_comparison_json(comparison)
        if report_format == "json"
        else render_comparison_markdown(comparison)
    )
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_pair(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
    focus_limit: int,
) -> None:
    if bm25.dataset_name != semantic.dataset_name:
        raise ValueError("paired runs use different dataset names")
    if bm25.dataset_version != semantic.dataset_version:
        raise ValueError("paired runs use different dataset versions")
    if not bm25.dataset_signature or (
        bm25.dataset_signature != semantic.dataset_signature
    ):
        raise ValueError("paired runs use different dataset signatures")
    if bm25.chunking_configuration != semantic.chunking_configuration:
        raise ValueError("paired runs use different chunking configurations")
    if bm25.corpus_statistics.chunk_count != semantic.corpus_statistics.chunk_count:
        raise ValueError("paired runs use different chunks")
    baseline_ids = tuple(result.query_id for result in bm25.query_results)
    candidate_ids = tuple(result.query_id for result in semantic.query_results)
    if baseline_ids != candidate_ids:
        raise ValueError("paired runs use different query order or membership")
    for key in ("limit_values", "token_budget"):
        if bm25.retrieval_configuration.get(
            key
        ) != semantic.retrieval_configuration.get(key):
            raise ValueError(f"paired runs use different {key}")
    limits = bm25.retrieval_configuration.get("limit_values")
    if not isinstance(limits, list) or focus_limit not in limits:
        raise ValueError(f"paired runs do not include K={focus_limit}")


def _aggregate_metric_deltas(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
    limit: int,
) -> tuple[MetricDelta, ...]:
    baseline = bm25.aggregate_metrics
    candidate = semantic.aggregate_metrics
    values = (
        ("Hit@5", baseline.hit_rate_at[limit], candidate.hit_rate_at[limit]),
        (
            "Precision@5",
            baseline.precision_at[limit],
            candidate.precision_at[limit],
        ),
        ("Recall@5", baseline.recall_at[limit], candidate.recall_at[limit]),
        ("MRR", baseline.mrr, candidate.mrr),
        ("MAP@5", baseline.map_at[limit], candidate.map_at[limit]),
        ("NDCG@5", baseline.ndcg_at[limit], candidate.ndcg_at[limit]),
        (
            "Negative no-result accuracy",
            baseline.no_result_accuracy,
            candidate.no_result_accuracy,
        ),
    )
    return tuple(
        MetricDelta(
            metric=name,
            bm25=left,
            semantic=right,
            delta=(right - left if left is not None and right is not None else None),
        )
        for name, left, right in values
    )


def _category_deltas(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
) -> tuple[CategoryDelta, ...]:
    candidate = {
        summary.category: summary.metrics.mrr for summary in semantic.category_metrics
    }
    return tuple(
        CategoryDelta(
            category=summary.category,
            bm25_mrr=summary.metrics.mrr,
            semantic_mrr=candidate[summary.category],
            delta=candidate[summary.category] - summary.metrics.mrr,
        )
        for summary in bm25.category_metrics
        if summary.category in candidate
    )


def _query_comparison(
    bm25: QueryEvaluation,
    semantic: QueryEvaluation,
    limit: int,
) -> QueryComparison:
    bm25_sources = tuple(
        dict.fromkeys(item.canonical_url for item in bm25.retrieved_items)
    )
    semantic_sources = tuple(
        dict.fromkeys(item.canonical_url for item in semantic.retrieved_items)
    )
    return QueryComparison(
        query_id=bm25.query_id,
        query=bm25.query,
        category=bm25.category,
        bm25_first_relevant_rank=bm25.first_relevant_rank,
        semantic_first_relevant_rank=semantic.first_relevant_rank,
        recall_at_5_delta=(
            semantic.metrics.recall_at[limit] - bm25.metrics.recall_at[limit]
        ),
        ndcg_at_5_delta=(semantic.metrics.ndcg_at[limit] - bm25.metrics.ndcg_at[limit]),
        bm25_sources=bm25_sources,
        semantic_sources=semantic_sources,
        bm25_only_sources=tuple(
            source for source in bm25_sources if source not in semantic_sources
        ),
        semantic_only_sources=tuple(
            source for source in semantic_sources if source not in bm25_sources
        ),
        bm25_failure=bm25.failure,
        semantic_failure=semantic.failure,
    )


def _query_quality(result: QueryEvaluation, limit: int) -> tuple[float, ...]:
    if result.failure is not None:
        return (-1.0, -1.0, -1.0)
    if result.category == "negative":
        return (
            float(result.metrics.no_result_correct is True),
            float(-result.candidate_count),
            0.0,
        )
    return (
        result.metrics.ndcg_at[limit],
        result.metrics.recall_at[limit],
        result.metrics.reciprocal_rank,
    )


def _query_hit(result: QueryEvaluation, limit: int) -> float:
    return result.metrics.hit_rate_at[limit]


def _query_recall(result: QueryEvaluation, limit: int) -> float:
    return result.metrics.recall_at[limit]


def _query_mrr(result: QueryEvaluation, _limit: int) -> float:
    return result.metrics.reciprocal_rank


def _query_ndcg(result: QueryEvaluation, limit: int) -> float:
    return result.metrics.ndcg_at[limit]


def _query_succeeded(result: QueryEvaluation, limit: int) -> bool:
    if result.failure is not None:
        return False
    if result.category == "negative":
        return result.metrics.no_result_correct is True
    return result.metrics.hit_rate_at[limit] > 0


def _paired_bootstrap(
    metric: str,
    deltas: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> BootstrapInterval:
    if not deltas:
        return BootstrapInterval(
            metric=metric,
            mean_delta=None,
            lower_95=None,
            upper_95=None,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            paired_query_count=0,
        )
    randomizer = random.Random(seed)
    size = len(deltas)
    samples = sorted(
        fmean(deltas[randomizer.randrange(size)] for _ in range(size))
        for _ in range(bootstrap_samples)
    )
    return BootstrapInterval(
        metric=metric,
        mean_delta=fmean(deltas),
        lower_95=_percentile(samples, 0.025),
        upper_95=_percentile(samples, 0.975),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        paired_query_count=size,
    )


def _percentile(sorted_values: list[float], quantile: float) -> float:
    index = (len(sorted_values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported comparison value: {type(value).__name__}")


def _value(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _rank(value: int | None) -> str:
    return str(value) if value is not None else "none"


def _signed(value: float | None) -> str:
    return f"{value:+.4f}" if value is not None else "n/a"


def _query_ids(values: tuple[str, ...]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "none"


def _sources(values: tuple[str, ...]) -> str:
    return _escape(", ".join(values)) if values else "none"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
