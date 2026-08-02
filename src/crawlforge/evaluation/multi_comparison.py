"""Versioned deterministic comparison of two or more evaluation runs."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Literal, cast

from crawlforge.evaluation.comparison import BootstrapInterval
from crawlforge.evaluation.models import EvaluationRun, QueryEvaluation, RetrievedItem

MULTI_COMPARISON_SCHEMA_VERSION = 1
_OVERLAP_LIMITS = (1, 3, 5, 10)

MultiComparisonReportFormat = Literal["json", "markdown"]
RunCollection = Mapping[str, EvaluationRun] | Sequence[tuple[str, EvaluationRun]]


@dataclass(frozen=True, slots=True)
class ComparedStrategy:
    """Caller alias and implementation name for one evaluated strategy."""

    alias: str
    retrieval_strategy: str


@dataclass(frozen=True, slots=True)
class StrategyMetricValue:
    """One metric value keyed by a caller-provided strategy alias."""

    strategy_alias: str
    value: float | None


@dataclass(frozen=True, slots=True)
class AggregateMetricComparison:
    """One aggregate metric across every compared strategy."""

    metric: str
    values: tuple[StrategyMetricValue, ...]


@dataclass(frozen=True, slots=True)
class CategoryMetricComparison:
    """One category MRR row across every compared strategy."""

    category: str
    values: tuple[StrategyMetricValue, ...]


@dataclass(frozen=True, slots=True)
class PairwiseMetricDelta:
    """Candidate-minus-baseline aggregate metric difference for one pair."""

    metric: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """Metric deltas and paired bootstrap intervals for one strategy pair."""

    baseline_alias: str
    candidate_alias: str
    metrics: tuple[PairwiseMetricDelta, ...]
    uncertainty: tuple[BootstrapInterval, ...]


@dataclass(frozen=True, slots=True)
class FinalItemEvidence:
    """Path-free identity, rank, score, and judgment evidence for one final item."""

    rank: int
    document_id: str
    section_id: str | None
    content_hash: str
    score: float
    relevance_grade: int
    matched_judgment_id: str | None


@dataclass(frozen=True, slots=True)
class StrategyQueryEvidence:
    """One strategy's preserved output and quality evidence for a paired query."""

    strategy_alias: str
    first_relevant_rank: int | None
    recall_at_5: float
    ndcg_at_5: float
    final_items: tuple[FinalItemEvidence, ...]
    missed_judgment_ids: tuple[str, ...]
    failure: str | None


@dataclass(frozen=True, slots=True)
class CandidateOverlap:
    """BM25/semantic stable-chunk overlap for one query and cutoff."""

    limit: int
    intersection_count: int
    union_count: int
    jaccard: float


@dataclass(frozen=True, slots=True)
class CandidateOverlapSummary:
    """Aggregate BM25/semantic stable-chunk overlap at one cutoff."""

    limit: int
    query_count: int
    intersection_count: int
    union_count: int
    jaccard: float
    mean_query_jaccard: float


@dataclass(frozen=True, slots=True)
class MultiQueryComparison:
    """All per-strategy final evidence for one query in shared query order."""

    query_id: str
    query: str
    category: str
    strategies: tuple[StrategyQueryEvidence, ...]
    bm25_semantic_overlap: tuple[CandidateOverlap, ...]


@dataclass(frozen=True, slots=True)
class JudgmentReference:
    """A judgment identity scoped to its query."""

    query_id: str
    judgment_id: str


@dataclass(frozen=True, slots=True)
class UniqueRelevantCoverage:
    """Aggregate unique ground-truth coverage by BM25 and semantic retrieval."""

    limit: int
    ground_truth_count: int
    found_by_both_count: int
    bm25_only_count: int
    semantic_only_count: int
    neither_count: int
    found_by_both: tuple[JudgmentReference, ...]
    bm25_only: tuple[JudgmentReference, ...]
    semantic_only: tuple[JudgmentReference, ...]
    neither: tuple[JudgmentReference, ...]


@dataclass(frozen=True, slots=True)
class OracleUnionRecall:
    """Diagnostic ground-truth BM25/semantic union recall at one cutoff."""

    limit: int
    positive_query_count: int
    found_judgment_count: int
    relevant_judgment_count: int
    recall: float
    mean_query_recall: float


@dataclass(frozen=True, slots=True)
class FusionRecoverySummary:
    """Relevant judgments unique to a component and recovered by hybrid."""

    limit: int
    component_only_count: int
    recovered_count: int
    recovery_rate: float
    bm25_only_count: int
    bm25_only_recovered_count: int
    semantic_only_count: int
    semantic_only_recovered_count: int
    recovered_judgments: tuple[JudgmentReference, ...]


@dataclass(frozen=True, slots=True)
class HybridContributionItem:
    """Auditable additive hybrid metadata for one final top-K result."""

    query_id: str
    rank: int
    document_id: str
    section_id: str | None
    content_hash: str
    score: float
    source_classification: str
    bm25_rank: int | None
    semantic_rank: int | None
    bm25_contribution: float | None
    semantic_contribution: float | None
    contribution_sum: float | None
    bm25_score: float | None
    cosine_similarity: float | None


@dataclass(frozen=True, slots=True)
class HybridContributionSummary:
    """Aggregate final composition and explicitly scoped retention diagnostics."""

    limit: int
    final_item_count: int
    dual_source_count: int
    dual_source_fraction: float
    bm25_only_count: int
    bm25_only_fraction: float
    semantic_only_count: int
    semantic_only_fraction: float
    unattributed_count: int
    unattributed_fraction: float
    average_bm25_contribution: float | None
    average_semantic_contribution: float | None
    average_bm25_score: float | None
    average_cosine_similarity: float | None
    dual_source_promotion_count: int
    dual_source_promotion_rate: float
    single_source_retention_scope: str
    single_source_candidate_count: int
    single_source_retained_count: int
    single_source_retention_rate: float
    items: tuple[HybridContributionItem, ...]


@dataclass(frozen=True, slots=True)
class MultiEvaluationComparison:
    """Versioned, generic comparison report for two or more evaluation runs."""

    schema_version: int
    dataset_name: str
    dataset_version: str
    dataset_signature: str
    focus_limit: int
    bootstrap_samples: int
    seed: int
    limit_values: tuple[int, ...]
    token_budget: int
    chunk_count: int
    chunking_configuration: dict[str, object]
    strategies: tuple[ComparedStrategy, ...]
    aggregate_metrics: tuple[AggregateMetricComparison, ...]
    category_metrics: tuple[CategoryMetricComparison, ...]
    pairwise_comparisons: tuple[PairwiseComparison, ...]
    query_comparisons: tuple[MultiQueryComparison, ...]
    candidate_overlap: tuple[CandidateOverlapSummary, ...]
    unique_relevant_coverage: UniqueRelevantCoverage | None
    oracle_union_recall: tuple[OracleUnionRecall, ...]
    fusion_recovery: FusionRecoverySummary | None
    hybrid_contributions: HybridContributionSummary | None
    hybrid_wins_over_both: tuple[str, ...]
    hybrid_beats_bm25: tuple[str, ...]
    hybrid_beats_semantic: tuple[str, ...]
    bm25_only_wins: tuple[str, ...]
    semantic_only_wins: tuple[str, ...]
    hybrid_regressions: tuple[str, ...]
    all_three_fail: tuple[str, ...]
    warnings: tuple[str, ...]


def compare_multiple_evaluation_runs(
    runs: RunCollection,
    *,
    focus_limit: int = 5,
    bootstrap_samples: int = 5000,
    seed: int = 20260729,
) -> MultiEvaluationComparison:
    """Validate and compare two or more paired runs without changing rankings."""
    if focus_limit != 5:
        raise ValueError("multi-run comparison requires focus_limit=5")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be greater than zero")
    normalized = _normalize_runs(runs)
    limit_values, token_budget = _validate_runs(normalized, focus_limit)

    run_by_alias = dict(normalized)
    if {"bm25", "semantic"}.issubset(run_by_alias) and any(
        limit not in limit_values for limit in _OVERLAP_LIMITS
    ):
        raise ValueError(
            "canonical BM25/semantic diagnostics require K values 1, 3, 5, and 10"
        )
    first_run = normalized[0][1]
    strategies = tuple(
        ComparedStrategy(alias=alias, retrieval_strategy=run.retrieval_strategy)
        for alias, run in normalized
    )
    aggregate_metrics = _aggregate_metrics(normalized, focus_limit)
    category_metrics = _category_metrics(normalized)
    pairwise = tuple(
        _pairwise_comparison(
            left_alias,
            left,
            right_alias,
            right,
            focus_limit=focus_limit,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for (left_alias, left), (right_alias, right) in combinations(normalized, 2)
    )
    query_comparisons = _query_comparisons(normalized, focus_limit)

    bm25 = run_by_alias.get("bm25")
    semantic = run_by_alias.get("semantic")
    hybrid = run_by_alias.get("hybrid")
    overlap: tuple[CandidateOverlapSummary, ...] = ()
    coverage: UniqueRelevantCoverage | None = None
    oracle: tuple[OracleUnionRecall, ...] = ()
    fusion: FusionRecoverySummary | None = None
    contributions: HybridContributionSummary | None = None
    classifications: tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ] = ((), (), (), (), (), (), ())
    warnings = [
        (
            "Paired bootstrap intervals are exploratory 95% percentile estimates; "
            "they do not establish significance or transferability."
        ),
        (
            "Scores and strategy contributions are implementation-specific and are "
            "not calibrated confidence values."
        ),
    ]
    if bm25 is not None and semantic is not None:
        overlap = _candidate_overlap_summaries(bm25, semantic)
        coverage = _unique_relevant_coverage(bm25, semantic, focus_limit)
        oracle = _oracle_union_recall(bm25, semantic)
        warnings.append(
            "Ground-truth BM25/semantic oracle union recall is diagnostic only and "
            "must not be treated as a deployable or production fusion policy."
        )
        if hybrid is not None:
            fusion = _fusion_recovery(bm25, semantic, hybrid, focus_limit)
            contributions = _hybrid_contribution_summary(
                bm25,
                semantic,
                hybrid,
                focus_limit,
            )
            classifications = _canonical_classifications(
                bm25,
                semantic,
                hybrid,
                focus_limit,
            )
            warnings.append(
                "Single-source retention compares exclusive identities from the "
                "standalone BM25 and semantic top-K rankings; it does not describe "
                "the larger hybrid candidate pools."
            )
        else:
            warnings.append(
                "Hybrid-only fusion recovery, contribution, and outcome diagnostics "
                "are unavailable because no `hybrid` alias was provided."
            )
    else:
        warnings.append(
            "BM25/semantic overlap, coverage, and oracle diagnostics are unavailable "
            "unless both canonical aliases are provided."
        )

    return MultiEvaluationComparison(
        schema_version=MULTI_COMPARISON_SCHEMA_VERSION,
        dataset_name=first_run.dataset_name,
        dataset_version=first_run.dataset_version,
        dataset_signature=first_run.dataset_signature,
        focus_limit=focus_limit,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        limit_values=limit_values,
        token_budget=token_budget,
        chunk_count=first_run.corpus_statistics.chunk_count,
        chunking_configuration=dict(first_run.chunking_configuration),
        strategies=strategies,
        aggregate_metrics=aggregate_metrics,
        category_metrics=category_metrics,
        pairwise_comparisons=pairwise,
        query_comparisons=query_comparisons,
        candidate_overlap=overlap,
        unique_relevant_coverage=coverage,
        oracle_union_recall=oracle,
        fusion_recovery=fusion,
        hybrid_contributions=contributions,
        hybrid_wins_over_both=classifications[0],
        hybrid_beats_bm25=classifications[1],
        hybrid_beats_semantic=classifications[2],
        bm25_only_wins=classifications[3],
        semantic_only_wins=classifications[4],
        hybrid_regressions=classifications[5],
        all_three_fail=classifications[6],
        warnings=tuple(warnings),
    )


def render_multi_comparison_json(comparison: MultiEvaluationComparison) -> str:
    """Render stable machine-readable JSON with strict finite numbers."""
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


def render_multi_comparison_markdown(comparison: MultiEvaluationComparison) -> str:
    """Render generic evidence plus canonical BM25/semantic/hybrid diagnostics."""
    aliases = tuple(strategy.alias for strategy in comparison.strategies)
    lines = [
        "# CrawlForge Retrieval Strategy Comparison",
        "",
        "## Reproducibility",
        "",
        f"- Schema version: `{comparison.schema_version}`",
        f"- Dataset: `{comparison.dataset_name}` `{comparison.dataset_version}`",
        f"- Dataset signature: `{comparison.dataset_signature}`",
        f"- Strategies: {', '.join(f'`{alias}`' for alias in aliases)}",
        (
            "- Shared K values: "
            + ", ".join(str(value) for value in comparison.limit_values)
        ),
        f"- Shared token budget: `{comparison.token_budget}`",
        f"- Shared chunk count: {comparison.chunk_count}",
        "",
        "## Aggregate comparison",
        "",
    ]
    if set(aliases) == {"bm25", "semantic", "hybrid"}:
        lines.extend(
            [
                (
                    "| Metric | BM25 | Semantic | Hybrid | Hybrid Δ vs BM25 | "
                    "Hybrid Δ vs Semantic |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in comparison.aggregate_metrics:
            values = _metric_lookup(metric.values)
            hybrid = values["hybrid"]
            lines.append(
                f"| {metric.metric} | {_value(values['bm25'])} | "
                f"{_value(values['semantic'])} | {_value(hybrid)} | "
                f"{_signed_delta(hybrid, values['bm25'])} | "
                f"{_signed_delta(hybrid, values['semantic'])} |"
            )
    else:
        lines.append(
            "| Metric | " + " | ".join(_escape(alias) for alias in aliases) + " |"
        )
        lines.append("| --- | " + " | ".join("---:" for _ in aliases) + " |")
        for metric in comparison.aggregate_metrics:
            values = _metric_lookup(metric.values)
            lines.append(
                f"| {metric.metric} | "
                + " | ".join(_value(values[alias]) for alias in aliases)
                + " |"
            )

    lines.extend(["", "## Category comparison", ""])
    if set(aliases) == {"bm25", "semantic", "hybrid"}:
        lines.extend(
            [
                "| Category | BM25 MRR | Semantic MRR | Hybrid MRR |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for category in comparison.category_metrics:
            values = _metric_lookup(category.values)
            lines.append(
                f"| `{_escape(category.category)}` | {_value(values['bm25'])} | "
                f"{_value(values['semantic'])} | {_value(values['hybrid'])} |"
            )
    else:
        lines.append(
            "| Category | "
            + " | ".join(f"{_escape(alias)} MRR" for alias in aliases)
            + " |"
        )
        lines.append("| --- | " + " | ".join("---:" for _ in aliases) + " |")
        for category in comparison.category_metrics:
            values = _metric_lookup(category.values)
            lines.append(
                f"| `{_escape(category.category)}` | "
                + " | ".join(_value(values[alias]) for alias in aliases)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Pairwise paired-bootstrap uncertainty",
            "",
            (
                "Intervals are deterministic 95% percentile intervals over paired, "
                "non-failed positive queries."
            ),
            "",
            "| Pair | Metric | Mean delta | Lower 95% | Upper 95% | Queries |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair in comparison.pairwise_comparisons:
        pair_name = f"{pair.candidate_alias} Δ vs {pair.baseline_alias}"
        for interval in pair.uncertainty:
            lines.append(
                f"| `{_escape(pair_name)}` | {interval.metric} | "
                f"{_signed(interval.mean_delta)} | {_signed(interval.lower_95)} | "
                f"{_signed(interval.upper_95)} | {interval.paired_query_count} |"
            )

    _append_canonical_diagnostics(lines, comparison)
    _append_query_evidence(lines, comparison)
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The comparison inherits the scope and judgment limits of its dataset.",
            (
                "- Bootstrap intervals quantify this paired sample only; they do not "
                "prove significance or generalization."
            ),
            (
                "- Ground-truth oracle union recall is optimistic diagnostic evidence, "
                "not a production retrieval algorithm."
            ),
            (
                "- Hybrid contribution fields describe this additive implementation "
                "and are not calibrated probabilities."
            ),
        ]
    )
    lines.extend(f"- Warning: {_escape(warning)}" for warning in comparison.warnings)
    lines.append("")
    return "\n".join(lines)


def write_multi_comparison_report(
    comparison: MultiEvaluationComparison,
    path: str | Path,
    *,
    report_format: MultiComparisonReportFormat,
) -> None:
    """Atomically replace one generic comparison report."""
    if report_format not in ("json", "markdown"):
        raise ValueError("report_format must be 'json' or 'markdown'")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = (
        render_multi_comparison_json(comparison)
        if report_format == "json"
        else render_multi_comparison_markdown(comparison)
    )
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _normalize_runs(runs: RunCollection) -> tuple[tuple[str, EvaluationRun], ...]:
    raw = tuple(runs.items()) if isinstance(runs, Mapping) else tuple(runs)
    if len(raw) < 2:
        raise ValueError("at least two evaluation runs are required")
    normalized: list[tuple[str, EvaluationRun]] = []
    aliases: set[str] = set()
    for entry in raw:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError("each evaluation run must be an (alias, run) tuple")
        alias, run = entry
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("strategy aliases must be non-empty strings")
        if alias != alias.strip():
            raise ValueError("strategy aliases must not have surrounding whitespace")
        if alias in aliases:
            raise ValueError(f"duplicate strategy alias: {alias}")
        if not isinstance(run, EvaluationRun):
            raise TypeError(f"strategy {alias!r} does not contain an EvaluationRun")
        aliases.add(alias)
        normalized.append((alias, run))
    return tuple(normalized)


def _validate_runs(
    runs: tuple[tuple[str, EvaluationRun], ...],
    focus_limit: int,
) -> tuple[tuple[int, ...], int]:
    first_alias, first = runs[0]
    if not first.dataset_signature:
        raise ValueError("paired runs require a non-empty dataset signature")
    first_ids = tuple(result.query_id for result in first.query_results)
    if len(first_ids) != len(set(first_ids)):
        raise ValueError(f"strategy {first_alias!r} contains duplicate query IDs")
    limits = _configured_limits(first_alias, first)
    if focus_limit not in limits:
        raise ValueError(f"paired runs do not include K={focus_limit}")
    token_budget = _token_budget(first_alias, first)
    _validate_query_metric_maps(first_alias, first, focus_limit)

    for alias, run in runs[1:]:
        if run.dataset_name != first.dataset_name:
            raise ValueError("paired runs use different dataset names")
        if run.dataset_version != first.dataset_version:
            raise ValueError("paired runs use different dataset versions")
        if run.dataset_signature != first.dataset_signature:
            raise ValueError("paired runs use different dataset signatures")
        if run.chunking_configuration != first.chunking_configuration:
            raise ValueError("paired runs use different chunking configurations")
        if run.corpus_statistics.chunk_count != first.corpus_statistics.chunk_count:
            raise ValueError("paired runs use different chunk counts")
        query_ids = tuple(result.query_id for result in run.query_results)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError(f"strategy {alias!r} contains duplicate query IDs")
        if query_ids != first_ids:
            raise ValueError("paired runs use different query order or membership")
        if _configured_limits(alias, run) != limits:
            raise ValueError("paired runs use different K values")
        if _token_budget(alias, run) != token_budget:
            raise ValueError("paired runs use different token budgets")
        _validate_query_metric_maps(alias, run, focus_limit)
        for expected, actual in zip(
            first.query_results, run.query_results, strict=True
        ):
            if (
                actual.query != expected.query
                or actual.category != expected.category
                or _judgment_signature(actual) != _judgment_signature(expected)
            ):
                raise ValueError(
                    "paired runs use different query text, category, or judgments"
                )
    return limits, token_budget


def _configured_limits(alias: str, run: EvaluationRun) -> tuple[int, ...]:
    raw = run.retrieval_configuration.get("limit_values")
    if not isinstance(raw, list) or not raw:
        raise TypeError(f"strategy {alias!r} limit_values must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise TypeError(f"strategy {alias!r} limit_values must contain integers")
    limits = tuple(raw)
    if any(limit <= 0 for limit in limits) or len(limits) != len(set(limits)):
        raise ValueError(
            f"strategy {alias!r} limit_values must be unique positive integers"
        )
    return limits


def _token_budget(alias: str, run: EvaluationRun) -> int:
    value = run.retrieval_configuration.get("token_budget")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"strategy {alias!r} token_budget must be a positive integer")
    return value


def _validate_query_metric_maps(
    alias: str,
    run: EvaluationRun,
    focus_limit: int,
) -> None:
    aggregate_maps = (
        run.aggregate_metrics.hit_rate_at,
        run.aggregate_metrics.precision_at,
        run.aggregate_metrics.recall_at,
        run.aggregate_metrics.map_at,
        run.aggregate_metrics.ndcg_at,
    )
    if any(focus_limit not in values for values in aggregate_maps):
        raise ValueError(
            f"strategy {alias!r} aggregate metrics do not include K={focus_limit}"
        )
    for result in run.query_results:
        query_maps = (
            result.metrics.hit_rate_at,
            result.metrics.precision_at,
            result.metrics.recall_at,
            result.metrics.average_precision_at,
            result.metrics.ndcg_at,
        )
        if any(focus_limit not in values for values in query_maps):
            raise ValueError(
                f"strategy {alias!r} query {result.query_id!r} metrics do not "
                f"include K={focus_limit}"
            )


def _judgment_signature(result: QueryEvaluation) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            judgment.judgment_id,
            judgment.document_id,
            judgment.relevance,
            judgment.canonical_source,
            judgment.section_id,
            judgment.heading_path,
            judgment.evidence,
        )
        for judgment in result.expected_sources
    )


def _aggregate_metrics(
    runs: tuple[tuple[str, EvaluationRun], ...],
    limit: int,
) -> tuple[AggregateMetricComparison, ...]:
    rows = tuple(_aggregate_values(run, limit) for _, run in runs)
    return tuple(
        AggregateMetricComparison(
            metric=metric,
            values=tuple(
                StrategyMetricValue(alias, dict(values)[metric])
                for (alias, _), values in zip(runs, rows, strict=True)
            ),
        )
        for metric, _ in rows[0]
    )


def _aggregate_values(
    run: EvaluationRun,
    limit: int,
) -> tuple[tuple[str, float | None], ...]:
    metrics = run.aggregate_metrics
    return (
        (f"Hit@{limit}", metrics.hit_rate_at[limit]),
        (f"Precision@{limit}", metrics.precision_at[limit]),
        (f"Recall@{limit}", metrics.recall_at[limit]),
        ("MRR", metrics.mrr),
        (f"MAP@{limit}", metrics.map_at[limit]),
        (f"NDCG@{limit}", metrics.ndcg_at[limit]),
        ("Negative no-result accuracy", metrics.no_result_accuracy),
    )


def _category_metrics(
    runs: tuple[tuple[str, EvaluationRun], ...],
) -> tuple[CategoryMetricComparison, ...]:
    categories = list(
        dict.fromkeys(
            summary.category for _, run in runs for summary in run.category_metrics
        )
    )
    by_alias = {
        alias: {
            summary.category: summary.metrics.mrr for summary in run.category_metrics
        }
        for alias, run in runs
    }
    return tuple(
        CategoryMetricComparison(
            category=category,
            values=tuple(
                StrategyMetricValue(alias, by_alias[alias].get(category))
                for alias, _ in runs
            ),
        )
        for category in categories
    )


def _pairwise_comparison(
    baseline_alias: str,
    baseline: EvaluationRun,
    candidate_alias: str,
    candidate: EvaluationRun,
    *,
    focus_limit: int,
    bootstrap_samples: int,
    seed: int,
) -> PairwiseComparison:
    baseline_values = dict(_aggregate_values(baseline, focus_limit))
    candidate_values = dict(_aggregate_values(candidate, focus_limit))
    metrics = tuple(
        PairwiseMetricDelta(
            metric=metric,
            baseline_value=value,
            candidate_value=candidate_values[metric],
            delta=_metric_delta(candidate_values[metric], value),
        )
        for metric, value in baseline_values.items()
    )
    positive_pairs = [
        (left, right)
        for left, right in zip(
            baseline.query_results,
            candidate.query_results,
            strict=True,
        )
        if left.category != "negative"
        and left.failure is None
        and right.failure is None
    ]
    query_metrics = (
        (f"Hit@{focus_limit}", _query_hit),
        (f"Recall@{focus_limit}", _query_recall),
        ("MRR", _query_mrr),
        (f"NDCG@{focus_limit}", _query_ndcg),
    )
    uncertainty = tuple(
        _paired_bootstrap(
            metric,
            [
                getter(right, focus_limit) - getter(left, focus_limit)
                for left, right in positive_pairs
            ],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for metric, getter in query_metrics
    )
    return PairwiseComparison(
        baseline_alias=baseline_alias,
        candidate_alias=candidate_alias,
        metrics=metrics,
        uncertainty=uncertainty,
    )


def _query_comparisons(
    runs: tuple[tuple[str, EvaluationRun], ...],
    limit: int,
) -> tuple[MultiQueryComparison, ...]:
    run_by_alias = dict(runs)
    bm25 = run_by_alias.get("bm25")
    semantic = run_by_alias.get("semantic")
    comparisons: list[MultiQueryComparison] = []
    for index, first_result in enumerate(runs[0][1].query_results):
        evidence = tuple(
            _strategy_query_evidence(alias, run.query_results[index], limit)
            for alias, run in runs
        )
        overlap = (
            _query_candidate_overlap(
                bm25.query_results[index],
                semantic.query_results[index],
            )
            if bm25 is not None and semantic is not None
            else ()
        )
        comparisons.append(
            MultiQueryComparison(
                query_id=first_result.query_id,
                query=first_result.query,
                category=first_result.category,
                strategies=evidence,
                bm25_semantic_overlap=overlap,
            )
        )
    return tuple(comparisons)


def _strategy_query_evidence(
    alias: str,
    result: QueryEvaluation,
    limit: int,
) -> StrategyQueryEvidence:
    return StrategyQueryEvidence(
        strategy_alias=alias,
        first_relevant_rank=result.first_relevant_rank,
        recall_at_5=result.metrics.recall_at[limit],
        ndcg_at_5=result.metrics.ndcg_at[limit],
        final_items=tuple(
            FinalItemEvidence(
                rank=item.rank,
                document_id=item.document_id,
                section_id=item.section_id,
                content_hash=item.content_hash,
                score=item.score,
                relevance_grade=item.relevance_grade,
                matched_judgment_id=item.matched_judgment_id,
            )
            for item in _ranked_items(result)
        ),
        missed_judgment_ids=result.missed_judgment_ids,
        failure=result.failure,
    )


def _query_candidate_overlap(
    bm25: QueryEvaluation,
    semantic: QueryEvaluation,
) -> tuple[CandidateOverlap, ...]:
    return tuple(
        _candidate_overlap(
            _item_identities(bm25, limit), _item_identities(semantic, limit), limit
        )
        for limit in _OVERLAP_LIMITS
    )


def _candidate_overlap(
    bm25_items: set[tuple[str, str | None, str]],
    semantic_items: set[tuple[str, str | None, str]],
    limit: int,
) -> CandidateOverlap:
    intersection = len(bm25_items & semantic_items)
    union = len(bm25_items | semantic_items)
    return CandidateOverlap(
        limit=limit,
        intersection_count=intersection,
        union_count=union,
        jaccard=(intersection / union if union else 1.0),
    )


def _candidate_overlap_summaries(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
) -> tuple[CandidateOverlapSummary, ...]:
    by_limit: dict[int, list[CandidateOverlap]] = {
        limit: [] for limit in _OVERLAP_LIMITS
    }
    for left, right in zip(bm25.query_results, semantic.query_results, strict=True):
        for overlap in _query_candidate_overlap(left, right):
            by_limit[overlap.limit].append(overlap)
    summaries: list[CandidateOverlapSummary] = []
    for limit in _OVERLAP_LIMITS:
        overlaps = by_limit[limit]
        intersection = sum(value.intersection_count for value in overlaps)
        union = sum(value.union_count for value in overlaps)
        summaries.append(
            CandidateOverlapSummary(
                limit=limit,
                query_count=len(overlaps),
                intersection_count=intersection,
                union_count=union,
                jaccard=(intersection / union if union else 1.0),
                mean_query_jaccard=(
                    fmean(value.jaccard for value in overlaps) if overlaps else 1.0
                ),
            )
        )
    return tuple(summaries)


def _unique_relevant_coverage(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
    limit: int,
) -> UniqueRelevantCoverage:
    both: list[JudgmentReference] = []
    bm25_only: list[JudgmentReference] = []
    semantic_only: list[JudgmentReference] = []
    neither: list[JudgmentReference] = []
    for left, right in zip(bm25.query_results, semantic.query_results, strict=True):
        ground_truth = _ground_truth_judgments(left, right)
        left_found = _matched_judgments(left, limit)
        right_found = _matched_judgments(right, limit)
        for judgment_id in ground_truth:
            reference = JudgmentReference(left.query_id, judgment_id)
            in_left = judgment_id in left_found
            in_right = judgment_id in right_found
            if in_left and in_right:
                both.append(reference)
            elif in_left:
                bm25_only.append(reference)
            elif in_right:
                semantic_only.append(reference)
            else:
                neither.append(reference)
    return UniqueRelevantCoverage(
        limit=limit,
        ground_truth_count=len(both)
        + len(bm25_only)
        + len(semantic_only)
        + len(neither),
        found_by_both_count=len(both),
        bm25_only_count=len(bm25_only),
        semantic_only_count=len(semantic_only),
        neither_count=len(neither),
        found_by_both=tuple(both),
        bm25_only=tuple(bm25_only),
        semantic_only=tuple(semantic_only),
        neither=tuple(neither),
    )


def _oracle_union_recall(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
) -> tuple[OracleUnionRecall, ...]:
    rows: list[OracleUnionRecall] = []
    for limit in _OVERLAP_LIMITS:
        found_count = 0
        relevant_count = 0
        query_recalls: list[float] = []
        for left, right in zip(
            bm25.query_results,
            semantic.query_results,
            strict=True,
        ):
            if left.category == "negative":
                continue
            ground_truth = set(_ground_truth_judgments(left, right))
            found = (
                _matched_judgments(left, limit) | _matched_judgments(right, limit)
            ) & ground_truth
            relevant_count += len(ground_truth)
            found_count += len(found)
            query_recalls.append(
                len(found) / len(ground_truth) if ground_truth else 0.0
            )
        rows.append(
            OracleUnionRecall(
                limit=limit,
                positive_query_count=len(query_recalls),
                found_judgment_count=found_count,
                relevant_judgment_count=relevant_count,
                recall=(found_count / relevant_count if relevant_count else 0.0),
                mean_query_recall=(fmean(query_recalls) if query_recalls else 0.0),
            )
        )
    return tuple(rows)


def _fusion_recovery(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
    hybrid: EvaluationRun,
    limit: int,
) -> FusionRecoverySummary:
    bm25_only_count = 0
    semantic_only_count = 0
    bm25_recovered = 0
    semantic_recovered = 0
    recovered: list[JudgmentReference] = []
    for lexical, vector, fused in zip(
        bm25.query_results,
        semantic.query_results,
        hybrid.query_results,
        strict=True,
    ):
        lexical_found = _matched_judgments(lexical, limit)
        vector_found = _matched_judgments(vector, limit)
        fused_found = _matched_judgments(fused, limit)
        lexical_only = lexical_found - vector_found
        vector_only = vector_found - lexical_found
        lexical_recovered = lexical_only & fused_found
        vector_recovered = vector_only & fused_found
        bm25_only_count += len(lexical_only)
        semantic_only_count += len(vector_only)
        bm25_recovered += len(lexical_recovered)
        semantic_recovered += len(vector_recovered)
        recovered.extend(
            JudgmentReference(lexical.query_id, judgment_id)
            for judgment_id in sorted(lexical_recovered | vector_recovered)
        )
    component_only_count = bm25_only_count + semantic_only_count
    recovered_count = bm25_recovered + semantic_recovered
    return FusionRecoverySummary(
        limit=limit,
        component_only_count=component_only_count,
        recovered_count=recovered_count,
        recovery_rate=(
            recovered_count / component_only_count if component_only_count else 0.0
        ),
        bm25_only_count=bm25_only_count,
        bm25_only_recovered_count=bm25_recovered,
        semantic_only_count=semantic_only_count,
        semantic_only_recovered_count=semantic_recovered,
        recovered_judgments=tuple(recovered),
    )


def _hybrid_contribution_summary(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
    hybrid: EvaluationRun,
    limit: int,
) -> HybridContributionSummary:
    items: list[HybridContributionItem] = []
    exclusive_component_candidates: set[tuple[str, str, str | None, str]] = set()
    retained_exclusive_candidates: set[tuple[str, str, str | None, str]] = set()
    for lexical, vector, fused in zip(
        bm25.query_results,
        semantic.query_results,
        hybrid.query_results,
        strict=True,
    ):
        lexical_ids = _item_identities(lexical, limit)
        vector_ids = _item_identities(vector, limit)
        exclusive = lexical_ids ^ vector_ids
        exclusive_component_candidates.update(
            (lexical.query_id, *identity) for identity in exclusive
        )
        fused_ids = _item_identities(fused, limit)
        retained_exclusive_candidates.update(
            (fused.query_id, *identity) for identity in fused_ids & exclusive
        )
        items.extend(
            _hybrid_contribution_item(fused.query_id, item)
            for item in _top_items(fused, limit)
        )

    dual = [item for item in items if item.source_classification == "dual"]
    bm25_only = [item for item in items if item.source_classification == "bm25_only"]
    semantic_only = [
        item for item in items if item.source_classification == "semantic_only"
    ]
    unattributed = [
        item for item in items if item.source_classification == "unattributed"
    ]
    promotions = sum(
        item.bm25_rank is not None
        and item.semantic_rank is not None
        and item.rank < min(item.bm25_rank, item.semantic_rank)
        for item in dual
    )
    count = len(items)
    return HybridContributionSummary(
        limit=limit,
        final_item_count=count,
        dual_source_count=len(dual),
        dual_source_fraction=_fraction(len(dual), count),
        bm25_only_count=len(bm25_only),
        bm25_only_fraction=_fraction(len(bm25_only), count),
        semantic_only_count=len(semantic_only),
        semantic_only_fraction=_fraction(len(semantic_only), count),
        unattributed_count=len(unattributed),
        unattributed_fraction=_fraction(len(unattributed), count),
        average_bm25_contribution=_optional_mean(
            item.bm25_contribution for item in items
        ),
        average_semantic_contribution=_optional_mean(
            item.semantic_contribution for item in items
        ),
        average_bm25_score=_optional_mean(item.bm25_score for item in items),
        average_cosine_similarity=_optional_mean(
            item.cosine_similarity for item in items
        ),
        dual_source_promotion_count=promotions,
        dual_source_promotion_rate=_fraction(promotions, len(dual)),
        single_source_retention_scope="standalone_component_top_k",
        single_source_candidate_count=len(exclusive_component_candidates),
        single_source_retained_count=len(retained_exclusive_candidates),
        single_source_retention_rate=_fraction(
            len(retained_exclusive_candidates),
            len(exclusive_component_candidates),
        ),
        items=tuple(items),
    )


def _hybrid_contribution_item(
    query_id: str,
    item: RetrievedItem,
) -> HybridContributionItem:
    metadata = item.strategy_metadata
    bm25_rank = _metadata_rank(metadata, "bm25_rank")
    semantic_rank = _metadata_rank(metadata, "semantic_rank")
    if bm25_rank is not None and semantic_rank is not None:
        classification = "dual"
    elif bm25_rank is not None:
        classification = "bm25_only"
    elif semantic_rank is not None:
        classification = "semantic_only"
    else:
        classification = "unattributed"
    bm25_contribution = _metadata_float(metadata, "bm25_contribution")
    semantic_contribution = _metadata_float(metadata, "semantic_contribution")
    contribution_sum = (
        (bm25_contribution or 0.0) + (semantic_contribution or 0.0)
        if bm25_contribution is not None or semantic_contribution is not None
        else None
    )
    return HybridContributionItem(
        query_id=query_id,
        rank=item.rank,
        document_id=item.document_id,
        section_id=item.section_id,
        content_hash=item.content_hash,
        score=item.score,
        source_classification=classification,
        bm25_rank=bm25_rank,
        semantic_rank=semantic_rank,
        bm25_contribution=bm25_contribution,
        semantic_contribution=semantic_contribution,
        contribution_sum=contribution_sum,
        bm25_score=_metadata_float(metadata, "bm25_score"),
        cosine_similarity=_metadata_float(metadata, "cosine_similarity"),
    )


def _canonical_classifications(
    bm25: EvaluationRun,
    semantic: EvaluationRun,
    hybrid: EvaluationRun,
    limit: int,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    hybrid_wins: list[str] = []
    hybrid_beats_bm25: list[str] = []
    hybrid_beats_semantic: list[str] = []
    bm25_wins: list[str] = []
    semantic_wins: list[str] = []
    regressions: list[str] = []
    all_fail: list[str] = []
    for lexical, vector, fused in zip(
        bm25.query_results,
        semantic.query_results,
        hybrid.query_results,
        strict=True,
    ):
        lexical_quality = _query_quality(lexical, limit)
        vector_quality = _query_quality(vector, limit)
        fused_quality = _query_quality(fused, limit)
        if fused_quality > lexical_quality:
            hybrid_beats_bm25.append(fused.query_id)
        if fused_quality > vector_quality:
            hybrid_beats_semantic.append(fused.query_id)
        if fused_quality > lexical_quality and fused_quality > vector_quality:
            hybrid_wins.append(fused.query_id)
        if lexical_quality > vector_quality and lexical_quality > fused_quality:
            bm25_wins.append(fused.query_id)
        if vector_quality > lexical_quality and vector_quality > fused_quality:
            semantic_wins.append(fused.query_id)
        if (
            _query_succeeded(lexical, limit) or _query_succeeded(vector, limit)
        ) and fused_quality < max(lexical_quality, vector_quality):
            regressions.append(fused.query_id)
        if not any(
            _query_succeeded(result, limit) for result in (lexical, vector, fused)
        ):
            all_fail.append(fused.query_id)
    return (
        tuple(hybrid_wins),
        tuple(hybrid_beats_bm25),
        tuple(hybrid_beats_semantic),
        tuple(bm25_wins),
        tuple(semantic_wins),
        tuple(regressions),
        tuple(all_fail),
    )


def _ground_truth_judgments(*results: QueryEvaluation) -> tuple[str, ...]:
    expected = tuple(
        judgment.judgment_id
        for judgment in results[0].expected_sources
        if judgment.relevance > 0
    )
    if expected:
        return expected
    inferred = {
        judgment_id
        for result in results
        for judgment_id in (
            *result.missed_judgment_ids,
            *(
                item.matched_judgment_id
                for item in result.retrieved_items
                if item.relevance_grade > 0 and item.matched_judgment_id is not None
            ),
        )
        if judgment_id is not None
    }
    return tuple(sorted(inferred))


def _matched_judgments(result: QueryEvaluation, limit: int) -> set[str]:
    return {
        item.matched_judgment_id
        for item in _top_items(result, limit)
        if item.relevance_grade > 0 and item.matched_judgment_id is not None
    }


def _ranked_items(result: QueryEvaluation) -> tuple[RetrievedItem, ...]:
    return tuple(
        sorted(
            result.retrieved_items,
            key=lambda item: (
                item.rank,
                item.document_id,
                item.section_id or "",
                item.content_hash,
            ),
        )
    )


def _top_items(result: QueryEvaluation, limit: int) -> tuple[RetrievedItem, ...]:
    return _ranked_items(result)[:limit]


def _item_identities(
    result: QueryEvaluation,
    limit: int,
) -> set[tuple[str, str | None, str]]:
    return {
        (item.document_id, item.section_id, item.content_hash)
        for item in _top_items(result, limit)
    }


def _metadata_rank(metadata: Mapping[str, object], key: str) -> int | None:
    value = metadata.get(key)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _metadata_float(metadata: Mapping[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _query_hit(result: QueryEvaluation, limit: int) -> float:
    return result.metrics.hit_rate_at[limit]


def _query_recall(result: QueryEvaluation, limit: int) -> float:
    return result.metrics.recall_at[limit]


def _query_mrr(result: QueryEvaluation, _limit: int) -> float:
    return result.metrics.reciprocal_rank


def _query_ndcg(result: QueryEvaluation, limit: int) -> float:
    return result.metrics.ndcg_at[limit]


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


def _append_canonical_diagnostics(
    lines: list[str],
    comparison: MultiEvaluationComparison,
) -> None:
    if comparison.candidate_overlap:
        lines.extend(
            [
                "",
                "## BM25/semantic candidate overlap",
                "",
                (
                    "Overlap uses stable chunk identity `(document_id, section_id, "
                    "content_hash)`, not source URL alone."
                ),
                "",
                "| K | Intersection | Union | Jaccard | Mean query Jaccard |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for overlap in comparison.candidate_overlap:
            lines.append(
                f"| {overlap.limit} | {overlap.intersection_count} | "
                f"{overlap.union_count} | {overlap.jaccard:.4f} | "
                f"{overlap.mean_query_jaccard:.4f} |"
            )

    coverage = comparison.unique_relevant_coverage
    if coverage is not None:
        lines.extend(
            [
                "",
                "## Unique relevant coverage",
                "",
                f"At K={coverage.limit}, across {coverage.ground_truth_count} "
                "ground-truth judgments:",
                "",
                f"- Found by both: {coverage.found_by_both_count}",
                f"- BM25 only: {coverage.bm25_only_count}",
                f"- Semantic only: {coverage.semantic_only_count}",
                f"- Neither: {coverage.neither_count}",
            ]
        )
    if comparison.oracle_union_recall:
        lines.extend(
            [
                "",
                "## Ground-truth oracle union recall",
                "",
                (
                    "This is an optimistic diagnostic over known judgments, not a "
                    "deployable or production fusion policy."
                ),
                "",
                "| K | Found | Relevant | Recall | Mean query recall |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison.oracle_union_recall:
            lines.append(
                f"| {row.limit} | {row.found_judgment_count} | "
                f"{row.relevant_judgment_count} | {row.recall:.4f} | "
                f"{row.mean_query_recall:.4f} |"
            )

    fusion = comparison.fusion_recovery
    if fusion is not None:
        lines.extend(
            [
                "",
                "## Fusion recovery",
                "",
                f"- Component-only relevant judgments at K={fusion.limit}: "
                f"{fusion.component_only_count}",
                f"- Recovered by hybrid: {fusion.recovered_count} "
                f"({fusion.recovery_rate:.2%})",
                f"- BM25-only recovered: {fusion.bm25_only_recovered_count}/"
                f"{fusion.bm25_only_count}",
                f"- Semantic-only recovered: {fusion.semantic_only_recovered_count}/"
                f"{fusion.semantic_only_count}",
            ]
        )

    contributions = comparison.hybrid_contributions
    if contributions is not None:
        lines.extend(
            [
                "",
                "## Hybrid contribution diagnostics",
                "",
                "| Final top-K source class | Count | Fraction |",
                "| --- | ---: | ---: |",
                (
                    f"| Dual source | {contributions.dual_source_count} | "
                    f"{contributions.dual_source_fraction:.2%} |"
                ),
                (
                    f"| BM25 only | {contributions.bm25_only_count} | "
                    f"{contributions.bm25_only_fraction:.2%} |"
                ),
                (
                    f"| Semantic only | {contributions.semantic_only_count} | "
                    f"{contributions.semantic_only_fraction:.2%} |"
                ),
                (
                    f"| Unattributed | {contributions.unattributed_count} | "
                    f"{contributions.unattributed_fraction:.2%} |"
                ),
                "",
                f"- Average BM25 contribution: "
                f"{_value(contributions.average_bm25_contribution)}",
                f"- Average semantic contribution: "
                f"{_value(contributions.average_semantic_contribution)}",
                f"- Dual-source promotion rate: "
                f"{contributions.dual_source_promotion_rate:.2%}",
                f"- Standalone component top-K single-source retention rate: "
                f"{contributions.single_source_retention_rate:.2%}",
            ]
        )

    if comparison.hybrid_contributions is not None:
        lines.extend(
            [
                "",
                "## Canonical query outcomes",
                "",
                f"- Hybrid wins over both: "
                f"{_query_ids(comparison.hybrid_wins_over_both)}",
                f"- Hybrid beats BM25: {_query_ids(comparison.hybrid_beats_bm25)}",
                (
                    f"- Hybrid beats semantic: "
                    f"{_query_ids(comparison.hybrid_beats_semantic)}"
                ),
                f"- BM25-only wins: {_query_ids(comparison.bm25_only_wins)}",
                (f"- Semantic-only wins: {_query_ids(comparison.semantic_only_wins)}"),
                f"- Hybrid regressions: {_query_ids(comparison.hybrid_regressions)}",
                f"- All three fail: {_query_ids(comparison.all_three_fail)}",
            ]
        )


def _append_query_evidence(
    lines: list[str],
    comparison: MultiEvaluationComparison,
) -> None:
    lines.extend(
        [
            "",
            "## Query-level evidence",
            "",
            (
                "| Query | Category | Strategy | First relevant | Recall@5 | "
                "NDCG@5 | Final items | Missed judgments | Failure |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for query in comparison.query_comparisons:
        for strategy in query.strategies:
            final_items = (
                "; ".join(
                    (
                        f"#{item.rank} {item.document_id}/"
                        f"{item.section_id or 'none'}/{item.content_hash} "
                        f"score={item.score:.4f}"
                    )
                    for item in strategy.final_items
                )
                or "none"
            )
            missed = ", ".join(strategy.missed_judgment_ids) or "none"
            lines.append(
                f"| `{query.query_id}` {_escape(query.query)} | "
                f"`{_escape(query.category)}` | "
                f"`{_escape(strategy.strategy_alias)}` | "
                f"{strategy.first_relevant_rank or 'none'} | "
                f"{strategy.recall_at_5:.4f} | {strategy.ndcg_at_5:.4f} | "
                f"{_escape(final_items)} | {_escape(missed)} | "
                f"{_escape(strategy.failure or 'none')} |"
            )


def _json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("multi-comparison reports require finite numbers")
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    raise TypeError(f"unsupported multi-comparison value: {type(value).__name__}")


def _metric_lookup(
    values: tuple[StrategyMetricValue, ...],
) -> dict[str, float | None]:
    return {value.strategy_alias: value.value for value in values}


def _metric_delta(candidate: float | None, baseline: float | None) -> float | None:
    return (
        candidate - baseline if candidate is not None and baseline is not None else None
    )


def _value(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _signed(value: float | None) -> str:
    return f"{value:+.4f}" if value is not None else "n/a"


def _signed_delta(candidate: float | None, baseline: float | None) -> str:
    return (
        _signed(candidate - baseline)
        if candidate is not None and baseline is not None
        else "n/a"
    )


def _query_ids(values: tuple[str, ...]) -> str:
    return ", ".join(f"`{_escape(value)}`" for value in values) if values else "none"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
