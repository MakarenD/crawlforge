"""Machine-readable and human-readable retrieval evaluation reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

from crawlforge.evaluation.models import EvaluationRun, QueryEvaluation

ReportFormat = Literal["json", "markdown"]


def evaluation_run_payload(run: EvaluationRun) -> dict[str, object]:
    """Return a JSON-compatible payload containing no local filesystem paths."""
    return cast(dict[str, object], _json_compatible(asdict(run)))


def render_json_report(run: EvaluationRun) -> str:
    """Render a stable-key JSON report with a trailing newline."""
    return (
        json.dumps(
            evaluation_run_payload(run),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def render_markdown_report(run: EvaluationRun) -> str:
    """Render the benchmark configuration, metrics, failures, and interpretation."""
    raw_limits = run.retrieval_configuration.get(
        "limit_values",
        [1, 3, 5, 10],
    )
    if not isinstance(raw_limits, list) or not all(
        isinstance(value, int) for value in raw_limits
    ):
        raise TypeError("limit_values must be an integer list")
    limits = tuple(raw_limits)
    focus_limit = 5 if 5 in limits else max(limits)
    is_semantic = run.retrieval_strategy.startswith("semantic")
    lines = [
        (
            "# CrawlForge Semantic Retrieval Baseline"
            if is_semantic
            else "# CrawlForge BM25 Retrieval Baseline"
        ),
        "",
        "## Dataset",
        "",
        f"- Name: `{run.dataset_name}`",
        f"- Version: `{run.dataset_version}`",
        f"- Signature: `{run.dataset_signature}`",
        f"- Retrieval strategy: `{run.retrieval_strategy}`",
        f"- Queries: {run.aggregate_metrics.query_count}",
        f"- Timestamp: `{run.timestamp}`",
        "",
        "## Corpus statistics",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Documents | {run.corpus_statistics.document_count} |",
        f"| Stable sections | {run.corpus_statistics.section_count} |",
        f"| Indexed chunks | {run.corpus_statistics.chunk_count} |",
        f"| Source bytes | {run.corpus_statistics.source_size_bytes} |",
        f"| Cleaned bytes | {run.corpus_statistics.cleaned_size_bytes} |",
        (
            "| Approximate source tokens | "
            f"{run.corpus_statistics.source_estimated_tokens} |"
        ),
        (
            "| Approximate cleaned tokens | "
            f"{run.corpus_statistics.cleaned_estimated_tokens} |"
        ),
        (
            "| Corpus processing and indexing | "
            f"{run.corpus_statistics.indexing_time_ms:.3f} ms |"
        ),
        "",
        "## Chunking configuration",
        "",
    ]
    for key, value in sorted(run.chunking_configuration.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Retrieval configuration", ""])
    for key, value in sorted(run.retrieval_configuration.items()):
        lines.append(f"- `{key}`: `{_configuration_value(value)}`")

    lines.extend(
        [
            "",
            "## Standard retrieval metrics",
            "",
            (
                "Positive-query metrics exclude the explicitly negative queries. "
                "Negative queries are evaluated separately with no-result accuracy."
            ),
            "",
            "| K | Hit Rate | Precision | Recall | MAP | NDCG |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for limit in limits:
        metrics = run.aggregate_metrics
        lines.append(
            f"| {limit} | {_percent(metrics.hit_rate_at[limit])} | "
            f"{_percent(metrics.precision_at[limit])} | "
            f"{_percent(metrics.recall_at[limit])} | "
            f"{_decimal(metrics.map_at[limit])} | "
            f"{_decimal(metrics.ndcg_at[limit])} |"
        )
    no_result = run.aggregate_metrics.no_result_accuracy
    lines.extend(
        [
            "",
            f"- MRR: `{run.aggregate_metrics.mrr:.4f}`",
            (
                "- No-result accuracy: "
                + (f"`{_percent(no_result)}`" if no_result is not None else "`n/a`")
            ),
            f"- Failed queries: {run.aggregate_metrics.failed_query_count}",
            "",
            "## Metrics by category",
            "",
            (
                f"| Category | Queries | Hit@{focus_limit} | "
                f"Recall@{focus_limit} | MRR | NDCG@{focus_limit} |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in run.category_metrics:
        metrics = summary.metrics
        if summary.category == "negative":
            hit = recall = ndcg = "n/a"
        else:
            hit = _percent(metrics.hit_rate_at[focus_limit])
            recall = _percent(metrics.recall_at[focus_limit])
            ndcg = _decimal(metrics.ndcg_at[focus_limit])
        lines.append(
            f"| `{summary.category}` | {metrics.query_count} | {hit} | "
            f"{recall} | {_decimal(metrics.mrr)} | {ndcg} |"
        )

    strongest = sorted(
        (
            result
            for result in run.query_results
            if result.failure is None and result.category != "negative"
        ),
        key=lambda result: (
            result.metrics.ndcg_at[focus_limit],
            result.metrics.reciprocal_rank,
            -result.irrelevant_estimated_token_ratio,
            result.query_id,
        ),
        reverse=True,
    )[:10]
    by_id = {result.query_id: result for result in run.query_results}
    weakest = [by_id[query_id] for query_id in run.worst_queries if query_id in by_id]
    lines.extend(
        [
            "",
            "## Strongest queries",
            "",
            _query_table(strongest, focus_limit),
            "",
            "## Weakest queries",
            "",
            _query_table(weakest, focus_limit),
            "",
            "## False positives",
            "",
        ]
    )
    false_positives = _false_positive_rows(run.query_results)
    if false_positives:
        lines.extend(
            [
                "| Query | Category | Rank | Retrieved source | Section |",
                "| --- | --- | ---: | --- | --- |",
                *false_positives[:20],
            ]
        )
    else:
        lines.append("No uncredited results were returned in the inspected rankings.")

    lines.extend(["", "## False negatives", ""])
    false_negatives = [
        result
        for result in run.query_results
        if result.category != "negative" and result.missed_judgment_ids
    ]
    if false_negatives:
        lines.extend(
            [
                "| Query | Category | Missed judgments | First relevant rank |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for result in false_negatives:
            first_rank = result.first_relevant_rank or 0
            lines.append(
                f"| `{result.query_id}` {_escape(result.query)} | "
                f"`{result.category}` | "
                f"{_escape(', '.join(result.missed_judgment_ids))} | "
                f"{first_rank or 'none'} |"
            )
    else:
        lines.append("Every positive judgment appeared in the evaluated top-K ranking.")

    latency = run.latency
    context = run.context_quality
    lines.extend(
        [
            "",
            "## Warm-index retrieval latency",
            "",
            (
                "Indexing is excluded. These timings are machine-specific and are "
                "not a portable quality gate."
            ),
            "",
            "| Samples | Repeats/query | Warm-ups | Mean | Median | P95 | Maximum |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {latency.sample_count} | {latency.repeat_count} | "
                f"{latency.warmup_count} | {latency.mean_ms:.3f} ms | "
                f"{latency.median_ms:.3f} ms | {latency.p95_ms:.3f} ms | "
                f"{latency.maximum_ms:.3f} ms |"
            ),
            "",
            "## CrawlForge-specific context efficiency",
            "",
            (
                "These project-specific measurements describe approximate bounded "
                "context, not standardized IR metrics or exact model-token savings."
            ),
            "",
            "| Measure | Value |",
            "| --- | ---: |",
            (
                "| Mean candidates before context selection | "
                f"{context.mean_candidate_count:.3f} |"
            ),
            (
                "| Mean returned estimated tokens | "
                f"{context.mean_returned_estimated_tokens:.3f} |"
            ),
            (
                "| Relevant chunks per 1000 estimated tokens | "
                f"{context.relevant_chunks_per_1000_estimated_tokens:.3f} |"
            ),
            (
                "| Irrelevant estimated-token ratio | "
                f"{_percent(context.irrelevant_estimated_token_ratio)} |"
            ),
            (
                "| Mean relevant-source coverage | "
                f"{_percent(context.mean_relevant_source_coverage)} |"
            ),
            (
                "| Mean estimated context reduction | "
                f"{_percent(context.mean_estimated_context_reduction)} |"
            ),
            "",
            "## Benchmark limitations",
            "",
            (
                "- The corpus and judgments are small, synthetic, and designed for "
                "transparent regression analysis rather than broad external validity."
            ),
            (
                "- Relevance matching uses stable document, canonical source, section, "
                "heading, and optional evidence checks; it does not infer semantics."
            ),
            (
                "- Negative-query abstention is strict because retrieval scores are "
                "not calibrated confidence values."
            ),
            (
                "- Token counts use CrawlForge's deterministic character heuristic and "
                "are not exact for a particular model."
            ),
            (
                "- Retrieval quality does not measure generated-answer correctness, "
                "faithfulness, or usefulness."
            ),
        ]
    )
    for warning in run.warnings:
        lines.append(f"- Warning: {_escape(warning)}")
    for failure in run.failures:
        lines.append(f"- Failure: {_escape(failure)}")
    if is_semantic:
        lines.extend(
            [
                "",
                "## Semantic baseline interpretation",
                "",
                (
                    "This is an isolated exact-cosine baseline. Its effect should be "
                    "interpreted only through the paired BM25 comparison on the same "
                    "dataset and chunks; it does not include fusion, reranking, or a "
                    "negative-query threshold."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Readiness for semantic retrieval",
                "",
                _semantic_conclusion(run, focus_limit),
                "",
            ]
        )
    return "\n".join(lines)


def write_evaluation_report(
    run: EvaluationRun,
    path: str | Path,
    *,
    report_format: ReportFormat,
) -> None:
    """Atomically replace one requested evaluation report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = (
        render_json_report(run)
        if report_format == "json"
        else render_markdown_report(run)
    )
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _query_table(
    results: list[QueryEvaluation],
    focus_limit: int,
) -> str:
    lines = [
        f"| Query | Category | First relevant | Hit@{focus_limit} | "
        f"NDCG@{focus_limit} | Irrelevant token ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        first_rank = result.first_relevant_rank or 0
        lines.append(
            f"| `{result.query_id}` {_escape(result.query)} | "
            f"`{result.category}` | {first_rank or 'none'} | "
            f"{_percent(result.metrics.hit_rate_at[focus_limit])} | "
            f"{_decimal(result.metrics.ndcg_at[focus_limit])} | "
            f"{_percent(result.irrelevant_estimated_token_ratio)} |"
        )
    return "\n".join(lines)


def _false_positive_rows(
    results: tuple[QueryEvaluation, ...],
) -> list[str]:
    rows: list[tuple[tuple[int, float, str, int], str]] = []
    for result in results:
        for item in result.retrieved_items:
            if item.relevance_grade > 0:
                continue
            section = " > ".join(item.heading_path) or "(document)"
            rows.append(
                (
                    (
                        0 if result.category == "negative" else 1,
                        -result.irrelevant_estimated_token_ratio,
                        result.query_id,
                        item.rank,
                    ),
                    f"| `{result.query_id}` {_escape(result.query)} | "
                    f"`{result.category}` | {item.rank} | "
                    f"{_escape(item.canonical_url)} | {_escape(section)} |",
                )
            )
    return [row for _priority, row in sorted(rows)]


def _semantic_conclusion(run: EvaluationRun, focus_limit: int) -> str:
    category_map = {summary.category: summary for summary in run.category_metrics}
    lexical_scores = [
        category_map[category].metrics.hit_rate_at[focus_limit]
        for category in ("exact_term", "code_symbol")
        if category in category_map
    ]
    semantic_scores = [
        category_map[category].metrics.hit_rate_at[focus_limit]
        for category in ("paraphrase", "conceptual")
        if category in category_map
    ]
    lexical_hit = sum(lexical_scores) / len(lexical_scores) if lexical_scores else 0
    semantic_hit = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    misses = [
        result
        for result in run.query_results
        if result.category in {"paraphrase", "conceptual"}
        and result.metrics.hit_rate_at[focus_limit] == 0
    ]
    if misses or semantic_hit + 0.1 < lexical_hit:
        examples = ", ".join(result.query_id for result in misses[:5]) or "none"
        return (
            f"At Hit@{focus_limit}, exact-term/code-symbol queries average "
            f"{_percent(lexical_hit)}, while paraphrase/conceptual queries average "
            f"{_percent(semantic_hit)}. Semantic retrieval is justified specifically "
            "for vocabulary-mismatch cases where the intended mechanism is described "
            f"without its indexed terms. Observed examples: {examples}. A future "
            "comparison should keep this dataset and add vector/hybrid strategies "
            "without changing the judgments."
        )
    return (
        f"Paraphrase/conceptual Hit@{focus_limit} ({_percent(semantic_hit)}) does not "
        f"materially trail exact-term/code-symbol Hit@{focus_limit} "
        f"({_percent(lexical_hit)}) on this corpus. This benchmark alone therefore "
        "does not yet justify semantic retrieval; expand independently authored "
        "vocabulary-mismatch queries before adding that complexity."
    )


def _json_compatible(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _configuration_value(value: object) -> str:
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _decimal(value: float) -> str:
    return f"{value:.4f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
