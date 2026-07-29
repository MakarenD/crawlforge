from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from crawlforge.evaluation.comparison import (
    compare_evaluation_runs,
    render_comparison_json,
    render_comparison_markdown,
)
from crawlforge.evaluation.models import (
    CategorySummary,
    ContextQualitySummary,
    CorpusStatistics,
    EvaluationRun,
    LatencySummary,
    MetricSummary,
    QueryCategory,
    QueryEvaluation,
    QueryMetricValues,
    RetrievedItem,
)


def _context_quality() -> ContextQualitySummary:
    return ContextQualitySummary(
        mean_candidate_count=1.0,
        mean_returned_estimated_tokens=10.0,
        relevant_chunks_per_1000_estimated_tokens=100.0,
        irrelevant_estimated_token_ratio=0.0,
        mean_relevant_source_coverage=1.0,
        mean_estimated_context_reduction=0.5,
    )


def _query(
    query_id: str,
    *,
    category: str,
    hit: float,
    recall: float,
    reciprocal_rank: float,
    ndcg: float,
    source: str,
    failure: str | None = None,
) -> QueryEvaluation:
    item = RetrievedItem(
        rank=1,
        document_id="doc",
        url=source,
        canonical_url=source,
        title="Document",
        section_id="section",
        heading_path=("Section",),
        text="Relevant text",
        score=1.0,
        estimated_tokens=10,
        source_estimated_tokens=20,
        content_hash=f"hash-{query_id}-{source}",
        relevance_grade=int(hit > 0),
    )
    metrics = QueryMetricValues(
        hit_rate_at={5: hit},
        precision_at={5: hit / 5},
        recall_at={5: recall},
        average_precision_at={5: reciprocal_rank},
        ndcg_at={5: ndcg},
        reciprocal_rank=reciprocal_rank,
        no_result_correct=(hit == 0 if category == "negative" else None),
    )
    return QueryEvaluation(
        query_id=query_id,
        query=f"query {query_id}",
        category=cast(QueryCategory, category),
        expected_sources=(),
        retrieved_items=(item,),
        context_item_ranks=(1,),
        context_relevant_chunk_count=int(hit > 0),
        first_relevant_rank=1 if hit > 0 else None,
        missed_judgment_ids=(),
        metrics=metrics,
        candidate_count=1,
        returned_estimated_tokens=10,
        relevant_estimated_tokens=10 if hit > 0 else 0,
        irrelevant_estimated_tokens=0 if hit > 0 else 10,
        source_estimated_tokens=20,
        estimated_context_reduction=0.5,
        relevant_chunks_per_1000_estimated_tokens=(100.0 if hit > 0 else 0.0),
        irrelevant_estimated_token_ratio=(0.0 if hit > 0 else 1.0),
        relevant_source_coverage=hit,
        latency_samples_ms=(1.0,),
        failure=failure,
    )


def _run(
    strategy: str,
    queries: tuple[QueryEvaluation, ...],
    *,
    mrr: float,
    hit: float,
    recall: float,
    ndcg: float,
) -> EvaluationRun:
    metrics = MetricSummary(
        query_count=len(queries),
        positive_query_count=len(queries),
        negative_query_count=0,
        failed_query_count=sum(query.failure is not None for query in queries),
        hit_rate_at={5: hit},
        precision_at={5: hit / 5},
        recall_at={5: recall},
        map_at={5: mrr},
        ndcg_at={5: ndcg},
        mrr=mrr,
        no_result_accuracy=None,
    )
    return EvaluationRun(
        dataset_name="dataset",
        dataset_version="1.0.0",
        retrieval_strategy=strategy,
        retrieval_configuration={
            "limit_values": [5],
            "token_budget": 3000,
        },
        chunking_configuration={"target_chars": 1200},
        timestamp="2026-01-01T00:00:00+00:00",
        corpus_statistics=CorpusStatistics(
            document_count=1,
            section_count=1,
            chunk_count=1,
            source_size_bytes=100,
            cleaned_size_bytes=80,
            source_estimated_tokens=25,
            cleaned_estimated_tokens=20,
            indexing_time_ms=1.0,
        ),
        query_results=queries,
        aggregate_metrics=metrics,
        category_metrics=(
            CategorySummary(
                category="conceptual",
                metrics=metrics,
                context_quality=_context_quality(),
            ),
        ),
        latency=LatencySummary(
            sample_count=2,
            repeat_count=1,
            warmup_count=0,
            mean_ms=1.0,
            median_ms=1.0,
            p95_ms=1.0,
            maximum_ms=1.0,
        ),
        context_quality=_context_quality(),
        worst_queries=(),
        failures=tuple(
            f"{query.query_id}: {query.failure}"
            for query in queries
            if query.failure is not None
        ),
        warnings=(),
        dataset_signature="a" * 64,
    )


def test_paired_comparison_deltas_and_bootstrap_are_deterministic() -> None:
    bm25_queries = (
        _query(
            "q001",
            category="conceptual",
            hit=1.0,
            recall=0.5,
            reciprocal_rank=0.5,
            ndcg=0.6,
            source="https://example.test/a",
        ),
        _query(
            "q002",
            category="conceptual",
            hit=0.0,
            recall=0.0,
            reciprocal_rank=0.0,
            ndcg=0.0,
            source="https://example.test/b",
        ),
    )
    semantic_queries = (
        _query(
            "q001",
            category="conceptual",
            hit=1.0,
            recall=1.0,
            reciprocal_rank=1.0,
            ndcg=1.0,
            source="https://example.test/c",
        ),
        _query(
            "q002",
            category="conceptual",
            hit=1.0,
            recall=0.5,
            reciprocal_rank=0.5,
            ndcg=0.6,
            source="https://example.test/b",
        ),
    )
    bm25 = _run(
        "bm25-fts5",
        bm25_queries,
        mrr=0.25,
        hit=0.5,
        recall=0.25,
        ndcg=0.3,
    )
    semantic = _run(
        "semantic-exact-cosine",
        semantic_queries,
        mrr=0.75,
        hit=1.0,
        recall=0.75,
        ndcg=0.8,
    )

    first = compare_evaluation_runs(
        bm25,
        semantic,
        bootstrap_samples=100,
        seed=7,
    )
    second = compare_evaluation_runs(
        bm25,
        semantic,
        bootstrap_samples=100,
        seed=7,
    )

    assert first == second
    metrics = {metric.metric: metric for metric in first.metrics}
    assert metrics["MRR"].delta == pytest.approx(0.5)
    assert metrics["Recall@5"].delta == pytest.approx(0.5)
    assert first.semantic_wins == ("q001", "q002")
    assert first.bm25_wins == ()
    assert first.both_succeed == ("q001",)
    assert first.query_comparisons[0].bm25_only_sources == ("https://example.test/a",)
    assert first.query_comparisons[0].semantic_only_sources == (
        "https://example.test/c",
    )
    assert all(interval.bootstrap_samples == 100 for interval in first.uncertainty)
    assert "## Selected query analysis" in render_comparison_markdown(first)


def test_comparison_rejects_dataset_or_pairing_mismatch() -> None:
    query = _query(
        "q001",
        category="conceptual",
        hit=1.0,
        recall=1.0,
        reciprocal_rank=1.0,
        ndcg=1.0,
        source="https://example.test/a",
    )
    bm25 = _run("bm25-fts5", (query,), mrr=1.0, hit=1.0, recall=1.0, ndcg=1.0)
    semantic = _run(
        "semantic-exact-cosine",
        (query,),
        mrr=1.0,
        hit=1.0,
        recall=1.0,
        ndcg=1.0,
    )

    with pytest.raises(ValueError, match="dataset signatures"):
        compare_evaluation_runs(
            bm25,
            replace(semantic, dataset_signature="b" * 64),
        )
    with pytest.raises(ValueError, match="query order"):
        compare_evaluation_runs(
            bm25,
            replace(semantic, query_results=()),
        )


def test_comparison_reports_regressions_failures_and_no_local_paths() -> None:
    bm25_query = _query(
        "q031",
        category="conceptual",
        hit=1.0,
        recall=1.0,
        reciprocal_rank=1.0,
        ndcg=1.0,
        source="https://example.test/a",
    )
    semantic_query = _query(
        "q031",
        category="conceptual",
        hit=0.0,
        recall=0.0,
        reciprocal_rank=0.0,
        ndcg=0.0,
        source="https://example.test/b",
        failure="RuntimeError",
    )
    bm25 = _run(
        "bm25-fts5",
        (bm25_query,),
        mrr=1.0,
        hit=1.0,
        recall=1.0,
        ndcg=1.0,
    )
    semantic = _run(
        "semantic-exact-cosine",
        (semantic_query,),
        mrr=0.0,
        hit=0.0,
        recall=0.0,
        ndcg=0.0,
    )

    comparison = compare_evaluation_runs(
        bm25,
        semantic,
        bootstrap_samples=10,
    )
    json_report = render_comparison_json(comparison)
    markdown = render_comparison_markdown(comparison)

    assert comparison.bm25_wins == ("q031",)
    assert comparison.semantic_regressions == ("q031",)
    assert all(interval.paired_query_count == 0 for interval in comparison.uncertainty)
    assert all(interval.mean_delta is None for interval in comparison.uncertainty)
    assert '"semantic_failure": "RuntimeError"' in json_report
    assert "q031" in markdown
    assert "/Users/" not in json_report
    assert "/Users/" not in markdown
