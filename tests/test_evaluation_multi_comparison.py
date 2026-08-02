from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest

from crawlforge.evaluation.metrics import query_metric_values, summarize_metrics
from crawlforge.evaluation.models import (
    CategorySummary,
    ContextQualitySummary,
    CorpusStatistics,
    EvaluationQuery,
    EvaluationRun,
    LatencySummary,
    QueryCategory,
    QueryEvaluation,
    RelevanceJudgment,
    RetrievedItem,
)
from crawlforge.evaluation.multi_comparison import (
    MULTI_COMPARISON_SCHEMA_VERSION,
    compare_multiple_evaluation_runs,
    render_multi_comparison_json,
    render_multi_comparison_markdown,
    write_multi_comparison_report,
)

_LIMITS = (1, 3, 5, 10)


def _context_quality() -> ContextQualitySummary:
    return ContextQualitySummary(
        mean_candidate_count=1.0,
        mean_returned_estimated_tokens=10.0,
        relevant_chunks_per_1000_estimated_tokens=100.0,
        irrelevant_estimated_token_ratio=0.0,
        mean_relevant_source_coverage=1.0,
        mean_estimated_context_reduction=0.5,
    )


def _judgment(query_id: str, index: int) -> RelevanceJudgment:
    return RelevanceJudgment(
        judgment_id=f"{query_id}-j{index}",
        document_id=f"doc-{query_id}",
        relevance=1,
        section_id=f"section-{index}",
    )


def _item(
    query_id: str,
    index: int,
    *,
    rank: int,
    content_hash: str,
    matched: bool = True,
    score: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> RetrievedItem:
    return RetrievedItem(
        rank=rank,
        document_id=f"doc-{query_id}",
        url=f"https://example.test/{query_id}/{index}",
        canonical_url=f"https://example.test/{query_id}/{index}",
        title="Document",
        section_id=f"section-{index}",
        heading_path=(f"Section {index}",),
        text="Relevant text",
        score=score,
        estimated_tokens=10,
        source_estimated_tokens=20,
        content_hash=content_hash,
        relevance_grade=int(matched),
        matched_judgment_id=f"{query_id}-j{index}" if matched else None,
        strategy_metadata=metadata or {},
    )


def _query(
    query_id: str,
    items: tuple[RetrievedItem, ...],
    *,
    relevant_count: int = 1,
    category: str = "conceptual",
    failure: str | None = None,
) -> QueryEvaluation:
    judgments = tuple(
        _judgment(query_id, index) for index in range(1, relevant_count + 1)
    )
    query = EvaluationQuery(
        query_id=query_id,
        query=f"query {query_id}",
        category=cast(QueryCategory, category),
        relevant_sources=judgments,
    )
    metrics = query_metric_values(query, items, _LIMITS)
    matched = {
        item.matched_judgment_id
        for item in items
        if item.matched_judgment_id is not None
    }
    relevant_items = tuple(item for item in items if item.relevance_grade > 0)
    returned_tokens = sum(item.estimated_tokens for item in items)
    relevant_tokens = sum(item.estimated_tokens for item in relevant_items)
    return QueryEvaluation(
        query_id=query_id,
        query=query.query,
        category=query.category,
        expected_sources=judgments,
        retrieved_items=items,
        context_item_ranks=tuple(item.rank for item in items),
        context_relevant_chunk_count=len(relevant_items),
        first_relevant_rank=(
            min(item.rank for item in relevant_items) if relevant_items else None
        ),
        missed_judgment_ids=tuple(
            judgment.judgment_id
            for judgment in judgments
            if judgment.judgment_id not in matched
        ),
        metrics=metrics,
        candidate_count=len(items),
        returned_estimated_tokens=returned_tokens,
        relevant_estimated_tokens=relevant_tokens,
        irrelevant_estimated_tokens=returned_tokens - relevant_tokens,
        source_estimated_tokens=returned_tokens * 2,
        estimated_context_reduction=0.5,
        relevant_chunks_per_1000_estimated_tokens=(
            len(relevant_items) * 1000 / returned_tokens if returned_tokens else 0.0
        ),
        irrelevant_estimated_token_ratio=(
            (returned_tokens - relevant_tokens) / returned_tokens
            if returned_tokens
            else 0.0
        ),
        relevant_source_coverage=(
            len(matched) / relevant_count if relevant_count else 0.0
        ),
        latency_samples_ms=(1.0,),
        failure=failure,
    )


def _negative_query(query_id: str) -> QueryEvaluation:
    return _query(query_id, (), relevant_count=0, category="negative")


def _run(strategy: str, queries: tuple[QueryEvaluation, ...]) -> EvaluationRun:
    aggregate = summarize_metrics(queries, _LIMITS)
    category_metrics = tuple(
        CategorySummary(
            category=cast(QueryCategory, category),
            metrics=summarize_metrics(
                tuple(query for query in queries if query.category == category),
                _LIMITS,
            ),
            context_quality=_context_quality(),
        )
        for category in dict.fromkeys(query.category for query in queries)
    )
    return EvaluationRun(
        dataset_name="dataset",
        dataset_version="1.0.0",
        retrieval_strategy=strategy,
        retrieval_configuration={
            "limit_values": list(_LIMITS),
            "token_budget": 3000,
            "model_cache_path": "/Users/private/model-cache",
        },
        chunking_configuration={"target_chars": 1200},
        timestamp="2026-01-01T00:00:00+00:00",
        corpus_statistics=CorpusStatistics(
            document_count=5,
            section_count=6,
            chunk_count=6,
            source_size_bytes=100,
            cleaned_size_bytes=80,
            source_estimated_tokens=25,
            cleaned_estimated_tokens=20,
            indexing_time_ms=1.0,
        ),
        query_results=queries,
        aggregate_metrics=aggregate,
        category_metrics=category_metrics,
        latency=LatencySummary(
            sample_count=len(queries),
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


def _triple() -> tuple[EvaluationRun, EvaluationRun, EvaluationRun]:
    bm25 = _run(
        "bm25-fts5",
        (
            _query(
                "q1",
                (_item("q1", 1, rank=1, content_hash="bm25-chunk"),),
                relevant_count=2,
            ),
            _query("q2", (_item("q2", 1, rank=1, content_hash="q2-bm25"),)),
            _query("q3", ()),
            _query("q4", ()),
            _negative_query("q5"),
        ),
    )
    semantic = _run(
        "semantic-exact-cosine",
        (
            _query(
                "q1",
                (_item("q1", 2, rank=1, content_hash="semantic-chunk"),),
                relevant_count=2,
            ),
            _query("q2", ()),
            _query("q3", (_item("q3", 1, rank=1, content_hash="q3-semantic"),)),
            _query("q4", ()),
            _negative_query("q5"),
        ),
    )
    hybrid = _run(
        "hybrid-additive",
        (
            _query(
                "q1",
                (
                    _item(
                        "q1",
                        1,
                        rank=1,
                        content_hash="bm25-chunk",
                        metadata={
                            "bm25_rank": 2,
                            "semantic_rank": 3,
                            "bm25_contribution": 0.6,
                            "semantic_contribution": 0.4,
                            "bm25_score": 0.9,
                            "cosine_similarity": 0.8,
                        },
                    ),
                    _item(
                        "q1",
                        2,
                        rank=2,
                        content_hash="semantic-chunk",
                        score=0.8,
                        metadata={
                            "bm25_rank": None,
                            "semantic_rank": 1,
                            "bm25_contribution": 0.0,
                            "semantic_contribution": 0.8,
                            "bm25_score": None,
                            "cosine_similarity": 0.8,
                        },
                    ),
                ),
                relevant_count=2,
            ),
            _query("q2", ()),
            _query("q3", ()),
            _query("q4", ()),
            _negative_query("q5"),
        ),
    )
    return bm25, semantic, hybrid


def test_three_run_comparison_is_deterministic_and_has_all_pairs() -> None:
    bm25, semantic, hybrid = _triple()

    first = compare_multiple_evaluation_runs(
        (("bm25", bm25), ("semantic", semantic), ("hybrid", hybrid)),
        bootstrap_samples=100,
        seed=7,
    )
    second = compare_multiple_evaluation_runs(
        (("bm25", bm25), ("semantic", semantic), ("hybrid", hybrid)),
        bootstrap_samples=100,
        seed=7,
    )

    assert first == second
    assert first.schema_version == MULTI_COMPARISON_SCHEMA_VERSION
    assert [strategy.alias for strategy in first.strategies] == [
        "bm25",
        "semantic",
        "hybrid",
    ]
    assert [
        (pair.baseline_alias, pair.candidate_alias)
        for pair in first.pairwise_comparisons
    ] == [
        ("bm25", "semantic"),
        ("bm25", "hybrid"),
        ("semantic", "hybrid"),
    ]
    assert all(len(pair.uncertainty) == 4 for pair in first.pairwise_comparisons)
    assert all(
        interval.bootstrap_samples == 100
        for pair in first.pairwise_comparisons
        for interval in pair.uncertainty
    )
    assert first.hybrid_wins_over_both == ("q1",)
    assert first.bm25_only_wins == ("q2",)
    assert first.semantic_only_wins == ("q3",)
    assert first.hybrid_regressions == ("q2", "q3")
    assert first.all_three_fail == ("q4",)


def test_overlap_uses_chunks_and_coverage_oracle_and_recovery_are_explicit() -> None:
    comparison = compare_multiple_evaluation_runs(
        dict(zip(("bm25", "semantic", "hybrid"), _triple(), strict=True)),
        bootstrap_samples=20,
    )

    q1 = comparison.query_comparisons[0]
    assert q1.strategies[0].final_items[0].document_id == (
        q1.strategies[1].final_items[0].document_id
    )
    overlap_at_1 = next(row for row in q1.bm25_semantic_overlap if row.limit == 1)
    assert overlap_at_1.intersection_count == 0
    assert overlap_at_1.union_count == 2
    assert overlap_at_1.jaccard == 0.0

    coverage = comparison.unique_relevant_coverage
    assert coverage is not None
    assert coverage.ground_truth_count == 5
    assert coverage.found_by_both_count == 0
    assert coverage.bm25_only_count == 2
    assert coverage.semantic_only_count == 2
    assert coverage.neither_count == 1

    oracle_at_5 = next(row for row in comparison.oracle_union_recall if row.limit == 5)
    assert oracle_at_5.found_judgment_count == 4
    assert oracle_at_5.relevant_judgment_count == 5
    assert oracle_at_5.recall == pytest.approx(0.8)

    recovery = comparison.fusion_recovery
    assert recovery is not None
    assert recovery.component_only_count == 4
    assert recovery.recovered_count == 2
    assert recovery.recovery_rate == pytest.approx(0.5)
    assert recovery.bm25_only_recovered_count == 1
    assert recovery.semantic_only_recovered_count == 1


def test_hybrid_contribution_classifications_counts_sums_and_rates() -> None:
    comparison = compare_multiple_evaluation_runs(
        dict(zip(("bm25", "semantic", "hybrid"), _triple(), strict=True)),
        bootstrap_samples=20,
    )

    contributions = comparison.hybrid_contributions
    assert contributions is not None
    assert contributions.final_item_count == 2
    assert contributions.dual_source_count == 1
    assert contributions.bm25_only_count == 0
    assert contributions.semantic_only_count == 1
    assert contributions.unattributed_count == 0
    assert contributions.dual_source_fraction == pytest.approx(0.5)
    assert contributions.semantic_only_fraction == pytest.approx(0.5)
    assert contributions.average_bm25_contribution == pytest.approx(0.3)
    assert contributions.average_semantic_contribution == pytest.approx(0.6)
    assert [item.source_classification for item in contributions.items] == [
        "dual",
        "semantic_only",
    ]
    assert [item.contribution_sum for item in contributions.items] == [
        pytest.approx(1.0),
        pytest.approx(0.8),
    ]
    assert contributions.dual_source_promotion_rate == pytest.approx(1.0)
    assert contributions.single_source_retention_scope == "standalone_component_top_k"
    assert contributions.single_source_candidate_count == 4
    assert contributions.single_source_retained_count == 2
    assert contributions.single_source_retention_rate == pytest.approx(0.5)


def test_json_markdown_and_writer_are_stable_path_free_and_honest(tmp_path) -> None:
    comparison = compare_multiple_evaluation_runs(
        dict(zip(("bm25", "semantic", "hybrid"), _triple(), strict=True)),
        bootstrap_samples=20,
        seed=11,
    )

    json_report = render_multi_comparison_json(comparison)
    markdown = render_multi_comparison_markdown(comparison)
    payload = json.loads(json_report)

    assert payload["schema_version"] == 1
    assert payload["seed"] == 11
    assert "/Users/" not in json_report
    assert "/Users/" not in markdown
    assert "model_cache_path" not in json_report
    assert (
        "| Metric | BM25 | Semantic | Hybrid | Hybrid Δ vs BM25 | "
        "Hybrid Δ vs Semantic |"
    ) in markdown
    assert "| Category | BM25 MRR | Semantic MRR | Hybrid MRR |" in markdown
    assert "`negative`" in markdown
    assert "diagnostic" in markdown
    assert "not a production" in markdown
    assert "## Limitations" in markdown

    json_path = tmp_path / "comparison.json"
    markdown_path = tmp_path / "comparison.md"
    write_multi_comparison_report(comparison, json_path, report_format="json")
    write_multi_comparison_report(comparison, markdown_path, report_format="markdown")
    assert json_path.read_text(encoding="utf-8") == json_report
    assert markdown_path.read_text(encoding="utf-8") == markdown


@pytest.mark.parametrize(
    ("candidate", "error"),
    [
        (lambda run: replace(run, dataset_version="2.0.0"), "dataset versions"),
        (lambda run: replace(run, dataset_signature="b" * 64), "dataset signatures"),
        (
            lambda run: replace(run, chunking_configuration={"target_chars": 999}),
            "chunking configurations",
        ),
        (
            lambda run: replace(
                run,
                corpus_statistics=replace(run.corpus_statistics, chunk_count=7),
            ),
            "chunk counts",
        ),
        (
            lambda run: replace(
                run,
                retrieval_configuration={
                    "limit_values": [1, 3, 5],
                    "token_budget": 3000,
                },
            ),
            "K values",
        ),
        (
            lambda run: replace(
                run,
                retrieval_configuration={
                    "limit_values": list(_LIMITS),
                    "token_budget": 42,
                },
            ),
            "token budgets",
        ),
        (
            lambda run: replace(run, query_results=tuple(reversed(run.query_results))),
            "query order",
        ),
    ],
)
def test_comparison_rejects_shared_run_mismatches(candidate, error: str) -> None:
    bm25, semantic, _ = _triple()

    with pytest.raises(ValueError, match=error):
        compare_multiple_evaluation_runs(
            (("bm25", bm25), ("semantic", candidate(semantic))),
            bootstrap_samples=10,
        )


def test_comparison_rejects_too_few_or_duplicate_aliases() -> None:
    bm25, semantic, _ = _triple()

    with pytest.raises(ValueError, match="at least two"):
        compare_multiple_evaluation_runs((("bm25", bm25),))
    with pytest.raises(ValueError, match="duplicate strategy alias"):
        compare_multiple_evaluation_runs(
            (("same", bm25), ("same", semantic)),
            bootstrap_samples=10,
        )


def test_canonical_diagnostics_require_rankings_through_k_10() -> None:
    bm25, semantic, _ = _triple()
    bm25 = replace(
        bm25,
        retrieval_configuration={"limit_values": [5], "token_budget": 3000},
    )
    semantic = replace(
        semantic,
        retrieval_configuration={"limit_values": [5], "token_budget": 3000},
    )

    with pytest.raises(ValueError, match="require K values 1, 3, 5, and 10"):
        compare_multiple_evaluation_runs(
            (("bm25", bm25), ("semantic", semantic)),
            bootstrap_samples=10,
        )


def test_oracle_k_10_uses_results_beyond_the_top_five() -> None:
    bm25_items = tuple(
        _item(
            "q10",
            index,
            rank=index - 1,
            content_hash=f"irrelevant-{index}",
            matched=False,
        )
        for index in range(2, 11)
    ) + (_item("q10", 1, rank=10, content_hash="relevant-at-ten"),)
    bm25 = _run("bm25-fts5", (_query("q10", bm25_items),))
    semantic = _run("semantic-exact-cosine", (_query("q10", ()),))

    comparison = compare_multiple_evaluation_runs(
        (("bm25", bm25), ("semantic", semantic)),
        bootstrap_samples=10,
    )
    oracle = {row.limit: row for row in comparison.oracle_union_recall}

    assert oracle[5].recall == 0.0
    assert oracle[10].recall == 1.0


def test_focus_limit_other_than_five_is_rejected_without_mislabelling() -> None:
    bm25, semantic, _ = _triple()

    with pytest.raises(ValueError, match="focus_limit=5"):
        compare_multiple_evaluation_runs(
            (("lexical", bm25), ("vector", semantic)),
            focus_limit=3,
        )


def test_generic_two_strategy_comparison_does_not_require_canonical_aliases() -> None:
    bm25, semantic, _ = _triple()

    comparison = compare_multiple_evaluation_runs(
        (("lexical", bm25), ("vector", semantic)),
        bootstrap_samples=10,
        seed=3,
    )
    markdown = render_multi_comparison_markdown(comparison)

    assert len(comparison.pairwise_comparisons) == 1
    assert comparison.candidate_overlap == ()
    assert comparison.unique_relevant_coverage is None
    assert comparison.oracle_union_recall == ()
    assert comparison.fusion_recovery is None
    assert comparison.hybrid_contributions is None
    assert "| Metric | lexical | vector |" in markdown
    assert "BM25/semantic overlap" in " ".join(comparison.warnings)


def test_four_strategy_markdown_keeps_every_generic_metric_column() -> None:
    bm25, semantic, hybrid = _triple()
    comparison = compare_multiple_evaluation_runs(
        (
            ("bm25", bm25),
            ("semantic", semantic),
            ("hybrid", hybrid),
            ("reranker", hybrid),
        ),
        bootstrap_samples=10,
    )

    markdown = render_multi_comparison_markdown(comparison)

    assert "| Metric | bm25 | semantic | hybrid | reranker |" in markdown
    assert (
        "| Category | bm25 MRR | semantic MRR | hybrid MRR | reranker MRR |" in markdown
    )
    assert "Hybrid Δ vs BM25" not in markdown
