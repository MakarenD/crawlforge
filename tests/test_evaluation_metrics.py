"""Independently check standard retrieval metric formulas."""

from __future__ import annotations

import pytest

from crawlforge.evaluation.metrics import (
    average_precision_at,
    dcg_at,
    ndcg_at,
    precision_at,
    recall_at,
    reciprocal_rank,
    summarize_latency,
)


def test_hand_checked_binary_and_rank_metrics() -> None:
    """A,C relevant in A,B,C,D produces the manually derived metric values."""
    grades = [3, 0, 2, 0]
    matched = ["A", None, "C", None]

    assert precision_at(grades, 1) == 1.0
    assert precision_at(grades, 3) == pytest.approx(2 / 3)
    assert precision_at(grades, 4) == 0.5
    assert recall_at(matched, 2, 1) == 0.5
    assert recall_at(matched, 2, 3) == 1.0
    assert reciprocal_rank(grades) == 1.0
    assert average_precision_at(grades, 2, 4) == pytest.approx(5 / 6)


def test_hand_checked_graded_dcg_and_ndcg() -> None:
    """Grades 3 and 2 at ranks one and three have an explicit expected NDCG."""
    grades = [3, 0, 2, 0]
    ideal_grades = [3, 2]

    assert dcg_at(grades, 4) == pytest.approx(8.5)
    assert dcg_at(ideal_grades, 4) == pytest.approx(8.892789260714373)
    assert ndcg_at(grades, ideal_grades, 4) == pytest.approx(0.95583058934618)


def test_metrics_handle_no_relevant_or_returned_items() -> None:
    """Undefined positive-query cases use documented zero values."""
    assert precision_at([], 5) == 0.0
    assert recall_at([], 0, 5) == 0.0
    assert average_precision_at([], 0, 5) == 0.0
    assert reciprocal_rank([]) == 0.0
    assert ndcg_at([], [], 5) == 0.0


def test_latency_summary_uses_nearest_rank_p95() -> None:
    """Latency aggregation is deterministic and does not define a CI floor."""
    summary = summarize_latency(
        [1.0, 2.0, 3.0, 4.0, 20.0],
        repeat_count=5,
        warmup_count=3,
    )

    assert summary.mean_ms == 6.0
    assert summary.median_ms == 3.0
    assert summary.p95_ms == 20.0
    assert summary.maximum_ms == 20.0
