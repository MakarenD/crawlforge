"""Smoke tests for the deterministic local context benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.context_benchmark import run_benchmarks  # noqa: E402

EXPECTED_KEYS = {
    "workload",
    "docs",
    "source_bytes",
    "clean_bytes",
    "chunks",
    "indexing_seconds",
    "search_iterations",
    "total_search_milliseconds",
    "average_search_milliseconds",
    "hits",
    "estimated_returned_tokens",
    "estimated_context_reduction",
}


@pytest.mark.asyncio
async def test_small_context_benchmark_reports_pipeline_metrics() -> None:
    """A tiny run reports real processing, FTS indexing, and retrieval metrics."""
    results = await run_benchmarks(
        (2,),
        large_paragraphs=3,
        search_iterations=2,
    )

    assert [result["workload"] for result in results] == [
        "documents_2",
        "large_html",
    ]
    assert [result["docs"] for result in results] == [2, 1]
    for result in results:
        assert set(result) == EXPECTED_KEYS
        assert result["source_bytes"] >= result["clean_bytes"] > 0
        assert result["chunks"] > 0
        assert result["indexing_seconds"] >= 0
        assert result["search_iterations"] == 2
        assert result["total_search_milliseconds"] >= 0
        assert result["average_search_milliseconds"] >= 0
        assert result["hits"] > 0
        assert result["estimated_returned_tokens"] >= 0
        assert 0 <= result["estimated_context_reduction"] <= 1


def test_context_benchmark_cli_emits_only_json() -> None:
    """The command-line demo keeps stdout machine-readable."""
    result = subprocess.run(
        [
            sys.executable,
            "examples/context_benchmark.py",
            "--scales",
            "1",
            "--large-paragraphs",
            "2",
            "--search-iterations",
            "1",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert [entry["workload"] for entry in payload] == [
        "documents_1",
        "large_html",
    ]
