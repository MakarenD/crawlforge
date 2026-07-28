"""Smoke tests for the deterministic performance benchmark."""

from __future__ import annotations

import json
import subprocess
import sys


def test_local_benchmark_reports_time_memory_and_throughput() -> None:
    """The benchmark measures both implementations without timing assertions."""
    result = subprocess.run(
        [
            sys.executable,
            "examples/performance_benchmark.py",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["pages"] == 20
    assert payload[0]["synchronous_seconds"] > 0
    assert payload[0]["asynchronous_seconds"] > 0
    assert payload[0]["asynchronous_peak_memory_mib"] > 0
    assert payload[0]["asynchronous_pages_per_second"] > 0
