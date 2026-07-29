"""CLI tests for deterministic retrieval evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "retrieval"


def test_evaluate_subcommands_preserve_legacy_help() -> None:
    """Nested evaluation commands do not replace existing root CLI flags."""
    root = _run("--help")
    evaluate = _run("evaluate", "--help")
    run = _run("evaluate", "run", "--help")
    validate = _run("evaluate", "validate", "--help")

    assert root.returncode == 0
    assert "--urls" in root.stdout
    assert "index" in root.stdout
    assert "search" in root.stdout
    assert "evaluate" in root.stdout
    assert evaluate.returncode == run.returncode == validate.returncode == 0
    assert "{run,validate}" in evaluate.stdout
    assert "--limit-values" in run.stdout
    assert "--dataset" in validate.stdout


def test_validate_outputs_machine_readable_dataset_summary() -> None:
    """Validation reports deterministic counts without touching a database."""
    result = _run(
        "evaluate",
        "validate",
        "--dataset",
        str(BENCHMARK),
        "--json",
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["documents"] == 10
    assert payload["sections"] == 40
    assert payload["queries"] == 64
    assert set(payload["categories"].values()) == {8}


def test_bundled_default_dataset_works_outside_repository_cwd(
    tmp_path: Path,
) -> None:
    """The default baseline is independent of the caller's current directory."""
    validation = _run("evaluate", "validate", "--json", cwd=tmp_path)
    output = tmp_path / "default.json"
    evaluation = _run(
        "evaluate",
        "run",
        "--database",
        str(tmp_path / "default.db"),
        "--output",
        str(output),
        "--query-id",
        "q001",
        "--repeat-latency",
        "1",
        "--json",
        cwd=tmp_path,
    )

    assert validation.returncode == 0
    assert json.loads(validation.stdout)["queries"] == 64
    assert evaluation.returncode == 0
    assert json.loads(evaluation.stdout)["queries"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["dataset_version"] == "1.0.0"


def test_run_uses_clean_index_and_writes_json_report(tmp_path: Path) -> None:
    """Evaluation replaces only its selected database and keeps JSON stdout clean."""
    database = tmp_path / "evaluation.db"
    database.write_bytes(b"old non-sqlite content")
    sidecars = tuple(
        Path(f"{database}{suffix}") for suffix in ("-wal", "-shm", "-journal")
    )
    for sidecar in sidecars:
        sidecar.write_bytes(b"stale sqlite state")
    neighbor = tmp_path / "evaluation.db-backup"
    neighbor.write_bytes(b"unrelated")
    output = tmp_path / "baseline.json"

    result = _run(
        "evaluate",
        "run",
        "--dataset",
        str(BENCHMARK),
        "--database",
        str(database),
        "--output",
        str(output),
        "--format",
        "json",
        "--query-id",
        "q001",
        "--query-id",
        "q009",
        "--query-id",
        "q057",
        "--repeat-latency",
        "1",
        "--json",
    )

    assert result.returncode == 0
    summary = json.loads(result.stdout)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert summary["queries"] == 3
    assert summary["strategy"] == "bm25-fts5"
    assert report["dataset_name"] == "crawlforge-retrieval-baseline"
    assert report["retrieval_strategy"] == "bm25-fts5"
    assert len(report["query_results"]) == 3
    assert str(ROOT) not in output.read_text(encoding="utf-8")
    assert "Traceback" not in result.stderr
    assert all(not sidecar.exists() for sidecar in sidecars)
    assert neighbor.read_bytes() == b"unrelated"


def test_canonical_default_report_requires_complete_baseline_run(
    tmp_path: Path,
) -> None:
    """Filtered and explicitly selected datasets cannot replace the baseline."""
    report = tmp_path / "reports" / "bm25-baseline.json"
    report.parent.mkdir()
    report.write_text("sentinel", encoding="utf-8")

    filtered = _run(
        "evaluate",
        "run",
        "--database",
        str(tmp_path / "filtered.db"),
        "--query-id",
        "q001",
        cwd=tmp_path,
    )
    configured = _run(
        "evaluate",
        "run",
        "--dataset",
        str(BENCHMARK),
        "--database",
        str(tmp_path / "configured.db"),
        cwd=tmp_path,
    )

    assert filtered.returncode == configured.returncode == 2
    assert "require an explicit --output path" in filtered.stderr
    assert "require an explicit --output path" in configured.stderr
    assert report.read_text(encoding="utf-8") == "sentinel"


def test_complete_default_run_writes_canonical_baseline(tmp_path: Path) -> None:
    """The complete bundled dataset retains the convenient canonical output."""
    result = _run(
        "evaluate",
        "run",
        "--database",
        str(tmp_path / "baseline.db"),
        "--json",
        cwd=tmp_path,
    )

    report = tmp_path / "reports" / "bm25-baseline.json"
    assert result.returncode == 0
    assert json.loads(result.stdout)["queries"] == 64
    assert (
        json.loads(report.read_text(encoding="utf-8"))["aggregate_metrics"][
            "query_count"
        ]
        == 64
    )


def test_report_cannot_use_sqlite_sidecar_path(tmp_path: Path) -> None:
    """Report output cannot collide with the selected database namespace."""
    database = tmp_path / "evaluation.db"
    database.write_bytes(b"sentinel database")
    output = Path(f"{database}-journal")

    result = _run(
        "evaluate",
        "run",
        "--database",
        str(database),
        "--output",
        str(output),
        "--query-id",
        "q001",
        "--repeat-latency",
        "1",
    )

    assert result.returncode == 2
    assert "conflicts with evaluation database files" in result.stderr
    assert database.read_bytes() == b"sentinel database"
    assert not output.exists()


def test_invalid_dataset_fails_with_code_two_and_no_traceback(
    tmp_path: Path,
) -> None:
    """Expected validation failures use the established argparse error contract."""
    result = _run(
        "evaluate",
        "validate",
        "--dataset",
        str(tmp_path / "missing"),
        "--json",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "dataset validation failed" in result.stderr
    assert "Traceback" not in result.stderr


def _run(
    *arguments: str,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "crawlforge", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
