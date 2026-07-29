"""CLI tests for deterministic retrieval evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from semantic_fakes import ConstantEmbeddingProvider

import crawlforge.cli as cli
from crawlforge.semantic_models import SemanticDependencyError

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
    assert "{run,compare,validate}" in evaluate.stdout
    assert "--limit-values" in run.stdout
    assert "--dataset" in validate.stdout


def test_semantic_cli_defaults_are_explicit_opt_in() -> None:
    """Existing search and evaluation defaults remain lexical."""
    parser = cli.build_parser()

    search = parser.parse_args(["search", "query"])
    evaluation = parser.parse_args(["evaluate", "run"])

    assert search.strategy == "bm25"
    assert evaluation.strategy == "bm25"


def test_missing_semantic_extra_has_actionable_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unavailable optional runtime exits cleanly without a traceback."""
    database = tmp_path / "index.db"
    database.touch()

    def missing_provider(_arguments: object) -> ConstantEmbeddingProvider:
        raise SemanticDependencyError(
            "Semantic retrieval requires the 'semantic' extra:\n"
            'pip install "crawlforge[semantic]"'
        )

    monkeypatch.setattr(cli, "_semantic_provider", missing_provider)

    with pytest.raises(SystemExit) as error:
        cli.main(["embed", "--database", str(database)])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "crawlforge[semantic]" in captured.err
    assert "Traceback" not in captured.err


def test_semantic_evaluation_and_paired_comparison_use_public_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Controlled vectors exercise semantic evaluation and paired reporting."""
    providers: list[ConstantEmbeddingProvider] = []

    def provider_factory(**_values: object) -> ConstantEmbeddingProvider:
        provider = ConstantEmbeddingProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(cli, "_semantic_provider_from_values", provider_factory)
    semantic_output = tmp_path / "semantic.json"
    semantic_exit = cli.main(
        [
            "evaluate",
            "run",
            "--strategy",
            "semantic",
            "--database",
            str(tmp_path / "semantic.db"),
            "--output",
            str(semantic_output),
            "--query-id",
            "q001",
            "--repeat-latency",
            "1",
            "--dimension",
            "3",
            "--json",
        ]
    )
    semantic_summary = json.loads(capsys.readouterr().out)
    semantic_report = json.loads(semantic_output.read_text(encoding="utf-8"))

    comparison_output = tmp_path / "comparison.json"
    comparison_exit = cli.main(
        [
            "evaluate",
            "compare",
            "--database",
            str(tmp_path / "comparison.db"),
            "--output",
            str(comparison_output),
            "--format",
            "json",
            "--query-id",
            "q001",
            "--repeat-latency",
            "1",
            "--bootstrap-samples",
            "10",
            "--dimension",
            "3",
            "--json",
        ]
    )
    comparison_summary = json.loads(capsys.readouterr().out)
    comparison_report = json.loads(comparison_output.read_text(encoding="utf-8"))

    assert semantic_exit == comparison_exit == 0
    assert semantic_summary["strategy"] == "semantic-exact-cosine"
    assert semantic_report["dataset_signature"]
    assert semantic_report["retrieval_configuration"]["model_id"] == "test/constant"
    assert comparison_summary["signature"] == semantic_report["dataset_signature"]
    assert comparison_report["baseline_strategy"] == "bm25-fts5"
    assert comparison_report["candidate_strategy"] == "semantic-exact-cosine"
    assert len(comparison_report["query_comparisons"]) == 1
    assert len(providers) == 2
    assert all(provider.close_calls == 1 for provider in providers)


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
