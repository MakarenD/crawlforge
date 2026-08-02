"""CLI coverage for strict rank-fused hybrid retrieval."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from semantic_fakes import ConstantEmbeddingProvider

import crawlforge.cli as cli
from crawlforge.context_engine import ContextEngine
from crawlforge.evaluation.dataset import load_dataset
from crawlforge.evaluation.runner import ingest_evaluation_corpus

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "retrieval"


def test_hybrid_cli_parsing_preserves_lexical_defaults() -> None:
    """Hybrid is explicit and its advanced defaults match production config."""
    parser = cli.build_parser()

    lexical = parser.parse_args(["search", "query"])
    hybrid = parser.parse_args(["search", "query", "--strategy", "hybrid"])
    evaluation = parser.parse_args(["evaluate", "run", "--strategy", "hybrid"])
    comparison = parser.parse_args(
        [
            "evaluate",
            "compare",
            "--strategies",
            "bm25,semantic,hybrid",
        ]
    )

    assert lexical.strategy == "bm25"
    assert hybrid.strategy == evaluation.strategy == "hybrid"
    assert comparison.strategies == "bm25,semantic,hybrid"
    for arguments in (hybrid, evaluation, comparison):
        assert arguments.rrf_k == 60
        assert arguments.bm25_weight == 1.0
        assert arguments.semantic_weight == 1.0
        assert arguments.bm25_candidates == 50
        assert arguments.semantic_candidates == 50


def test_hybrid_search_json_exposes_fusion_and_component_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Machine output preserves final rank, component ranks, and provenance."""
    database = tmp_path / "hybrid-search.db"
    asyncio.run(_prepare_index(database, semantic=True))
    provider = ConstantEmbeddingProvider()
    monkeypatch.setattr(cli, "_semantic_provider", lambda _arguments: provider)

    exit_code = cli.main(
        [
            "search",
            "retry timeout backoff",
            "--database",
            str(database),
            "--strategy",
            "hybrid",
            "--limit",
            "3",
            "--token-budget",
            "300",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["retrieval_strategy"] == "hybrid-rrf"
    assert payload["fusion_strategy"] == "reciprocal-rank-fusion"
    assert payload["score_type"] == "rrf_score"
    assert payload["fusion_configuration"] == {
        "bm25_candidate_limit": 50,
        "bm25_weight": 1.0,
        "rrf_k": 60,
        "semantic_candidate_limit": 50,
        "semantic_weight": 1.0,
    }
    assert payload["estimated_tokens"] <= 300
    assert payload["results"]
    first = payload["results"][0]
    assert first["rank"] == 1
    assert isinstance(first["rrf_score"], float)
    assert first["bm25_rank"] is not None or first["semantic_rank"] is not None
    assert first["url"].startswith("https://")
    assert isinstance(first["section"], list)
    assert isinstance(first["estimated_tokens"], int)
    assert provider.close_calls == 1


def test_hybrid_search_requires_prebuilt_semantic_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Search reports the embed command and never builds vectors implicitly."""
    database = tmp_path / "lexical-only.db"
    asyncio.run(_prepare_index(database, semantic=False))
    provider = ConstantEmbeddingProvider()
    monkeypatch.setattr(cli, "_semantic_provider", lambda _arguments: provider)

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "search",
                "retry",
                "--database",
                str(database),
                "--strategy",
                "hybrid",
            ]
        )

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "crawlforge embed" in captured.err
    assert "Traceback" not in captured.err
    assert provider.document_calls == 0
    assert provider.close_calls == 1


def test_hybrid_evaluation_and_three_way_comparison_reuse_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hybrid run and triple comparison exercise production adapters end to end."""
    providers: list[ConstantEmbeddingProvider] = []

    def provider_factory(**_values: object) -> ConstantEmbeddingProvider:
        provider = ConstantEmbeddingProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr(cli, "_semantic_provider_from_values", provider_factory)
    hybrid_output = tmp_path / "hybrid.json"
    hybrid_exit = cli.main(
        [
            "evaluate",
            "run",
            "--strategy",
            "hybrid",
            "--database",
            str(tmp_path / "hybrid.db"),
            "--output",
            str(hybrid_output),
            "--query-id",
            "q001",
            "--repeat-latency",
            "1",
            "--dimension",
            "3",
            "--json",
        ]
    )
    hybrid_summary = json.loads(capsys.readouterr().out)
    hybrid_report = json.loads(hybrid_output.read_text(encoding="utf-8"))

    comparison_output = tmp_path / "triple.json"
    comparison_exit = cli.main(
        [
            "evaluate",
            "compare",
            "--strategies",
            "bm25,semantic,hybrid",
            "--database",
            str(tmp_path / "triple.db"),
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

    assert hybrid_exit == comparison_exit == 0
    assert hybrid_summary["strategy"] == "hybrid-rrf"
    assert hybrid_report["retrieval_configuration"]["rrf_k"] == 60
    assert hybrid_report["retrieval_configuration"]["execution_mode"] == "sequential"
    assert comparison_summary["strategies"] == ["bm25", "semantic", "hybrid"]
    assert [item["alias"] for item in comparison_report["strategies"]] == [
        "bm25",
        "semantic",
        "hybrid",
    ]
    assert comparison_report["hybrid_contributions"] is not None
    assert len(providers) == 2
    assert all(provider.close_calls == 1 for provider in providers)
    assert providers[1].document_calls == 2  # Fifty chunks in two configured batches.


async def _prepare_index(database: Path, *, semantic: bool) -> None:
    dataset = load_dataset(BENCHMARK)
    async with ContextEngine(database) as engine:
        await ingest_evaluation_corpus(engine, dataset)
        if semantic:
            await engine.index_embeddings(ConstantEmbeddingProvider())
