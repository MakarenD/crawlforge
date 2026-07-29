"""Dataset, relevance, runner, reporting, and real BM25 evaluation tests."""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from crawlforge.context_engine import ContextEngine
from crawlforge.evaluation.dataset import (
    DatasetValidationError,
    filter_dataset,
    load_dataset,
    validate_dataset,
)
from crawlforge.evaluation.metrics import query_metric_values
from crawlforge.evaluation.models import (
    CorpusStatistics,
    EvaluationDataset,
    EvaluationDocument,
    EvaluationQuery,
    EvaluationSection,
    RelevanceJudgment,
    RetrievedItem,
)
from crawlforge.evaluation.relevance import (
    judgment_matches,
    match_retrieved_items,
)
from crawlforge.evaluation.reporting import (
    render_json_report,
    render_markdown_report,
)
from crawlforge.evaluation.runner import (
    BM25ContextEngineStrategy,
    RetrievalEvaluationRunner,
    ingest_evaluation_corpus,
)

ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "benchmarks" / "retrieval"


def test_versioned_benchmark_dataset_is_valid_and_balanced() -> None:
    """The checked-in offline dataset has stable counts and category coverage."""
    dataset = load_dataset(BENCHMARK)
    category_counts = {
        category: sum(query.category == category for query in dataset.queries)
        for category in (
            "exact_term",
            "code_symbol",
            "error_lookup",
            "paraphrase",
            "conceptual",
            "ambiguous",
            "multi_relevant",
            "negative",
        )
    }

    assert len(dataset.documents) == 10
    assert sum(len(document.sections) for document in dataset.documents) == 40
    assert len(dataset.queries) == 64
    assert set(category_counts.values()) == {8}


def test_dataset_signature_tracks_exact_frozen_files(tmp_path: Path) -> None:
    """Identical copies match while any source-byte change changes identity."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    shutil.copytree(BENCHMARK, first_root)
    shutil.copytree(BENCHMARK, second_root)

    first = load_dataset(first_root)
    second = load_dataset(second_root)
    assert first.signature == second.signature
    assert len(first.signature) == 64
    assert filter_dataset(first, query_ids=frozenset({"q001"})).signature == (
        first.signature
    )

    document = second_root / second.documents[0].path
    document.write_text(
        document.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    assert load_dataset(second_root).signature != first.signature


def test_dataset_loader_rejects_absolute_document_paths(tmp_path: Path) -> None:
    """Versioned datasets cannot embed machine-specific absolute paths."""
    document = tmp_path / "document.html"
    document.write_text("<h1>Title</h1><h2>Section</h2><p>Text</p>")
    (tmp_path / "manifest.json").write_text(
        (
            '{"schema_version":1,"name":"invalid","version":"1.0.0",'
            '"description":"invalid path","documents":[{"document_id":"doc",'
            f'"path":"{document}","url":"https://benchmark.invalid/doc",'
            '"title":"Title","sections":[{"section_id":"section",'
            '"heading_path":["Title","Section"]}]}]}'
        ),
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        '{"query_id":"q001","query":"absent","category":"negative",'
        '"relevant_sources":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="safe relative path"):
        load_dataset(tmp_path)


def test_relevance_requires_stable_section_and_evidence() -> None:
    """Document matches cannot substitute for a judged section-level match."""
    dataset = _small_dataset()
    judgment = dataset.queries[0].relevant_sources[0]
    correct = _item(
        rank=1,
        document_id="network",
        heading_path=("Network Control", "Concurrency Limit"),
        text="SemaphoreGate bounds simultaneous request concurrency.",
    )
    wrong_section = replace(
        correct,
        heading_path=("Network Control", "Other Section"),
        section_id="other",
    )
    wrong_section_id = replace(correct, section_id="other")
    missing_section = replace(correct, section_id=None)

    assert judgment_matches(dataset, correct, judgment)
    assert not judgment_matches(dataset, wrong_section, judgment)
    assert not judgment_matches(dataset, wrong_section_id, judgment)
    assert not judgment_matches(dataset, missing_section, judgment)

    matched = match_retrieved_items(
        dataset,
        dataset.queries[0],
        (correct, replace(correct, rank=2)),
    )
    assert matched.items[0].relevance_grade == 3
    assert matched.items[0].matched_judgment_id == "q001-j1"
    assert matched.items[1].relevance_grade == 0
    assert matched.matched_judgment_ids == {"q001-j1"}
    metrics = query_metric_values(
        dataset.queries[0],
        matched.items,
        (1, 2),
    )
    assert metrics.recall_at[2] == 1.0
    assert metrics.average_precision_at[2] == 1.0
    assert metrics.ndcg_at[2] == 1.0


@pytest.mark.asyncio
async def test_real_pipeline_evaluates_quality_context_and_reports(
    tmp_path: Path,
) -> None:
    """HTML -> processing -> chunks -> FTS5 -> metrics -> reports stays offline."""
    dataset = _small_dataset()
    async with ContextEngine(tmp_path / "evaluation.db") as engine:
        corpus = await ingest_evaluation_corpus(engine, dataset)
        runner = RetrievalEvaluationRunner(
            dataset=dataset,
            retriever=BM25ContextEngineStrategy(engine, dataset),
            corpus_statistics=corpus,
            retrieval_configuration={"index": "sqlite_fts5"},
            chunking_configuration={
                "target_chars": 1200,
                "max_chars": 1600,
                "overlap_chars": 160,
            },
            clock=lambda: datetime(2026, 1, 2, tzinfo=UTC),
        )
        evaluation = await runner.run(
            limits=(1, 3, 5, 10),
            token_budget=3000,
            repeat_latency=2,
        )

    by_id = {result.query_id: result for result in evaluation.query_results}
    assert corpus.document_count == 3
    assert corpus.chunk_count == 3
    assert by_id["q001"].metrics.hit_rate_at[1] == 1.0
    assert by_id["q002"].metrics.hit_rate_at[3] == 1.0
    assert by_id["q003"].metrics.no_result_correct is True
    assert by_id["q004"].metrics.recall_at[5] == 1.0
    assert {summary.category for summary in evaluation.category_metrics} == {
        "exact_term",
        "paraphrase",
        "multi_relevant",
        "negative",
    }
    assert evaluation.latency.sample_count == 8

    json_report = render_json_report(evaluation)
    markdown_report = render_markdown_report(evaluation)
    assert json_report == render_json_report(evaluation)
    assert markdown_report == render_markdown_report(evaluation)
    json.loads(
        json_report,
        parse_constant=lambda value: pytest.fail(
            f"non-standard JSON constant: {value}"
        ),
    )
    assert str(tmp_path) not in json_report
    assert str(tmp_path) not in markdown_report
    assert '"retrieval_strategy": "bm25-fts5"' in json_report
    assert "## False positives" in markdown_report
    assert "## Readiness for semantic retrieval" in markdown_report


@pytest.mark.asyncio
async def test_context_quality_rematches_selected_smaller_chunk() -> None:
    """A later fitting chunk retains relevance when an earlier match is too large."""
    complete = _small_dataset()
    dataset = replace(complete, queries=(complete.queries[0],))
    first = replace(
        _item(
            rank=1,
            document_id="network",
            heading_path=("Network Control", "Concurrency Limit"),
            text="SemaphoreGate bounds simultaneous request concurrency.",
        ),
        estimated_tokens=20,
    )
    second = replace(first, rank=2, estimated_tokens=5, content_hash="network-2")

    class RankedStrategy:
        name = "ranked"

        async def search(
            self,
            _query: str,
            *,
            limit: int,
        ) -> tuple[RetrievedItem, ...]:
            assert limit == 2
            return (first, second)

    runner = RetrievalEvaluationRunner(
        dataset=dataset,
        retriever=RankedStrategy(),
        corpus_statistics=_empty_corpus_statistics(),
    )
    evaluation = await runner.run(
        limits=(1, 2),
        token_budget=5,
        repeat_latency=1,
    )

    result = evaluation.query_results[0]
    assert result.context_item_ranks == (2,)
    assert result.context_relevant_chunk_count == 1
    assert result.relevant_estimated_tokens == 5
    assert result.irrelevant_estimated_tokens == 0
    assert result.irrelevant_estimated_token_ratio == 0.0
    assert result.relevant_source_coverage == 1.0
    assert evaluation.context_quality.relevant_chunks_per_1000_estimated_tokens == 200


@pytest.mark.asyncio
async def test_versioned_benchmark_ci_subset_has_stable_quality_floors(
    tmp_path: Path,
) -> None:
    """A category-balanced subset catches obvious BM25 pipeline regressions."""
    complete = load_dataset(BENCHMARK)
    dataset = filter_dataset(
        complete,
        query_ids=frozenset(
            {
                "q001",
                "q009",
                "q017",
                "q025",
                "q033",
                "q041",
                "q049",
                "q057",
            }
        ),
    )
    async with ContextEngine(tmp_path / "benchmark.db") as engine:
        corpus = await ingest_evaluation_corpus(engine, dataset)
        runner = RetrievalEvaluationRunner(
            dataset=dataset,
            retriever=BM25ContextEngineStrategy(engine, dataset),
            corpus_statistics=corpus,
        )
        evaluation = await runner.run(
            limits=(1, 3, 5, 10),
            repeat_latency=1,
        )

    by_id = {result.query_id: result for result in evaluation.query_results}
    assert by_id["q001"].metrics.hit_rate_at[5] == 1.0
    assert by_id["q009"].metrics.hit_rate_at[5] == 1.0
    assert evaluation.aggregate_metrics.mrr >= 0.35
    assert evaluation.failures == ()


@pytest.mark.asyncio
async def test_runner_propagates_cancellation() -> None:
    """Evaluation never converts caller cancellation into a query failure."""

    class CancelledStrategy:
        name = "cancelled"

        async def search(
            self,
            _query: str,
            *,
            limit: int,
        ) -> tuple[RetrievedItem, ...]:
            assert limit == 10
            raise asyncio.CancelledError

    runner = RetrievalEvaluationRunner(
        dataset=_small_dataset(),
        retriever=CancelledStrategy(),
        corpus_statistics=_empty_corpus_statistics(),
    )

    with pytest.raises(asyncio.CancelledError):
        await runner.run(repeat_latency=1)


def test_validator_rejects_unknown_sections() -> None:
    """Every section-level judgment must reference the manifest inventory."""
    dataset = _small_dataset()
    query = dataset.queries[0]
    invalid_judgment = replace(
        query.relevant_sources[0],
        section_id="missing-section",
    )
    invalid = replace(
        dataset,
        queries=(
            replace(query, relevant_sources=(invalid_judgment,)),
            *dataset.queries[1:],
        ),
    )

    with pytest.raises(DatasetValidationError, match="unknown section_id"):
        validate_dataset(invalid)


def test_validator_rejects_duplicate_relevance_target() -> None:
    """One query cannot grade the same stable section more than once."""
    dataset = _small_dataset()
    query = dataset.queries[0]
    duplicate = replace(
        query.relevant_sources[0],
        judgment_id="q001-j2",
        section_id=None,
    )
    invalid = replace(
        dataset,
        queries=(
            replace(
                query,
                relevant_sources=(*query.relevant_sources, duplicate),
            ),
            *dataset.queries[1:],
        ),
    )

    with pytest.raises(DatasetValidationError, match="duplicate relevance target"):
        validate_dataset(invalid)


def test_validator_rejects_duplicate_heading_path() -> None:
    """Heading paths must map to exactly one stable section per document."""
    dataset = _small_dataset()
    document = dataset.documents[0]
    duplicate = EvaluationSection(
        section_id="duplicate-section",
        heading_path=document.sections[0].heading_path,
    )
    invalid_document = replace(
        document,
        sections=(*document.sections, duplicate),
        content=document.content.replace(
            "</body>",
            '<section id="duplicate-section"></section></body>',
        ),
    )
    invalid = replace(
        dataset,
        documents=(invalid_document, *dataset.documents[1:]),
    )

    with pytest.raises(DatasetValidationError, match="duplicate heading_path"):
        validate_dataset(invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    (
        "nan-score",
        "infinite-score",
        "negative-tokens",
        "negative-source-tokens",
        "empty-provenance",
    ),
)
async def test_runner_rejects_invalid_strategy_items(
    case: str,
) -> None:
    """Invalid strategy provenance and numeric values become explicit failures."""
    valid_item = _item(
        rank=1,
        document_id="network",
        heading_path=("Network Control", "Concurrency Limit"),
        text="SemaphoreGate bounds simultaneous request concurrency.",
    )
    invalid_item = {
        "nan-score": replace(valid_item, score=float("nan")),
        "infinite-score": replace(valid_item, score=float("inf")),
        "negative-tokens": replace(valid_item, estimated_tokens=-1),
        "negative-source-tokens": replace(
            valid_item,
            source_estimated_tokens=-1,
        ),
        "empty-provenance": replace(
            valid_item,
            document_id="",
            content_hash="",
        ),
    }[case]

    class InvalidStrategy:
        name = "invalid"

        async def search(
            self,
            _query: str,
            *,
            limit: int,
        ) -> tuple[RetrievedItem, ...]:
            assert limit == 10
            return (invalid_item,)

    runner = RetrievalEvaluationRunner(
        dataset=_small_dataset(),
        retriever=InvalidStrategy(),
        corpus_statistics=_empty_corpus_statistics(),
    )

    evaluation = await runner.run(repeat_latency=1)

    assert len(evaluation.failures) == len(evaluation.query_results)
    assert all(result.failure == "ValueError" for result in evaluation.query_results)


def _small_dataset() -> EvaluationDataset:
    documents = (
        _document(
            "network",
            "Network Control",
            "Concurrency Limit",
            "concurrency-limit",
            "SemaphoreGate bounds simultaneous request concurrency with a semaphore.",
        ),
        _document(
            "timeouts",
            "HTTP Timing",
            "Timeout Handling",
            "timeout-handling",
            "Connect timeout limits socket setup. Read timeout limits a stalled body.",
        ),
        _document(
            "retries",
            "Retry Control",
            "Backoff Handling",
            "backoff-handling",
            "Exponential backoff delays transient retries after temporary failures.",
        ),
    )
    queries = (
        EvaluationQuery(
            query_id="q001",
            query="SemaphoreGate",
            category="exact_term",
            relevant_sources=(
                RelevanceJudgment(
                    judgment_id="q001-j1",
                    document_id="network",
                    section_id="concurrency-limit",
                    heading_path=("Network Control", "Concurrency Limit"),
                    relevance=3,
                    evidence="SemaphoreGate bounds simultaneous request concurrency",
                ),
            ),
        ),
        EvaluationQuery(
            query_id="q002",
            query="How are too many simultaneous requests prevented?",
            category="paraphrase",
            relevant_sources=(
                RelevanceJudgment(
                    judgment_id="q002-j1",
                    document_id="network",
                    section_id="concurrency-limit",
                    heading_path=("Network Control", "Concurrency Limit"),
                    relevance=3,
                ),
            ),
        ),
        EvaluationQuery(
            query_id="q003",
            query="Kubernetes ingress controller",
            category="negative",
            relevant_sources=(),
        ),
        EvaluationQuery(
            query_id="q004",
            query="timeout backoff",
            category="multi_relevant",
            relevant_sources=(
                RelevanceJudgment(
                    judgment_id="q004-j1",
                    document_id="timeouts",
                    section_id="timeout-handling",
                    heading_path=("HTTP Timing", "Timeout Handling"),
                    relevance=3,
                ),
                RelevanceJudgment(
                    judgment_id="q004-j2",
                    document_id="retries",
                    section_id="backoff-handling",
                    heading_path=("Retry Control", "Backoff Handling"),
                    relevance=3,
                ),
            ),
        ),
    )
    dataset = EvaluationDataset(
        schema_version=1,
        name="small-offline-retrieval",
        version="1.0.0",
        description="Small deterministic evaluation fixture.",
        documents=documents,
        queries=queries,
        root=Path("fixture"),
    )
    validate_dataset(dataset)
    return dataset


def _document(
    document_id: str,
    title: str,
    heading: str,
    section_id: str,
    body: str,
) -> EvaluationDocument:
    return EvaluationDocument(
        document_id=document_id,
        path=f"documents/{document_id}.html",
        url=f"https://benchmark.invalid/{document_id}",
        title=title,
        sections=(
            EvaluationSection(
                section_id=section_id,
                heading_path=(title, heading),
            ),
        ),
        content=(
            f"<html><head><title>{title}</title></head><body>"
            f'<h1>{title}</h1><section id="{section_id}">'
            f"<h2>{heading}</h2><p>{body}</p></section></body></html>"
        ),
    )


def _item(
    *,
    rank: int,
    document_id: str,
    heading_path: tuple[str, ...],
    text: str,
) -> RetrievedItem:
    return RetrievedItem(
        rank=rank,
        document_id=document_id,
        url=f"https://benchmark.invalid/{document_id}",
        canonical_url=f"https://benchmark.invalid/{document_id}",
        title=heading_path[0],
        section_id="concurrency-limit",
        heading_path=heading_path,
        text=text,
        score=-1.0,
        estimated_tokens=20,
        source_estimated_tokens=100,
        content_hash=f"{document_id}-{rank}",
    )


def _empty_corpus_statistics() -> CorpusStatistics:
    return CorpusStatistics(
        document_count=0,
        section_count=0,
        chunk_count=0,
        source_size_bytes=0,
        cleaned_size_bytes=0,
        source_estimated_tokens=0,
        cleaned_estimated_tokens=0,
        indexing_time_ms=0.0,
    )
