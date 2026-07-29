"""Command-line interface for CrawlForge."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from crawlforge import __version__
from crawlforge.advanced import AdvancedCrawler
from crawlforge.config import CrawlerConfig, LoggingConfig, ReportConfig
from crawlforge.context_engine import ContextEngine
from crawlforge.context_models import ContextResult, IndexingResult, SearchHit
from crawlforge.logging_config import configure_logging

if TYPE_CHECKING:
    from crawlforge.evaluation.models import EvaluationDataset, EvaluationRun

_EVALUATION_CATEGORIES = (
    "exact_term",
    "code_symbol",
    "error_lookup",
    "paraphrase",
    "conceptual",
    "ambiguous",
    "multi_relevant",
    "negative",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="crawlforge",
        description="High-performance asynchronous web crawler for Python.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show the installed CrawlForge version and exit",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        metavar="URL",
        help="one or more starting URLs",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="maximum number of pages to attempt",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="maximum discovered-link depth",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="maximum active requests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON result file",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        default=None,
        help="standalone HTML report file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file",
    )
    parser.add_argument(
        "--respect-robots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable robots.txt enforcement",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=None,
        metavar="REQUESTS_PER_SECOND",
        help="request rate limit",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="rotating log file",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="console and file logging level",
    )
    commands = parser.add_subparsers(
        dest="command",
        metavar="{index,search,evaluate}",
    )
    index_parser = commands.add_parser(
        "index",
        help="crawl a site into a local lexical context index",
    )
    index_parser.add_argument("url", help="starting HTTP or HTTPS URL")
    index_parser.add_argument(
        "--database",
        type=Path,
        default=Path(".crawlforge/index.db"),
        help="local SQLite context index",
    )
    index_parser.add_argument("--max-pages", type=int, default=100)
    index_parser.add_argument("--max-depth", type=int, default=2)
    index_parser.add_argument("--max-concurrent", type=int, default=10)
    index_parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        metavar="REQUESTS_PER_SECOND",
    )
    index_parser.add_argument(
        "--respect-robots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    index_parser.add_argument(
        "--same-domain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restrict discovered pages to the starting hostname",
    )
    index_parser.add_argument(
        "--json",
        action="store_true",
        help="write machine-readable output to stdout",
    )

    search_parser = commands.add_parser(
        "search",
        help="retrieve a compact context from a local index",
    )
    search_parser.add_argument("query", help="lexical search query")
    search_parser.add_argument(
        "--database",
        type=Path,
        default=Path(".crawlforge/index.db"),
        help="local SQLite context index",
    )
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--token-budget", type=int, default=3000)
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="write machine-readable output to stdout",
    )

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="run or validate deterministic retrieval evaluations",
    )
    evaluate_commands = evaluate_parser.add_subparsers(
        dest="evaluate_command",
        required=True,
        metavar="{run,validate}",
    )
    evaluate_run_parser = evaluate_commands.add_parser(
        "run",
        help="build a clean local index and evaluate retrieval",
    )
    evaluate_run_parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="versioned offline evaluation dataset (default: bundled baseline)",
    )
    evaluate_run_parser.add_argument(
        "--database",
        type=Path,
        default=Path(".crawlforge/evaluation.db"),
        help="disposable SQLite evaluation index",
    )
    evaluate_run_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="evaluation report path",
    )
    evaluate_run_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="report serialization format",
    )
    evaluate_run_parser.add_argument(
        "--limit-values",
        default="1,3,5,10",
        help="comma-separated retrieval metric cutoffs",
    )
    evaluate_run_parser.add_argument(
        "--token-budget",
        type=int,
        default=3000,
        help="approximate context token budget",
    )
    evaluate_run_parser.add_argument(
        "--repeat-latency",
        type=int,
        default=5,
        help="warm-index timing repetitions per query",
    )
    evaluate_run_parser.add_argument(
        "--category",
        choices=_EVALUATION_CATEGORIES,
        help="evaluate only one validated query category",
    )
    evaluate_run_parser.add_argument(
        "--query-id",
        action="append",
        help="evaluate one query ID; may be repeated",
    )
    evaluate_run_parser.add_argument(
        "--json",
        action="store_true",
        help="write the concise run summary as JSON to stdout",
    )

    evaluate_validate_parser = evaluate_commands.add_parser(
        "validate",
        help="validate dataset structure and relevance references",
    )
    evaluate_validate_parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="versioned offline evaluation dataset (default: bundled baseline)",
    )
    evaluate_validate_parser.add_argument(
        "--json",
        action="store_true",
        help="write the validation summary as JSON to stdout",
    )
    return parser


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    """Run one configured crawl and return its final statistics."""
    config = _config_from_arguments(arguments)
    configure_logging(config.logging)
    crawler = AdvancedCrawler(config)
    try:
        await crawler.crawl()
        return crawler.get_stats()
    finally:
        await crawler.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CrawlForge command-line interface."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "index":
        try:
            result = asyncio.run(_run_index(arguments))
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as error:
            parser.error(str(error))
        _print_index_result(result, json_output=arguments.json)
        return 0
    if arguments.command == "search":
        try:
            context = asyncio.run(_run_search(arguments))
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as error:
            parser.error(str(error))
        _print_context_result(context, json_output=arguments.json)
        return 0
    if arguments.command == "evaluate":
        try:
            if arguments.evaluate_command == "validate":
                _validate_evaluation_dataset(arguments)
                return 0
            return _execute_evaluation(arguments)
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as error:
            parser.error(str(error))
    if arguments.urls is None and arguments.config is None:
        parser.print_help()
        return 0
    try:
        stats = asyncio.run(run(arguments))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "total_pages": stats["total_pages"],
                "successful": stats["successful"],
                "failed": stats["failed"],
                "average_speed": stats["average_speed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


async def _run_index(arguments: argparse.Namespace) -> IndexingResult:
    if arguments.max_pages <= 0:
        raise ValueError("max-pages must be greater than zero")
    if arguments.max_depth < 0:
        raise ValueError("max-depth must be zero or greater")
    if arguments.max_concurrent <= 0:
        raise ValueError("max-concurrent must be greater than zero")
    database: Path = arguments.database
    await asyncio.to_thread(database.parent.mkdir, parents=True, exist_ok=True)
    async with ContextEngine(database) as engine:
        return await engine.ingest_url(
            arguments.url,
            max_pages=arguments.max_pages,
            max_depth=arguments.max_depth,
            max_concurrent=arguments.max_concurrent,
            requests_per_second=arguments.rate_limit,
            respect_robots=arguments.respect_robots,
            same_domain_only=arguments.same_domain,
        )


async def _run_search(arguments: argparse.Namespace) -> ContextResult:
    database: Path = arguments.database
    if not database.is_file():
        raise ValueError(f"context index does not exist: {database}")
    async with ContextEngine(database) as engine:
        return await engine.build_context(
            arguments.query,
            limit=arguments.limit,
            token_budget=arguments.token_budget,
        )


def _validate_evaluation_dataset(arguments: argparse.Namespace) -> None:
    from crawlforge.evaluation.dataset import load_dataset

    dataset = load_dataset(_evaluation_dataset_path(arguments.dataset))
    category_counts = {
        category: sum(query.category == category for query in dataset.queries)
        for category in _EVALUATION_CATEGORIES
    }
    payload = {
        "valid": True,
        "dataset": dataset.name,
        "version": dataset.version,
        "documents": len(dataset.documents),
        "sections": sum(len(document.sections) for document in dataset.documents),
        "queries": len(dataset.queries),
        "categories": category_counts,
    }
    if arguments.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    print(
        f"Dataset {dataset.name} {dataset.version} is valid: "
        f"{len(dataset.documents)} documents, "
        f"{payload['sections']} sections, "
        f"{len(dataset.queries)} queries."
    )


def _execute_evaluation(arguments: argparse.Namespace) -> int:
    from crawlforge.evaluation.dataset import filter_dataset, load_dataset
    from crawlforge.evaluation.reporting import write_evaluation_report

    complete_dataset = load_dataset(_evaluation_dataset_path(arguments.dataset))
    dataset = filter_dataset(
        complete_dataset,
        category=arguments.category,
        query_ids=(frozenset(arguments.query_id) if arguments.query_id else None),
    )
    limits = _parse_limit_values(arguments.limit_values)
    database: Path = arguments.database
    output = _evaluation_output_path(
        arguments,
        dataset=complete_dataset,
        limits=limits,
    )
    _prepare_evaluation_paths(
        dataset=complete_dataset,
        database=database,
        output=output,
    )
    evaluation = asyncio.run(
        _run_retrieval_evaluation(
            dataset=dataset,
            database=database,
            limits=limits,
            token_budget=arguments.token_budget,
            repeat_latency=arguments.repeat_latency,
        )
    )
    write_evaluation_report(
        evaluation,
        output,
        report_format=arguments.format,
    )
    _print_evaluation_summary(
        evaluation,
        output=output,
        json_output=arguments.json,
    )
    return 1 if evaluation.failures else 0


def _evaluation_output_path(
    arguments: argparse.Namespace,
    *,
    dataset: EvaluationDataset,
    limits: tuple[int, ...],
) -> Path:
    configured_output: Path | None = arguments.output
    if configured_output is not None:
        return configured_output

    is_canonical_baseline = (
        arguments.dataset is None
        and arguments.category is None
        and not arguments.query_id
        and dataset.name == "crawlforge-retrieval-baseline"
        and dataset.version == "1.0.0"
        and limits == (1, 3, 5, 10)
        and arguments.token_budget == 3000
        and arguments.repeat_latency == 5
    )
    if not is_canonical_baseline:
        raise ValueError(
            "custom or filtered evaluations require an explicit --output path"
        )
    return Path(
        "reports/bm25-baseline." + ("json" if arguments.format == "json" else "md")
    )


def _evaluation_dataset_path(configured: Path | None) -> Path:
    if configured is not None:
        return configured

    source_dataset = Path(__file__).resolve().parents[2] / "benchmarks" / "retrieval"
    if (source_dataset / "manifest.json").is_file():
        return source_dataset

    from importlib.resources import files

    bundled = files("crawlforge.evaluation").joinpath("data", "retrieval")
    bundled_path = Path(str(bundled))
    if not (bundled_path / "manifest.json").is_file():
        raise RuntimeError(
            "bundled evaluation dataset is unavailable; provide --dataset"
        )
    return bundled_path


async def _run_retrieval_evaluation(
    *,
    dataset: EvaluationDataset,
    database: Path,
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
) -> EvaluationRun:
    from crawlforge.chunking import ChunkingConfig, TextChunker
    from crawlforge.evaluation.runner import (
        BM25ContextEngineStrategy,
        RetrievalEvaluationRunner,
        ingest_evaluation_corpus,
    )

    chunking = ChunkingConfig()
    async with ContextEngine(
        database,
        chunker=TextChunker(config=chunking),
    ) as engine:
        corpus_statistics = await ingest_evaluation_corpus(engine, dataset)
        strategy = BM25ContextEngineStrategy(engine, dataset)
        runner = RetrievalEvaluationRunner(
            dataset=dataset,
            retriever=strategy,
            corpus_statistics=corpus_statistics,
            retrieval_configuration={
                "index": "sqlite_fts5",
                "ranking": "bm25",
                "score_order": "ascending",
            },
            chunking_configuration=asdict(chunking),
        )
        return await runner.run(
            limits=limits,
            token_budget=token_budget,
            repeat_latency=repeat_latency,
        )


def _prepare_evaluation_paths(
    *,
    dataset: EvaluationDataset,
    database: Path,
    output: Path,
) -> None:
    dataset_root = dataset.root.resolve()
    database_path = database.resolve()
    output_path = output.resolve()
    database_namespace = (
        database_path,
        Path(f"{database_path}-wal"),
        Path(f"{database_path}-shm"),
        Path(f"{database_path}-journal"),
    )
    if output_path in database_namespace:
        raise ValueError(
            "evaluation report output conflicts with evaluation database files"
        )
    if database_path.is_relative_to(dataset_root):
        raise ValueError("evaluation database must be outside the dataset")
    if output_path.is_relative_to(dataset_root):
        raise ValueError("evaluation report must be outside the dataset")

    database.parent.mkdir(parents=True, exist_ok=True)
    for candidate in database_namespace:
        candidate.unlink(missing_ok=True)


def _parse_limit_values(value: str) -> tuple[int, ...]:
    try:
        limits = tuple(
            sorted({int(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as error:
        raise ValueError("limit-values must be comma-separated integers") from error
    if not limits or limits[0] <= 0:
        raise ValueError("limit-values must contain positive integers")
    return limits


def _print_evaluation_summary(
    evaluation: EvaluationRun,
    *,
    output: Path,
    json_output: bool,
) -> None:
    metrics = evaluation.aggregate_metrics
    focus_limit = 5 if 5 in metrics.hit_rate_at else max(metrics.hit_rate_at)
    payload = {
        "dataset": evaluation.dataset_name,
        "version": evaluation.dataset_version,
        "strategy": evaluation.retrieval_strategy,
        "queries": metrics.query_count,
        f"hit_rate_at_{focus_limit}": metrics.hit_rate_at[focus_limit],
        "mrr": metrics.mrr,
        "no_result_accuracy": metrics.no_result_accuracy,
        "failures": len(evaluation.failures),
        "report": str(output),
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    print(
        f"Evaluated {metrics.query_count} queries with {evaluation.retrieval_strategy}."
    )
    print(
        f"Hit Rate@{focus_limit}: {metrics.hit_rate_at[focus_limit]:.1%}; "
        f"MRR: {metrics.mrr:.4f}."
    )
    print(f"Report: {output}")


def _print_index_result(result: IndexingResult, *, json_output: bool) -> None:
    payload = asdict(result)
    if json_output:
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(
        f"Indexed {result.documents_indexed}/{result.documents_seen} documents "
        f"and {result.chunks_indexed} new chunks."
    )
    print(f"Database session: {result.session_id}")
    print(
        "Content: "
        f"{result.source_size_bytes} raw bytes -> "
        f"{result.cleaned_size_bytes} cleaned bytes"
    )
    print(
        "Estimated tokens: "
        f"{result.source_estimated_tokens} raw -> "
        f"{result.cleaned_estimated_tokens} cleaned"
    )
    print(
        f"Duplicates: {result.duplicate_documents} documents, "
        f"{result.duplicate_chunks} chunks"
    )
    if result.failed_pages:
        categories = ", ".join(result.failure_categories)
        print(
            f"Omitted pages: {result.failed_pages}"
            + (f" ({categories})" if categories else "")
        )


def _print_context_result(result: ContextResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(_context_payload(result), ensure_ascii=False))
        return
    if not result.hits:
        print("No matching context found.")
    for hit in result.hits:
        section = " > ".join(hit.chunk.heading_path)
        print(f"{hit.rank}. {hit.source.title or hit.source.url}")
        print(f"   URL: {hit.source.url}")
        print(f"   BM25: {hit.bm25_score:.6f} (lower is more relevant)")
        if section:
            print(f"   Section: {section}")
        print(f"   {hit.chunk.text}")
    print(
        "Context: "
        f"~{result.estimated_tokens}/{result.token_budget} estimated tokens, "
        f"{result.total_size_chars} characters, "
        f"{result.candidates_considered} candidates"
    )
    print(f"Estimated context reduction: {result.estimated_context_reduction:.1%}")


def _context_payload(result: ContextResult) -> dict[str, object]:
    return {
        "query": result.query,
        "results": [_hit_payload(hit) for hit in result.hits],
        "total_size_chars": result.total_size_chars,
        "estimated_tokens": result.estimated_tokens,
        "candidates_considered": result.candidates_considered,
        "search_time_ms": result.search_time_ms,
        "limit": result.limit,
        "token_budget": result.token_budget,
        "source_estimated_tokens": result.source_estimated_tokens,
        "estimated_context_reduction": result.estimated_context_reduction,
        "index_hit": result.index_hit,
    }


def _hit_payload(hit: SearchHit) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "bm25_score": hit.bm25_score,
        "url": hit.source.url,
        "canonical_url": hit.source.canonical_url,
        "title": hit.source.title,
        "fetched_at": hit.source.fetched_at.isoformat(),
        "section": list(hit.chunk.heading_path),
        "text": hit.chunk.text,
        "size_chars": hit.chunk.size_chars,
        "estimated_tokens": hit.chunk.estimated_tokens,
        "chunk_id": hit.chunk.id,
        "document_id": hit.source.document_id,
    }


def _config_from_arguments(arguments: argparse.Namespace) -> CrawlerConfig:
    if arguments.config is not None:
        config = CrawlerConfig.from_file(arguments.config)
    else:
        config = CrawlerConfig(
            start_urls=tuple(arguments.urls or ()),
            reports=ReportConfig(
                json=arguments.output or Path("results.json"),
                html=arguments.html_report,
            ),
            logging=LoggingConfig(
                level=arguments.log_level or "INFO",
                file=arguments.log_file,
            ),
        )

    config = config.with_overrides(
        start_urls=arguments.urls,
        max_pages=arguments.max_pages,
        max_depth=arguments.max_depth,
        rate_limit=arguments.rate_limit,
        respect_robots=arguments.respect_robots,
        json_report=arguments.output,
    )
    if arguments.max_concurrent is not None:
        config = replace(config, max_concurrent=arguments.max_concurrent)
    if arguments.html_report is not None:
        config = replace(
            config,
            reports=replace(config.reports, html=arguments.html_report),
        )
    if arguments.log_level is not None or arguments.log_file is not None:
        config = replace(
            config,
            logging=replace(
                config.logging,
                level=arguments.log_level or config.logging.level,
                file=(
                    arguments.log_file
                    if arguments.log_file is not None
                    else config.logging.file
                ),
            ),
        )
    return config
