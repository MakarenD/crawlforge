"""Command-line interface for CrawlForge."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from crawlforge import __version__
from crawlforge.advanced import AdvancedCrawler
from crawlforge.config import CrawlerConfig, LoggingConfig, ReportConfig
from crawlforge.context_engine import ContextEngine
from crawlforge.context_models import ContextResult, IndexingResult, SearchHit
from crawlforge.hybrid import HybridRetriever
from crawlforge.hybrid_models import (
    HybridContextResult,
    HybridSearchConfig,
    HybridSearchHit,
)
from crawlforge.logging_config import configure_logging
from crawlforge.semantic_models import (
    DEFAULT_SEMANTIC_DIMENSION,
    DEFAULT_SEMANTIC_MODEL_ID,
    DEFAULT_SEMANTIC_MODEL_REVISION,
    DeviceName,
    EmbeddingProvider,
    SemanticContextResult,
    SemanticIndexInfo,
    SemanticIndexingResult,
    SemanticSearchHit,
)

if TYPE_CHECKING:
    from crawlforge.evaluation.comparison import EvaluationComparison
    from crawlforge.evaluation.models import (
        CorpusStatistics,
        EvaluationDataset,
        EvaluationRun,
    )
    from crawlforge.evaluation.multi_comparison import MultiEvaluationComparison

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
        metavar="{index,embed,search,evaluate}",
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

    embed_parser = commands.add_parser(
        "embed",
        help="build optional local semantic embeddings for indexed chunks",
    )
    embed_parser.add_argument(
        "--database",
        type=Path,
        default=Path(".crawlforge/index.db"),
        help="existing local SQLite context index",
    )
    _add_semantic_model_arguments(embed_parser)
    embed_parser.add_argument(
        "--json",
        action="store_true",
        help="write machine-readable output to stdout",
    )

    search_parser = commands.add_parser(
        "search",
        help="retrieve a compact context from a local index",
    )
    search_parser.add_argument("query", help="retrieval query")
    search_parser.add_argument(
        "--database",
        type=Path,
        default=Path(".crawlforge/index.db"),
        help="local SQLite context index",
    )
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--token-budget", type=int, default=3000)
    search_parser.add_argument(
        "--strategy",
        choices=("bm25", "semantic", "hybrid"),
        default="bm25",
        help="retrieval strategy (default: bm25)",
    )
    _add_semantic_model_arguments(search_parser)
    _add_hybrid_arguments(search_parser)
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
        metavar="{run,compare,validate}",
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
        "--strategy",
        choices=("bm25", "semantic", "hybrid"),
        default="bm25",
        help="retrieval strategy (default: bm25)",
    )
    _add_semantic_model_arguments(evaluate_run_parser)
    _add_hybrid_arguments(evaluate_run_parser)
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

    evaluate_compare_parser = evaluate_commands.add_parser(
        "compare",
        help="compare two or more retrieval strategies",
    )
    evaluate_compare_parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="versioned offline evaluation dataset (default: bundled baseline)",
    )
    evaluate_compare_parser.add_argument(
        "--database",
        type=Path,
        default=Path(".crawlforge/evaluation-compare.db"),
        help="disposable shared SQLite comparison index",
    )
    evaluate_compare_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="paired comparison report path",
    )
    evaluate_compare_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="report serialization format",
    )
    evaluate_compare_parser.add_argument(
        "--strategies",
        default="bm25,semantic",
        help="comma-separated bm25, semantic, and hybrid strategies",
    )
    evaluate_compare_parser.add_argument(
        "--limit-values",
        default="1,3,5,10",
        help="comma-separated retrieval metric cutoffs",
    )
    evaluate_compare_parser.add_argument(
        "--token-budget",
        type=int,
        default=3000,
        help="approximate context token budget",
    )
    evaluate_compare_parser.add_argument(
        "--repeat-latency",
        type=int,
        default=5,
        help="warm-index timing repetitions per query",
    )
    evaluate_compare_parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="deterministic paired bootstrap sample count",
    )
    evaluate_compare_parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260729,
        help="deterministic paired bootstrap seed",
    )
    evaluate_compare_parser.add_argument(
        "--category",
        choices=_EVALUATION_CATEGORIES,
        help="compare only one validated query category",
    )
    evaluate_compare_parser.add_argument(
        "--query-id",
        action="append",
        help="compare one query ID; may be repeated",
    )
    _add_semantic_model_arguments(evaluate_compare_parser)
    _add_hybrid_arguments(evaluate_compare_parser)
    evaluate_compare_parser.add_argument(
        "--json",
        action="store_true",
        help="write the concise comparison summary as JSON to stdout",
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


def _add_semantic_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_SEMANTIC_MODEL_ID,
        help="Sentence Transformers model ID",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_SEMANTIC_MODEL_REVISION,
        help="immutable model revision",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=DEFAULT_SEMANTIC_DIMENSION,
        help="expected embedding dimension",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="semantic inference device",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="semantic document inference batch size",
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=None,
        help="optional local model cache directory",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the pinned model to exist in the local cache",
    )


def _add_hybrid_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = HybridSearchConfig()
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=defaults.rrf_k,
        help=f"RRF rank constant (default: {defaults.rrf_k})",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=defaults.bm25_weight,
        help=f"BM25 RRF weight (default: {defaults.bm25_weight:g})",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=defaults.semantic_weight,
        help=f"semantic RRF weight (default: {defaults.semantic_weight:g})",
    )
    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=defaults.bm25_candidate_limit,
        help=(
            "BM25 candidate depth before fusion "
            f"(default: {defaults.bm25_candidate_limit})"
        ),
    )
    parser.add_argument(
        "--semantic-candidates",
        type=int,
        default=defaults.semantic_candidate_limit,
        help=(
            "semantic candidate depth before fusion "
            f"(default: {defaults.semantic_candidate_limit})"
        ),
    )


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
    if arguments.command == "embed":
        try:
            embedding_result = asyncio.run(_run_embed(arguments))
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as error:
            parser.error(str(error))
        _print_embedding_result(embedding_result, json_output=arguments.json)
        return 1 if embedding_result.failed_chunks else 0
    if arguments.command == "search":
        try:
            context = asyncio.run(_run_search(arguments))
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as error:
            parser.error(str(error))
        if isinstance(context, HybridContextResult):
            _print_hybrid_context_result(context, json_output=arguments.json)
        elif isinstance(context, SemanticContextResult):
            _print_semantic_context_result(context, json_output=arguments.json)
        else:
            _print_context_result(context, json_output=arguments.json)
        return 0
    if arguments.command == "evaluate":
        try:
            if arguments.evaluate_command == "validate":
                _validate_evaluation_dataset(arguments)
                return 0
            if arguments.evaluate_command == "compare":
                return _execute_evaluation_comparison(arguments)
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


async def _run_embed(arguments: argparse.Namespace) -> SemanticIndexingResult:
    database: Path = arguments.database
    if not database.is_file():
        raise ValueError(f"context index does not exist: {database}")
    provider = _semantic_provider(arguments)
    try:
        async with ContextEngine(database) as engine:
            return await engine.index_embeddings(
                provider,
                batch_size=arguments.batch_size,
            )
    finally:
        await provider.close()


async def _run_search(
    arguments: argparse.Namespace,
) -> ContextResult | SemanticContextResult | HybridContextResult:
    database: Path = arguments.database
    if not database.is_file():
        raise ValueError(f"context index does not exist: {database}")
    async with ContextEngine(database) as engine:
        if arguments.strategy == "bm25":
            return await engine.build_context(
                arguments.query,
                limit=arguments.limit,
                token_budget=arguments.token_budget,
            )
        provider = _semantic_provider(arguments)
        try:
            if arguments.strategy == "hybrid":
                retriever = HybridRetriever(
                    context_engine=engine,
                    embedding_provider=provider,
                    config=_hybrid_config(arguments),
                )
                return await retriever.build_context(
                    arguments.query,
                    limit=arguments.limit,
                    token_budget=arguments.token_budget,
                )
            return await engine.build_semantic_context(
                arguments.query,
                provider=provider,
                limit=arguments.limit,
                token_budget=arguments.token_budget,
            )
        finally:
            await provider.close()


def _hybrid_config(arguments: argparse.Namespace) -> HybridSearchConfig:
    return HybridSearchConfig(
        rrf_k=arguments.rrf_k,
        bm25_weight=arguments.bm25_weight,
        semantic_weight=arguments.semantic_weight,
        bm25_candidate_limit=arguments.bm25_candidates,
        semantic_candidate_limit=arguments.semantic_candidates,
    )


def _semantic_provider(
    arguments: argparse.Namespace,
) -> EmbeddingProvider:
    return _semantic_provider_from_values(
        model_id=arguments.model,
        model_revision=arguments.revision,
        dimension=arguments.dimension,
        device=arguments.device,
        batch_size=arguments.batch_size,
        cache_directory=arguments.cache_directory,
        local_files_only=arguments.local_files_only,
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
        "signature": dataset.signature,
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
            strategy_name=arguments.strategy,
            limits=limits,
            token_budget=arguments.token_budget,
            repeat_latency=arguments.repeat_latency,
            model_id=arguments.model,
            model_revision=arguments.revision,
            dimension=arguments.dimension,
            device=arguments.device,
            batch_size=arguments.batch_size,
            cache_directory=arguments.cache_directory,
            local_files_only=arguments.local_files_only,
            hybrid_config=_hybrid_config(arguments),
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


def _execute_evaluation_comparison(arguments: argparse.Namespace) -> int:
    from crawlforge.evaluation.dataset import filter_dataset, load_dataset

    strategies = _parse_evaluation_strategies(arguments.strategies)
    complete_dataset = load_dataset(_evaluation_dataset_path(arguments.dataset))
    dataset = filter_dataset(
        complete_dataset,
        category=arguments.category,
        query_ids=(frozenset(arguments.query_id) if arguments.query_id else None),
    )
    limits = _parse_limit_values(arguments.limit_values)
    if 5 not in limits:
        raise ValueError("paired comparison requires K=5 in --limit-values")
    output = _comparison_output_path(
        arguments,
        dataset=complete_dataset,
        limits=limits,
    )
    database: Path = arguments.database
    _prepare_evaluation_paths(
        dataset=complete_dataset,
        database=database,
        output=output,
    )
    if strategies == ("bm25", "semantic"):
        from crawlforge.evaluation.comparison import write_comparison_report

        comparison = asyncio.run(
            _run_paired_retrieval_evaluation(
                dataset=dataset,
                database=database,
                limits=limits,
                token_budget=arguments.token_budget,
                repeat_latency=arguments.repeat_latency,
                bootstrap_samples=arguments.bootstrap_samples,
                bootstrap_seed=arguments.bootstrap_seed,
                model_id=arguments.model,
                model_revision=arguments.revision,
                dimension=arguments.dimension,
                device=arguments.device,
                batch_size=arguments.batch_size,
                cache_directory=arguments.cache_directory,
                local_files_only=arguments.local_files_only,
            )
        )
        write_comparison_report(
            comparison,
            output,
            report_format=arguments.format,
        )
        _print_comparison_summary(
            comparison,
            output=output,
            json_output=arguments.json,
        )
        has_failures = any(
            query.bm25_failure is not None or query.semantic_failure is not None
            for query in comparison.query_comparisons
        )
        return 1 if has_failures else 0

    from crawlforge.evaluation.multi_comparison import write_multi_comparison_report

    multi_comparison = asyncio.run(
        _run_multi_retrieval_evaluation(
            dataset=dataset,
            database=database,
            strategies=strategies,
            limits=limits,
            token_budget=arguments.token_budget,
            repeat_latency=arguments.repeat_latency,
            bootstrap_samples=arguments.bootstrap_samples,
            bootstrap_seed=arguments.bootstrap_seed,
            model_id=arguments.model,
            model_revision=arguments.revision,
            dimension=arguments.dimension,
            device=arguments.device,
            batch_size=arguments.batch_size,
            cache_directory=arguments.cache_directory,
            local_files_only=arguments.local_files_only,
            hybrid_config=_hybrid_config(arguments),
        )
    )
    write_multi_comparison_report(
        multi_comparison,
        output,
        report_format=arguments.format,
    )
    _print_multi_comparison_summary(
        multi_comparison,
        output=output,
        json_output=arguments.json,
    )
    has_failures = any(
        evidence.failure is not None
        for query in multi_comparison.query_comparisons
        for evidence in query.strategies
    )
    return 1 if has_failures else 0


def _evaluation_output_path(
    arguments: argparse.Namespace,
    *,
    dataset: EvaluationDataset,
    limits: tuple[int, ...],
) -> Path:
    configured_output: Path | None = arguments.output
    if configured_output is not None:
        return configured_output

    is_canonical_semantic = (
        arguments.strategy in ("semantic", "hybrid")
        and arguments.model == DEFAULT_SEMANTIC_MODEL_ID
        and arguments.revision == DEFAULT_SEMANTIC_MODEL_REVISION
        and arguments.dimension == DEFAULT_SEMANTIC_DIMENSION
        and arguments.device == "cpu"
        and arguments.batch_size == 32
        and not arguments.local_files_only
    )
    is_canonical_hybrid = (
        arguments.strategy != "hybrid"
        or _hybrid_config(arguments) == HybridSearchConfig()
    )
    is_canonical_baseline = (
        arguments.dataset is None
        and arguments.category is None
        and not arguments.query_id
        and dataset.name == "crawlforge-retrieval-baseline"
        and dataset.version == "1.0.0"
        and limits == (1, 3, 5, 10)
        and arguments.token_budget == 3000
        and arguments.repeat_latency == 5
        and (arguments.strategy == "bm25" or is_canonical_semantic)
        and is_canonical_hybrid
    )
    if not is_canonical_baseline:
        raise ValueError(
            "custom or filtered evaluations require an explicit --output path"
        )
    report_name = {
        "bm25": "bm25-baseline",
        "semantic": "semantic-baseline",
        "hybrid": "hybrid-baseline",
    }[arguments.strategy]
    return Path(
        f"reports/{report_name}." + ("json" if arguments.format == "json" else "md")
    )


def _comparison_output_path(
    arguments: argparse.Namespace,
    *,
    dataset: EvaluationDataset,
    limits: tuple[int, ...],
) -> Path:
    configured_output: Path | None = arguments.output
    if configured_output is not None:
        return configured_output
    is_canonical_semantic = (
        arguments.model == DEFAULT_SEMANTIC_MODEL_ID
        and arguments.revision == DEFAULT_SEMANTIC_MODEL_REVISION
        and arguments.dimension == DEFAULT_SEMANTIC_DIMENSION
        and arguments.device == "cpu"
        and arguments.batch_size == 32
        and not arguments.local_files_only
    )
    strategies = _parse_evaluation_strategies(arguments.strategies)
    is_canonical = (
        arguments.dataset is None
        and arguments.category is None
        and not arguments.query_id
        and dataset.name == "crawlforge-retrieval-baseline"
        and dataset.version == "1.0.0"
        and limits == (1, 3, 5, 10)
        and arguments.token_budget == 3000
        and arguments.repeat_latency == 5
        and arguments.bootstrap_samples == 5000
        and arguments.bootstrap_seed == 20260729
        and is_canonical_semantic
        and strategies in (("bm25", "semantic"), ("bm25", "semantic", "hybrid"))
        and (
            "hybrid" not in strategies
            or _hybrid_config(arguments) == HybridSearchConfig()
        )
    )
    if not is_canonical:
        raise ValueError(
            "custom or filtered comparisons require an explicit --output path"
        )
    report_name = (
        "bm25-vs-semantic"
        if strategies == ("bm25", "semantic")
        else "bm25-vs-semantic-vs-hybrid"
    )
    return Path(
        f"reports/{report_name}." + ("json" if arguments.format == "json" else "md")
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
    strategy_name: str,
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
    model_id: str,
    model_revision: str | None,
    dimension: int,
    device: str,
    batch_size: int,
    cache_directory: Path | None,
    local_files_only: bool,
    hybrid_config: HybridSearchConfig,
) -> EvaluationRun:
    from crawlforge.chunking import ChunkingConfig, TextChunker
    from crawlforge.evaluation.runner import ingest_evaluation_corpus

    chunking = ChunkingConfig()
    async with ContextEngine(
        database,
        chunker=TextChunker(config=chunking),
    ) as engine:
        corpus_statistics = await ingest_evaluation_corpus(engine, dataset)
        if strategy_name == "bm25":
            return await _evaluate_bm25(
                engine=engine,
                dataset=dataset,
                corpus_statistics=corpus_statistics,
                chunking_configuration=asdict(chunking),
                limits=limits,
                token_budget=token_budget,
                repeat_latency=repeat_latency,
            )
        if strategy_name not in ("semantic", "hybrid"):
            raise ValueError(f"unsupported retrieval strategy: {strategy_name}")
        if strategy_name == "hybrid":
            return await _evaluate_hybrid(
                engine=engine,
                dataset=dataset,
                corpus_statistics=corpus_statistics,
                chunking_configuration=asdict(chunking),
                limits=limits,
                token_budget=token_budget,
                repeat_latency=repeat_latency,
                model_id=model_id,
                model_revision=model_revision,
                dimension=dimension,
                device=device,
                batch_size=batch_size,
                cache_directory=cache_directory,
                local_files_only=local_files_only,
                hybrid_config=hybrid_config,
            )
        return await _evaluate_semantic(
            engine=engine,
            dataset=dataset,
            corpus_statistics=corpus_statistics,
            chunking_configuration=asdict(chunking),
            limits=limits,
            token_budget=token_budget,
            repeat_latency=repeat_latency,
            model_id=model_id,
            model_revision=model_revision,
            dimension=dimension,
            device=device,
            batch_size=batch_size,
            cache_directory=cache_directory,
            local_files_only=local_files_only,
        )


async def _run_paired_retrieval_evaluation(
    *,
    dataset: EvaluationDataset,
    database: Path,
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    model_id: str,
    model_revision: str | None,
    dimension: int,
    device: str,
    batch_size: int,
    cache_directory: Path | None,
    local_files_only: bool,
) -> EvaluationComparison:
    from crawlforge.chunking import ChunkingConfig, TextChunker
    from crawlforge.evaluation.comparison import compare_evaluation_runs
    from crawlforge.evaluation.runner import ingest_evaluation_corpus

    chunking = ChunkingConfig()
    chunking_configuration = asdict(chunking)
    async with ContextEngine(
        database,
        chunker=TextChunker(config=chunking),
    ) as engine:
        corpus_statistics = await ingest_evaluation_corpus(engine, dataset)
        bm25 = await _evaluate_bm25(
            engine=engine,
            dataset=dataset,
            corpus_statistics=corpus_statistics,
            chunking_configuration=chunking_configuration,
            limits=limits,
            token_budget=token_budget,
            repeat_latency=repeat_latency,
        )
        semantic = await _evaluate_semantic(
            engine=engine,
            dataset=dataset,
            corpus_statistics=corpus_statistics,
            chunking_configuration=chunking_configuration,
            limits=limits,
            token_budget=token_budget,
            repeat_latency=repeat_latency,
            model_id=model_id,
            model_revision=model_revision,
            dimension=dimension,
            device=device,
            batch_size=batch_size,
            cache_directory=cache_directory,
            local_files_only=local_files_only,
        )
    return compare_evaluation_runs(
        bm25,
        semantic,
        focus_limit=5,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )


async def _run_multi_retrieval_evaluation(
    *,
    dataset: EvaluationDataset,
    database: Path,
    strategies: tuple[str, ...],
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    model_id: str,
    model_revision: str | None,
    dimension: int,
    device: str,
    batch_size: int,
    cache_directory: Path | None,
    local_files_only: bool,
    hybrid_config: HybridSearchConfig,
) -> MultiEvaluationComparison:
    from crawlforge.chunking import ChunkingConfig, TextChunker
    from crawlforge.evaluation.multi_comparison import (
        compare_multiple_evaluation_runs,
    )
    from crawlforge.evaluation.runner import ingest_evaluation_corpus

    chunking = ChunkingConfig()
    chunking_configuration = asdict(chunking)
    async with ContextEngine(
        database,
        chunker=TextChunker(config=chunking),
    ) as engine:
        corpus_statistics = await ingest_evaluation_corpus(engine, dataset)
        provider = _semantic_provider_from_values(
            model_id=model_id,
            model_revision=model_revision,
            dimension=dimension,
            device=device,
            batch_size=batch_size,
            cache_directory=cache_directory,
            local_files_only=local_files_only,
        )
        try:
            indexing = await engine.index_embeddings(provider, batch_size=batch_size)
            index_info = await engine.get_semantic_index_info(provider)
            runs: list[tuple[str, EvaluationRun]] = []
            for strategy_name in strategies:
                if strategy_name == "bm25":
                    evaluation = await _evaluate_bm25(
                        engine=engine,
                        dataset=dataset,
                        corpus_statistics=corpus_statistics,
                        chunking_configuration=chunking_configuration,
                        limits=limits,
                        token_budget=token_budget,
                        repeat_latency=repeat_latency,
                    )
                elif strategy_name == "semantic":
                    evaluation = await _evaluate_semantic_prepared(
                        engine=engine,
                        provider=provider,
                        dataset=dataset,
                        corpus_statistics=corpus_statistics,
                        chunking_configuration=chunking_configuration,
                        limits=limits,
                        token_budget=token_budget,
                        repeat_latency=repeat_latency,
                        batch_size=batch_size,
                        indexing=indexing,
                        index_info=index_info,
                    )
                else:
                    evaluation = await _evaluate_hybrid_prepared(
                        engine=engine,
                        provider=provider,
                        dataset=dataset,
                        corpus_statistics=corpus_statistics,
                        chunking_configuration=chunking_configuration,
                        limits=limits,
                        token_budget=token_budget,
                        repeat_latency=repeat_latency,
                        batch_size=batch_size,
                        indexing=indexing,
                        index_info=index_info,
                        hybrid_config=hybrid_config,
                    )
                runs.append((strategy_name, evaluation))
        finally:
            await provider.close()
    return compare_multiple_evaluation_runs(
        runs,
        focus_limit=5,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )


async def _evaluate_bm25(
    *,
    engine: ContextEngine,
    dataset: EvaluationDataset,
    corpus_statistics: CorpusStatistics,
    chunking_configuration: dict[str, object],
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
) -> EvaluationRun:
    from crawlforge.evaluation.runner import (
        BM25ContextEngineStrategy,
        RetrievalEvaluationRunner,
    )

    runner = RetrievalEvaluationRunner(
        dataset=dataset,
        retriever=BM25ContextEngineStrategy(engine, dataset),
        corpus_statistics=corpus_statistics,
        retrieval_configuration={
            "index": "sqlite_fts5",
            "ranking": "bm25",
            "score_order": "ascending",
        },
        chunking_configuration=chunking_configuration,
    )
    return await runner.run(
        limits=limits,
        token_budget=token_budget,
        repeat_latency=repeat_latency,
    )


async def _evaluate_semantic(
    *,
    engine: ContextEngine,
    dataset: EvaluationDataset,
    corpus_statistics: CorpusStatistics,
    chunking_configuration: dict[str, object],
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
    model_id: str,
    model_revision: str | None,
    dimension: int,
    device: str,
    batch_size: int,
    cache_directory: Path | None,
    local_files_only: bool,
) -> EvaluationRun:
    provider = _semantic_provider_from_values(
        model_id=model_id,
        model_revision=model_revision,
        dimension=dimension,
        device=device,
        batch_size=batch_size,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    try:
        indexing = await engine.index_embeddings(provider, batch_size=batch_size)
        index_info = await engine.get_semantic_index_info(provider)
        return await _evaluate_semantic_prepared(
            engine=engine,
            provider=provider,
            dataset=dataset,
            corpus_statistics=corpus_statistics,
            chunking_configuration=chunking_configuration,
            limits=limits,
            token_budget=token_budget,
            repeat_latency=repeat_latency,
            batch_size=batch_size,
            indexing=indexing,
            index_info=index_info,
        )
    finally:
        await provider.close()


async def _evaluate_semantic_prepared(
    *,
    engine: ContextEngine,
    provider: EmbeddingProvider,
    dataset: EvaluationDataset,
    corpus_statistics: CorpusStatistics,
    chunking_configuration: dict[str, object],
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
    batch_size: int,
    indexing: SemanticIndexingResult,
    index_info: SemanticIndexInfo,
) -> EvaluationRun:
    from crawlforge.evaluation.runner import RetrievalEvaluationRunner
    from crawlforge.evaluation.semantic_strategy import (
        SemanticContextEngineStrategy,
    )

    model = indexing.model
    strategy = SemanticContextEngineStrategy(
        engine,
        provider,
        dataset,
        indexing_result=indexing,
        index_info=index_info,
    )
    configuration: dict[str, object] = {
        "index": "sqlite_float32_blob",
        "ranking": "exact_cosine_similarity",
        "score_order": "descending",
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "model_fingerprint": model.fingerprint,
        "provider": model.provider,
        "dimension": model.dimension,
        "precision": model.precision,
        "normalized": model.normalized,
        "document_format_version": model.document_format_version,
        "query_format_version": model.query_format_version,
        "batch_size": batch_size,
    }
    runner = RetrievalEvaluationRunner(
        dataset=dataset,
        retriever=strategy,
        corpus_statistics=corpus_statistics,
        retrieval_configuration=configuration,
        chunking_configuration=chunking_configuration,
    )
    evaluation = await runner.run(
        limits=limits,
        token_budget=token_budget,
        repeat_latency=repeat_latency,
    )
    return replace(
        evaluation,
        retrieval_configuration={
            **evaluation.retrieval_configuration,
            **strategy.performance_metadata(),
        },
    )


async def _evaluate_hybrid(
    *,
    engine: ContextEngine,
    dataset: EvaluationDataset,
    corpus_statistics: CorpusStatistics,
    chunking_configuration: dict[str, object],
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
    model_id: str,
    model_revision: str | None,
    dimension: int,
    device: str,
    batch_size: int,
    cache_directory: Path | None,
    local_files_only: bool,
    hybrid_config: HybridSearchConfig,
) -> EvaluationRun:
    provider = _semantic_provider_from_values(
        model_id=model_id,
        model_revision=model_revision,
        dimension=dimension,
        device=device,
        batch_size=batch_size,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    try:
        indexing = await engine.index_embeddings(provider, batch_size=batch_size)
        index_info = await engine.get_semantic_index_info(provider)
        return await _evaluate_hybrid_prepared(
            engine=engine,
            provider=provider,
            dataset=dataset,
            corpus_statistics=corpus_statistics,
            chunking_configuration=chunking_configuration,
            limits=limits,
            token_budget=token_budget,
            repeat_latency=repeat_latency,
            batch_size=batch_size,
            indexing=indexing,
            index_info=index_info,
            hybrid_config=hybrid_config,
        )
    finally:
        await provider.close()


async def _evaluate_hybrid_prepared(
    *,
    engine: ContextEngine,
    provider: EmbeddingProvider,
    dataset: EvaluationDataset,
    corpus_statistics: CorpusStatistics,
    chunking_configuration: dict[str, object],
    limits: tuple[int, ...],
    token_budget: int,
    repeat_latency: int,
    batch_size: int,
    indexing: SemanticIndexingResult,
    index_info: SemanticIndexInfo,
    hybrid_config: HybridSearchConfig,
) -> EvaluationRun:
    from crawlforge.evaluation.hybrid_strategy import HybridContextEngineStrategy
    from crawlforge.evaluation.runner import RetrievalEvaluationRunner

    model = indexing.model
    strategy = HybridContextEngineStrategy(
        engine,
        provider,
        dataset,
        config=hybrid_config,
        indexing_result=indexing,
        index_info=index_info,
    )
    configuration: dict[str, object] = {
        "index": "sqlite_fts5+sqlite_float32_blob",
        "ranking": "reciprocal_rank_fusion",
        "fusion_strategy": "reciprocal-rank-fusion",
        "score_order": "descending",
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "model_fingerprint": model.fingerprint,
        "provider": model.provider,
        "dimension": model.dimension,
        "precision": model.precision,
        "normalized": model.normalized,
        "document_format_version": model.document_format_version,
        "query_format_version": model.query_format_version,
        "batch_size": batch_size,
        **asdict(hybrid_config),
    }
    runner = RetrievalEvaluationRunner(
        dataset=dataset,
        retriever=strategy,
        corpus_statistics=corpus_statistics,
        retrieval_configuration=configuration,
        chunking_configuration=chunking_configuration,
    )
    evaluation = await runner.run(
        limits=limits,
        token_budget=token_budget,
        repeat_latency=repeat_latency,
    )
    return replace(
        evaluation,
        retrieval_configuration={
            **evaluation.retrieval_configuration,
            **strategy.performance_metadata(),
        },
    )


def _semantic_provider_from_values(
    *,
    model_id: str,
    model_revision: str | None,
    dimension: int,
    device: str,
    batch_size: int,
    cache_directory: Path | None,
    local_files_only: bool,
) -> EmbeddingProvider:
    from crawlforge.semantic_provider import SentenceTransformerEmbeddingProvider

    return SentenceTransformerEmbeddingProvider(
        model_id=model_id,
        revision=model_revision,
        dimension=dimension,
        device=cast(DeviceName, device),
        batch_size=batch_size,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
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


def _parse_evaluation_strategies(value: str) -> tuple[str, ...]:
    strategies = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(strategies) < 2:
        raise ValueError("strategies must contain at least two retrieval strategies")
    unsupported = tuple(
        strategy
        for strategy in strategies
        if strategy not in ("bm25", "semantic", "hybrid")
    )
    if unsupported:
        raise ValueError("unsupported retrieval strategies: " + ", ".join(unsupported))
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates")
    return strategies


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


def _print_comparison_summary(
    comparison: EvaluationComparison,
    *,
    output: Path,
    json_output: bool,
) -> None:
    metrics = {metric.metric: metric for metric in comparison.metrics}
    payload = {
        "dataset": comparison.dataset_name,
        "version": comparison.dataset_version,
        "signature": comparison.dataset_signature,
        "bm25_mrr": metrics["MRR"].bm25,
        "semantic_mrr": metrics["MRR"].semantic,
        "mrr_delta": metrics["MRR"].delta,
        "semantic_wins": len(comparison.semantic_wins),
        "bm25_wins": len(comparison.bm25_wins),
        "both_fail": len(comparison.both_fail),
        "report": str(output),
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    print(
        "Compared "
        f"{comparison.baseline_strategy} with {comparison.candidate_strategy} "
        f"on {len(comparison.query_comparisons)} queries."
    )
    print(
        f"MRR: {_optional_metric(metrics['MRR'].bm25)} -> "
        f"{_optional_metric(metrics['MRR'].semantic)} "
        f"({_optional_delta(metrics['MRR'].delta)})."
    )
    print(f"Report: {output}")


def _print_multi_comparison_summary(
    comparison: MultiEvaluationComparison,
    *,
    output: Path,
    json_output: bool,
) -> None:
    mrr = next(
        metric for metric in comparison.aggregate_metrics if metric.metric == "MRR"
    )
    mrr_values = {value.strategy_alias: value.value for value in mrr.values}
    aliases = tuple(strategy.alias for strategy in comparison.strategies)
    payload = {
        "dataset": comparison.dataset_name,
        "version": comparison.dataset_version,
        "signature": comparison.dataset_signature,
        "strategies": list(aliases),
        "mrr": mrr_values,
        "queries": len(comparison.query_comparisons),
        "report": str(output),
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    print(
        f"Compared {', '.join(aliases)} on {len(comparison.query_comparisons)} queries."
    )
    print(
        "MRR: "
        + "; ".join(
            f"{alias}={_optional_metric(mrr_values[alias])}" for alias in aliases
        )
        + "."
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


def _print_embedding_result(
    result: SemanticIndexingResult,
    *,
    json_output: bool,
) -> None:
    payload = asdict(result)
    if json_output:
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    print(f"Model: {result.model.model_id}")
    print(f"Revision: {result.model.model_revision or 'unversioned'}")
    print(f"Fingerprint: {result.model.fingerprint}")
    print(f"Dimension: {result.model.dimension}")
    print(
        "Chunks: "
        f"{result.considered_chunks} total, "
        f"{result.embedded_chunks} embedded, "
        f"{result.cache_hits} cache hits"
    )
    print(
        "Invalidated: "
        f"{result.invalidated_embeddings}; failures: {result.failed_chunks}"
    )
    print(
        f"Vector bytes: {result.stored_vector_bytes} written, "
        f"{result.total_stored_vector_bytes} total; "
        f"elapsed: {result.elapsed_time_ms:.3f} ms"
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


def _print_semantic_context_result(
    result: SemanticContextResult,
    *,
    json_output: bool,
) -> None:
    if json_output:
        print(
            json.dumps(
                _semantic_context_payload(result),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if not result.hits:
        print("No matching semantic context found.")
    for hit in result.hits:
        section = " > ".join(hit.chunk.heading_path)
        print(f"{hit.rank}. {hit.source.title or hit.source.url}")
        print(f"   URL: {hit.source.url}")
        print(
            f"   Cosine similarity: {hit.cosine_similarity:.6f} "
            "(higher is more relevant)"
        )
        if section:
            print(f"   Section: {section}")
        print(f"   {hit.chunk.text}")
    print(
        "Context: "
        f"~{result.estimated_tokens}/{result.token_budget} estimated tokens, "
        f"{result.total_size_chars} characters, "
        f"{result.candidates_considered} candidates"
    )
    print(f"Model fingerprint: {result.model_fingerprint}")
    print(f"Estimated context reduction: {result.estimated_context_reduction:.1%}")


def _print_hybrid_context_result(
    result: HybridContextResult,
    *,
    json_output: bool,
) -> None:
    if json_output:
        print(
            json.dumps(
                _hybrid_context_payload(result),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if not result.hits:
        print("No matching hybrid context found.")
    for hit in result.hits:
        section = " > ".join(hit.chunk.heading_path)
        print(f"{hit.rank}. {hit.source.title or hit.source.url}")
        print(f"   URL: {hit.source.url}")
        print(f"   RRF score: {hit.rrf_score:.8f} (higher is more relevant)")
        print(
            "   Component ranks: "
            f"BM25 {_optional_rank(hit.bm25_rank)}; "
            f"semantic {_optional_rank(hit.semantic_rank)}"
        )
        if section:
            print(f"   Heading: {section}")
        print(f"   Estimated tokens: {hit.chunk.estimated_tokens}")
        print(f"   {hit.chunk.text}")
    print(
        "Context: "
        f"~{result.estimated_tokens}/{result.token_budget} estimated tokens, "
        f"{result.total_size_chars} characters, "
        f"{result.candidates_considered} candidates"
    )
    config = result.fusion_configuration
    print(
        "Fusion: reciprocal-rank-fusion "
        f"(k={config.rrf_k}, BM25 weight={config.bm25_weight:g}, "
        f"semantic weight={config.semantic_weight:g})"
    )
    print(f"Model fingerprint: {result.model_fingerprint}")
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


def _semantic_context_payload(result: SemanticContextResult) -> dict[str, object]:
    return {
        "query": result.query,
        "retrieval_strategy": result.retrieval_strategy,
        "score_type": result.score_type,
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "model_fingerprint": result.model_fingerprint,
        "results": [_semantic_hit_payload(hit) for hit in result.hits],
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


def _semantic_hit_payload(hit: SemanticSearchHit) -> dict[str, object]:
    return {
        "rank": hit.rank,
        "cosine_similarity": hit.cosine_similarity,
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


def _hybrid_context_payload(result: HybridContextResult) -> dict[str, object]:
    return {
        "query": result.query,
        "retrieval_strategy": result.retrieval_strategy,
        "fusion_strategy": result.fusion_strategy,
        "score_type": result.score_type,
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "model_fingerprint": result.model_fingerprint,
        "fusion_configuration": asdict(result.fusion_configuration),
        "results": [_hybrid_hit_payload(hit) for hit in result.hits],
        "total_size_chars": result.total_size_chars,
        "estimated_tokens": result.estimated_tokens,
        "candidates_considered": result.candidates_considered,
        "search_time_ms": result.search_time_ms,
        "context_selection_time_ms": result.context_selection_time_ms,
        "limit": result.limit,
        "token_budget": result.token_budget,
        "source_estimated_tokens": result.source_estimated_tokens,
        "estimated_context_reduction": result.estimated_context_reduction,
        "index_hit": result.index_hit,
        "metrics": asdict(result.metrics),
    }


def _hybrid_hit_payload(hit: HybridSearchHit) -> dict[str, object]:
    return {
        "identity": hit.identity,
        "rank": hit.rank,
        "retrieval_strategy": hit.retrieval_strategy,
        "fusion_strategy": hit.fusion_strategy,
        "score_type": hit.score_type,
        "rrf_score": hit.rrf_score,
        "bm25_rank": hit.bm25_rank,
        "semantic_rank": hit.semantic_rank,
        "bm25_contribution": hit.bm25_contribution,
        "semantic_contribution": hit.semantic_contribution,
        "bm25_score": hit.bm25_score,
        "cosine_similarity": hit.cosine_similarity,
        "contributions": [asdict(item) for item in hit.contributions],
        "model_id": hit.model_id,
        "model_revision": hit.model_revision,
        "model_fingerprint": hit.model_fingerprint,
        "fusion_configuration": asdict(hit.fusion_configuration),
        "url": hit.source.url,
        "canonical_url": hit.source.canonical_url,
        "title": hit.source.title,
        "fetched_at": hit.source.fetched_at.isoformat(),
        "section": list(hit.chunk.heading_path),
        "text": hit.chunk.text,
        "size_chars": hit.chunk.size_chars,
        "estimated_tokens": hit.chunk.estimated_tokens,
        "chunk_id": hit.chunk.id,
        "content_hash": hit.chunk.content_hash,
        "document_id": hit.source.document_id,
    }


def _optional_rank(value: int | None) -> str:
    return str(value) if value is not None else "n/a"


def _optional_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _optional_delta(value: float | None) -> str:
    return f"{value:+.4f}" if value is not None else "n/a"


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
