"""Benchmark CrawlForge's local content-to-context pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TypedDict

from crawlforge import ContextEngine
from crawlforge.crawler import CrawledPage

DEFAULT_DOCUMENT_SCALES = (10, 100)
DEFAULT_LARGE_PARAGRAPHS = 500
DEFAULT_SEARCH_ITERATIONS = 50
_FETCHED_AT = datetime(2026, 1, 1, tzinfo=UTC)


class BenchmarkResult(TypedDict):
    """Measurements for one processing, indexing, and search workload."""

    workload: str
    docs: int
    source_bytes: int
    clean_bytes: int
    chunks: int
    indexing_seconds: float
    search_iterations: int
    total_search_milliseconds: float
    average_search_milliseconds: float
    hits: int
    estimated_returned_tokens: int
    estimated_context_reduction: float


async def run_benchmarks(
    document_scales: Sequence[int] = DEFAULT_DOCUMENT_SCALES,
    *,
    large_paragraphs: int = DEFAULT_LARGE_PARAGRAPHS,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
) -> list[BenchmarkResult]:
    """Run deterministic workloads against temporary SQLite FTS5 indexes."""
    scales = tuple(document_scales)
    if any(scale <= 0 for scale in scales):
        raise ValueError("document scales must be greater than zero")
    if large_paragraphs <= 0:
        raise ValueError("large_paragraphs must be greater than zero")
    if search_iterations <= 0:
        raise ValueError("search_iterations must be greater than zero")

    with TemporaryDirectory(prefix="crawlforge-context-benchmark-") as directory:
        database_root = Path(directory)
        results: list[BenchmarkResult] = []
        for position, scale in enumerate(scales):
            results.append(
                await _run_workload(
                    workload=f"documents_{scale}",
                    pages=_document_pages(scale),
                    query="deterministic context retrieval",
                    database=database_root / f"documents-{position}.db",
                    search_iterations=search_iterations,
                )
            )
        results.append(
            await _run_workload(
                workload="large_html",
                pages=(_large_page(large_paragraphs),),
                query="large document anchor",
                database=database_root / "large-html.db",
                search_iterations=search_iterations,
            )
        )
        return results


async def _run_workload(
    *,
    workload: str,
    pages: Sequence[CrawledPage],
    query: str,
    database: Path,
    search_iterations: int,
) -> BenchmarkResult:
    async with ContextEngine(database) as engine:
        indexing_started = perf_counter()
        indexed = await engine.index_pages(pages)
        indexing_seconds = perf_counter() - indexing_started

        total_search_seconds = 0.0
        hits = []
        for _ in range(search_iterations):
            search_started = perf_counter()
            hits = await engine.search(query, limit=5)
            total_search_seconds += perf_counter() - search_started

        context = await engine.build_context(
            query,
            limit=5,
            token_budget=1_000,
        )

    total_search_milliseconds = total_search_seconds * 1_000
    return {
        "workload": workload,
        "docs": indexed.documents_seen,
        "source_bytes": indexed.source_size_bytes,
        "clean_bytes": indexed.cleaned_size_bytes,
        "chunks": indexed.chunks_indexed,
        "indexing_seconds": indexing_seconds,
        "search_iterations": search_iterations,
        "total_search_milliseconds": total_search_milliseconds,
        "average_search_milliseconds": (total_search_milliseconds / search_iterations),
        "hits": len(hits),
        "estimated_returned_tokens": context.estimated_tokens,
        "estimated_context_reduction": context.estimated_context_reduction,
    }


def _document_pages(count: int) -> tuple[CrawledPage, ...]:
    return tuple(
        _page(
            path=f"documents/{index}",
            title=f"Benchmark document {index}",
            paragraphs=(
                (
                    "Deterministic context retrieval keeps local benchmark "
                    f"document {index} reproducible."
                ),
                (
                    f"Section {index} describes bounded crawling, content "
                    "cleaning, chunk indexing, and lexical search."
                ),
                (
                    "SQLite FTS5 returns relevant chunks without any network "
                    f"dependency; evidence marker {index:04d} is unique."
                ),
            ),
        )
        for index in range(count)
    )


def _large_page(paragraph_count: int) -> CrawledPage:
    paragraphs = tuple(
        (
            "Large document anchor identifies a deterministic section "
            f"{index:04d}. This section explains local processing, chunk "
            "boundaries, lexical retrieval, and reproducible benchmark data. "
            f"Its unique evidence value is {index * 17 + 3}."
        )
        for index in range(paragraph_count)
    )
    return _page(
        path="large-document",
        title="Large benchmark document",
        paragraphs=paragraphs,
    )


def _page(
    *,
    path: str,
    title: str,
    paragraphs: Sequence[str],
) -> CrawledPage:
    url = f"https://benchmark.invalid/{path}"
    rendered_paragraphs = "\n".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    html = f"""<!doctype html>
<html>
  <head>
    <title>{title}</title>
    <link rel="canonical" href="{url}">
    <script>discardedBenchmarkScript()</script>
  </head>
  <body>
    <nav>Discarded benchmark navigation</nav>
    <main>
      <h1>{title}</h1>
      {rendered_paragraphs}
    </main>
    <footer>Discarded benchmark footer</footer>
  </body>
</html>
"""
    return CrawledPage(
        url=url,
        final_url=url,
        html=html,
        status_code=200,
        content_type="text/html; charset=utf-8",
        fetched_at=_FETCHED_AT,
        depth=0,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark and emit one machine-readable JSON list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=DEFAULT_DOCUMENT_SCALES,
        metavar="DOCS",
        help="document counts for the multi-document workloads",
    )
    parser.add_argument(
        "--large-paragraphs",
        type=int,
        default=DEFAULT_LARGE_PARAGRAPHS,
        help="paragraph count for the single large-HTML workload",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=DEFAULT_SEARCH_ITERATIONS,
        help="ready-index searches to time per workload",
    )
    arguments = parser.parse_args(argv)
    payload = asyncio.run(
        run_benchmarks(
            arguments.scales,
            large_paragraphs=arguments.large_paragraphs,
            search_iterations=arguments.search_iterations,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
