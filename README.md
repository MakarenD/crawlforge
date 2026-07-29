# CrawlForge

[![CI](https://github.com/MakarenD/crawlforge/actions/workflows/ci.yml/badge.svg)](https://github.com/MakarenD/crawlforge/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Typing: typed](https://img.shields.io/badge/typing-typed-blue.svg)](src/crawlforge/py.typed)

CrawlForge is a local-first web-context engine for AI agents.

> Crawl once. Search locally. Send bounded context with sources.

AI agents often receive complete HTML pages containing navigation, scripts,
repeated layout blocks, and irrelevant sections. CrawlForge performs
deterministic processing locally and returns a bounded set of source-linked
chunks ranked for relevance. BM25 is the default; optional local semantic
retrieval uses pinned Sentence Transformers embeddings and exact cosine search.
The checked baselines below make both strategies' limitations visible. Python
and the CLI support both strategies, while the local MCP stdio server remains
lexical.

## How it works

```mermaid
flowchart LR
    A[Website] --> B[Async crawler]
    B --> C[Clean and normalize]
    C --> D[Heading-aware chunks]
    D --> E[SQLite FTS5 / BM25]
    D --> H[Optional float32 embeddings]
    H --> I[Exact cosine search]
    E --> F[Token-budgeted context]
    I --> F
    F --> G[Python / CLI / MCP]
```

Unlike a crawler that stops after downloading or extracting pages, CrawlForge
maintains a local retrieval index and selects complete passages under an
explicit context budget. It does not generate answers or send indexed content
to an external retrieval service.

## Key capabilities

- Asynchronous HTTP-first crawling
- Queue, depth, page, and concurrency limits
- robots.txt enforcement, rate limiting, and bounded retries
- Deterministic content cleaning and heading-aware chunking
- Local deduplicated SQLite FTS5/BM25 index
- Optional local Sentence Transformers embeddings and exact cosine retrieval
- Token-budgeted context with source provenance
- Local MCP stdio integration with bounded typed outputs
- Deterministic offline retrieval evaluation

## Quick start

After [installing from source](#installation), index a documentation site:

```bash
uv run crawlforge index https://example.com/docs \
  --database .crawlforge/index.db \
  --max-pages 100 \
  --max-depth 2
```

Search the local index and select complete chunks within the token estimate:

```bash
uv run crawlforge search "How are retries configured?" \
  --database .crawlforge/index.db \
  --limit 5 \
  --token-budget 3000
```

A human-readable result keeps its source and score. For example, abridged:

```text
1. Retry configuration
   URL: https://example.com/docs/retries
   BM25: -2.174630 (lower is more relevant)
   ...
Context: ~420/3000 estimated tokens, 1680 characters, 5 candidates
```

Add `--json` for machine-readable standard output. Progress and diagnostics
remain on standard error.

## Installation

The `crawlforge` package is not currently published on PyPI. Install from a
source checkout:

```bash
git clone https://github.com/MakarenD/crawlforge.git
cd crawlforge
uv sync
```

Include the optional MCP SDK integration when needed:

```bash
uv sync --extra mcp
```

Install the local semantic runtime separately when needed:

```bash
uv sync --extra semantic
```

Editable pip installation is also supported:

```bash
python -m pip install -e .
python -m pip install -e ".[mcp]"
python -m pip install -e ".[semantic]"
```

CrawlForge requires Python 3.12 or newer and a Python SQLite build with FTS5.
The base package never imports the ML stack. On Intel macOS, the current
upstream PyTorch wheels limit the semantic extra to Python 3.12; the base
package remains supported on Python 3.12–3.14.

## Optional semantic retrieval

Build embeddings after the lexical index exists, then opt into semantic search:

```bash
uv run --extra semantic crawlforge embed \
  --database .crawlforge/index.db \
  --device cpu

uv run --extra semantic crawlforge search \
  "How does the crawler avoid overwhelming a host?" \
  --database .crawlforge/index.db \
  --strategy semantic \
  --limit 5
```

The default model is pinned to an immutable
`sentence-transformers/all-MiniLM-L6-v2` revision. Vectors remain in the local
SQLite index as normalized float32 blobs; model files stay in the normal model
cache. See [Semantic retrieval](docs/semantic-retrieval.md) for the Python API,
cache invalidation, storage and latency costs, and score limitations.

## MCP integration

Start the local stdio server from the source checkout:

```bash
uv run crawlforge-mcp \
  --database .crawlforge/index.db
```

A generic stdio client configuration looks like this:

```json
{
  "mcpServers": {
    "crawlforge": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/crawlforge",
        "run",
        "--extra",
        "mcp",
        "crawlforge-mcp",
        "--database",
        "/absolute/path/to/index.db"
      ]
    }
  }
}
```

The server exposes four tools:

- `index_site` crawls and indexes within server-owned limits;
- `search_index` returns BM25-ranked chunks;
- `build_context` returns a bounded source-linked context;
- `get_index_info` reports local index readiness and counts.

MCP runs locally over stdio, and the SQLite database remains under the user's
control. Web content is untrusted data. Public HTTP(S) targets are allowed by
default, while private and non-routable network targets are blocked. See the
[MCP server documentation](docs/mcp.md) for configuration, tool schemas,
limits, and the complete trust model.

## Retrieval evaluation

The checked-in offline benchmark runs the production cleaning, chunking,
indexing, and public search path against graded section-level relevance
judgments. The current baseline contains 10 original HTML documents, 40 stable
sections, 50 indexed chunks, 64 queries, 114 positive judgments, and 8 query
categories.

Aggregate results from the current frozen-dataset
[paired report](reports/bm25-vs-semantic.md):

| Metric | BM25 | Semantic | Delta |
| --- | ---: | ---: | ---: |
| Hit Rate@5 | 96.4% | 98.2% | +1.8 pp |
| Recall@5 | 80.2% | 82.3% | +2.1 pp |
| MRR | 0.8681 | 0.8563 | -0.0118 |
| NDCG@5 | 0.8100 | 0.8102 | +0.0002 |
| Negative no-result accuracy | 12.5% | 0.0% | -12.5 pp |

Selected category MRR:

| Category | BM25 | Semantic | Delta |
| --- | ---: | ---: | ---: |
| Exact term | 1.0000 | 0.8125 | -0.1875 |
| Code symbol | 1.0000 | 0.8438 | -0.1562 |
| Paraphrase | 0.7639 | 0.8542 | +0.0903 |
| Conceptual | 0.7083 | 0.9062 | +0.1979 |
| Ambiguous | 0.6667 | 0.6708 | +0.0042 |

Semantic retrieval improved paraphrase and conceptual queries but regressed
exact terms, code symbols, aggregate MRR, and strict negative-query behavior.
It won 15 queries while BM25 won 21. This small synthetic English dataset is
useful for deterministic regression analysis, not broad external validity.
Neither BM25 nor cosine scores are calibrated confidence, and latency is
machine-dependent.

See the [BM25](reports/bm25-baseline.md) and
[semantic](reports/semantic-baseline.md) reports, the complete
[paired comparison](reports/bm25-vs-semantic.md), and the
[evaluation methodology](docs/retrieval-evaluation.md).

## Python API

`ContextEngine` owns crawling-to-index ingestion, lexical search, bounded
context selection, and SQLite lifecycle:

```python
import asyncio

from crawlforge import ContextEngine


async def main() -> None:
    async with ContextEngine(".crawlforge/index.db") as engine:
        await engine.ingest_url(
            "https://example.com/docs",
            max_pages=100,
            max_depth=2,
        )
        context = await engine.build_context(
            "How are retries configured?",
            limit=10,
            token_budget=3000,
        )

    for hit in context.hits:
        print(hit.chunk.text)
        print(hit.source.url)


asyncio.run(main())
```

`SearchHit.bm25_score` is the raw SQLite FTS5 score; lower values are more
relevant. Semantic results expose descending cosine similarity and model
identity through separate typed models. See
[Web-context architecture](docs/web-context.md) for processing, chunking,
schema, deduplication, migrations, and metrics.

## Standalone crawler

The asynchronous crawler remains a standalone public component:

```python
import asyncio

from crawlforge import AsyncCrawler


async def main() -> None:
    async with AsyncCrawler(
        max_concurrent=10,
        max_concurrent_per_domain=2,
        max_depth=2,
    ) as crawler:
        pages = await crawler.crawl(
            ["https://example.com"],
            max_pages=50,
            same_domain_only=True,
        )

    print(f"Processed: {len(pages)}")
    print(f"Failed: {len(crawler.failed_urls)}")


asyncio.run(main())
```

Sitemaps, JSON configuration, statistics, reports, queue behavior, politeness,
retries, parsing, storage backends, examples, benchmarks, and the full crawler
CLI are documented in [Crawler and storage](docs/crawler.md).

## Security and trust model

- The crawler respects robots.txt and applies configured rate limits and
  retries.
- MCP indexing allows public HTTP(S) targets only by default.
- Literal IPs, DNS answers, and every redirect destination pass SSRF checks.
- Page sizes, robots.txt sizes, crawl duration, page counts, search results,
  context budgets, and serialized output are bounded.
- External page text remains untrusted content rather than application or MCP
  instructions.
- Source URLs and heading paths remain available as provenance.

See [MCP server](docs/mcp.md) for the detailed network policy and operation
limits.

## Current limitations

- Exact semantic scan is linear in chunk count and embedding dimension
- No hybrid retrieval, reranking, or calibrated abstention threshold
- No generated answers
- No JavaScript browser rendering
- Approximate, model-agnostic token estimator
- SQLite FTS5 is required
- Local stdio MCP only
- Semantic inference requires a separately installed and cached model
- Semantic retrieval is not exposed through MCP

## Documentation

- [Web-context architecture](docs/web-context.md)
- [MCP server](docs/mcp.md)
- [Retrieval evaluation](docs/retrieval-evaluation.md)
- [Semantic retrieval](docs/semantic-retrieval.md)
- [Crawler and storage](docs/crawler.md)

## Development

```bash
uv sync --extra dev --extra mcp
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

GitHub Actions runs the base test suite on Python 3.12, 3.13, and 3.14, with
linting, formatting, type checking, CLI, dependency, and package-build checks.
An offline Python 3.12 job installs the semantic extra and runs
controlled-vector tests without downloading a model.

## Roadmap

- Hybrid retrieval
- Reranking
- Calibrated negative-query abstention
- Larger real-world evaluation corpora
- Optional browser rendering

These items describe possible next evaluations and extensions, not committed
release dates.

## License

Distributed under the [MIT License](LICENSE).
