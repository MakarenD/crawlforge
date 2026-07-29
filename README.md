# CrawlForge

[![CI][ci-badge]][ci-workflow]

CrawlForge is a local HTTP-first web-context engine with BM25 retrieval and an
MCP adapter for AI agents. Its asynchronous crawler remains available as a
composable foundation for reliable, polite, and observable web crawling.

## Status

CrawlForge provides queue-driven website crawling, an importable asynchronous
HTTP client, structured HTML parsing, and a command-line interface with help and
version output. Crawls support depth and page limits, URL filters, duplicate
suppression, global and per-domain concurrency, live progress logging, and
structured success and failure state. Requests support per-domain or global
rate limiting, robots.txt enforcement, configurable delays, User-Agent
rotation, classified failures, configurable retries with exponential backoff,
retry statistics, and asynchronous JSON, CSV, or SQLite persistence.
The integrated crawler also supports recursive sitemap indexes, validated JSON
configuration, advanced status and domain statistics, JSON and standalone HTML
reports, rotating file logs, live percentage/speed/ETA reporting, and a
production command-line entry point.
CrawlForge can also clean successful pages, split them into stable
heading-aware chunks, index them in local SQLite FTS5, retrieve BM25-ranked
fragments, and build a source-linked context under an approximate token budget.
The local stdio MCP adapter exposes those same application-service operations
through four bounded tools.

## Local web-context retrieval

The web-context layer reduces raw page noise before retrieval. It removes
scripts, styles, and conservative navigation boilerplate; preserves headings,
lists, links, code, Unicode, and simple tables; and produces plain text plus a
minimal normalized Markdown representation. The resulting chunks are stored in
a transactional, deduplicated SQLite FTS5 index.

Create or update a local index:

```bash
crawlforge index https://example.com/docs \
  --database .crawlforge/index.db \
  --max-pages 100 \
  --max-depth 2
```

Search and select complete chunks within a budget:

```bash
crawlforge search "How are retries configured?" \
  --database .crawlforge/index.db \
  --limit 5 \
  --token-budget 3000
```

Add `--json` to either command for machine-readable standard output. Diagnostic
logs remain on standard error.

The same operations are available through the application service used by the
CLI:

```python
import asyncio

from crawlforge import ContextEngine


async def retrieve() -> None:
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


asyncio.run(retrieve())
```

`SearchHit.bm25_score` is the raw SQLite FTS5 score: lower values are more
relevant. The default token estimator is a deterministic character heuristic,
not the tokenizer of a particular model. Consequently,
`estimated_context_reduction` is an engineering estimate rather than a measured
model-specific token saving.

The default database is `.crawlforge/index.db`. Remove that selected local file
to clear the index. Detailed architecture, schema, metrics, cleanup, chunking,
and limitations are documented in
[`docs/web-context.md`](docs/web-context.md).

## Local MCP adapter

The MCP workflow is deliberately small:

```text
crawl/index once -> query locally many times -> return bounded relevant context
```

Install the optional official MCP SDK integration and start the stdio server:

```bash
uv sync --extra mcp
uv run crawlforge-mcp --database .crawlforge/index.db
```

The server exposes only `index_site`, `search_index`, `build_context`, and
`get_index_info`. It owns one `ContextEngine` and a fixed database path for its
lifecycle. Tool calls cannot select a local file, execute SQL, relax server
caps, or enable private-network access. `index_site` makes real network
requests, obeys robots.txt, and blocks private or non-routable targets by
default, including redirect destinations and resolved DNS addresses. Server
startup also bounds page and robots body sizes, individual request attempts,
and total crawl duration.

MCP runs locally over stdio; the SQLite database remains under the user's
control. Retrieval is lexical BM25, token counts are approximate, and retrieved
website text is untrusted external content rather than server instructions.
Keep result URLs as provenance. Installation, client configuration, tools,
security policy, and Inspector verification are documented in
[`docs/mcp.md`](docs/mcp.md).

This stage does not include embeddings, a vector database, hybrid search,
reranking, external model calls, generated answers, browser rendering, remote
MCP hosting, or background crawl jobs.

## Integrated crawler

`AdvancedCrawler` composes the existing queue, concurrency, politeness, retry,
parsing, and storage components. It adds sitemap discovery, configuration,
reporting, and a no-argument `crawl()` operation:

```python
import asyncio

from crawlforge import AdvancedCrawler


async def main() -> None:
    crawler = AdvancedCrawler.from_config("crawler.json")
    try:
        await crawler.crawl()
        stats = crawler.get_stats()
        print(f"Processed: {stats['total_pages']} pages")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        crawler.export_to_html_report("report.html")
    finally:
        await crawler.close()


asyncio.run(main())
```

The asynchronous context manager is also supported. Configured result reports
are written in worker threads after the crawl, and all HTTP, sitemap, and
storage resources are closed together.

## Sitemap support

`SitemapParser.fetch_sitemap()` accepts both `urlset` documents and recursive
`sitemapindex` trees. It handles XML namespaces, preserves first-seen URL order,
deduplicates pages and sitemap documents, and stops index cycles.

```python
import asyncio

from crawlforge import SitemapParser


async def discover() -> None:
    async with SitemapParser() as parser:
        urls = await parser.fetch_sitemap(
            "https://example.com/sitemap.xml",
        )
    print(len(urls))


asyncio.run(discover())
```

The standalone parser owns a lazy `aiohttp` session. Inside `AdvancedCrawler`,
it instead uses the crawler's asynchronous fetch path, so rate limits,
robots.txt rules, retries, redirects, User-Agent selection, and request
timeouts remain consistent. Traversal is bounded by configurable depth,
sitemap count, URL count, and per-document byte limits. Invalid XML, unsupported
roots, relative locations, and non-HTTP locations fail with source-aware
errors.

Sitemap page URLs are regular crawl seeds after they pass configured
same-domain, include, and exclude filters. Explicit `urls` remain starting URLs
and preserve the existing `AsyncCrawler` behavior. A failed sitemap is recorded
in `sitemap_failures` without discarding seeds from other sources. If every
configured source is empty, invalid, or filtered out, the crawl fails before
starting page tasks.

## JSON configuration

Configuration paths are resolved relative to the configuration file. Unknown
options and invalid values fail before network or output resources are opened.
Command-line values override only the options explicitly supplied:

```json
{
  "urls": ["https://example.com/"],
  "sitemaps": ["https://example.com/sitemap.xml"],
  "crawler": {
    "max_pages": 100,
    "max_depth": 2,
    "max_concurrent": 10,
    "max_concurrent_per_domain": 2,
    "rate_limit": 2.0,
    "rate_limit_per_domain": true,
    "respect_robots": true,
    "min_delay": 0.0,
    "jitter": 0.0,
    "max_retries": 2,
    "connect_timeout": 10.0,
    "read_timeout": 30.0,
    "total_timeout": 60.0
  },
  "filters": {
    "same_domain_only": true,
    "include": ["/docs/"],
    "exclude": ["/private/"]
  },
  "storage": {
    "format": "json",
    "path": "pages.jsonl",
    "json_lines": true
  },
  "logging": {
    "level": "INFO",
    "file": "crawlforge.log",
    "max_bytes": 5000000,
    "backup_count": 3
  },
  "reports": {
    "json": "results.json",
    "html": "report.html"
  }
}
```

The `storage.format` value can be `json`, `csv`, or `sqlite`. JSON storage also
accepts `json_lines`, `indent`, and `encoding`; CSV accepts `encoding`; SQLite
accepts `batch_size`. Omit `storage` to keep results only in memory. Either
`urls`, `sitemaps`, or both must contain at least one absolute HTTP URL.

The complete checked example is
[`examples/advanced_config.json`](examples/advanced_config.json), with a Python
runner in [`examples/advanced_crawl.py`](examples/advanced_crawl.py).

## Statistics, reports, and live progress

`CrawlerStats` records total, successful, and failed pages; average processing
speed; elapsed time; HTTP status-code distribution; and the ten busiest
domains. Live snapshots also include completion percentage, estimated remaining
time, queued pages, and active page tasks. `AsyncCrawler.get_advanced_stats()`
exposes these metrics without changing the established `get_stats()` contract.
`AdvancedCrawler.get_stats()` returns both sets in one mapping.

Every completed page writes an `INFO` progress record containing counts,
pages/second, percentage, ETA, and active tasks. `configure_logging()` installs
timestamped console output plus an optional `RotatingFileHandler`; repeated
configuration replaces only handlers owned by CrawlForge.

`AdvancedCrawler.export_to_json()` writes statistics, successful parsed pages,
and failures. `export_to_html_report()` creates a standalone escaped HTML file
with summary cards, status-code and domain bar charts, and success/failure
tables. `CrawlerStats` also provides statistics-only JSON and HTML exporters.

## Queue-driven website crawling

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
            exclude_patterns=[r"/private(?:/|$)"],
        )

    print(f"Processed: {len(pages)} pages")
    print(f"Failed: {len(crawler.failed_urls)} pages")


asyncio.run(main())
```

`CrawlerQueue` orders URLs by descending integer priority and preserves
insertion order when priorities match. It ignores duplicate URLs across queued,
active, processed, and failed states. `SemaphoreManager` applies both a global
request limit and an optional per-domain limit while exposing active and peak
request counts.

`AsyncCrawler.crawl()` always processes valid starting URLs. Discovered links
can be restricted to the starting hostnames with `same_domain_only`, excluded
by `exclude_patterns`, or admitted only by `include_patterns`. Patterns are
regular expressions matched against the complete normalized URL. Fragments are
removed before queueing, and each normalized URL is visited at most once.

Depth zero contains only the starting URLs. Links found on a page at
`max_depth` are not queued. `max_pages` limits attempted pages exactly; any
remaining URLs stay visible in queue statistics.

The crawler exposes:

- `visited_urls`, containing every attempted URL;
- `processed_urls`, mapping successful URLs to structured `ParsedPage` data;
- `failed_urls`, mapping failed URLs to error descriptions;
- `get_stats()`, reporting processed, queued, active, failed, visited, and
  page and request throughput, average request delay, and robots.txt blocks.

Progress is logged after every completed page at `INFO` level. The complete
demonstration accepts one or more starting URLs, shows live progress, and saves
successful pages, failures, and final statistics as JSON:

```bash
python examples/crawl_site.py https://example.com \
  --max-depth 2 \
  --max-pages 50 \
  --output crawl-output.json
```

## Polite request controls

`AsyncCrawler` checks robots.txt before each requested URL and each redirect
destination. Rules are cached by origin for the crawler lifetime. A missing
robots.txt file permits crawling; authorization failures, server errors, and
network failures fail closed. Temporary robots.txt server, timeout, and network
failures use the configured retry strategy before the denial is cached. Blocked
URLs are logged, returned as empty strings by `fetch_url()`, and recorded in
`failed_urls` during a crawl.

```python
crawler = AsyncCrawler(
    max_concurrent=5,
    requests_per_second=2.0,
    respect_robots=True,
    min_delay=0.5,
    jitter=0.2,
    user_agent="MyBot/1.0",
)
```

The default request limit is one request per second per domain. Set
`rate_limit_per_domain=False` to share one global schedule. The effective
interval is the greatest of the configured rate interval, `min_delay`, and the
matching robots.txt `Crawl-delay`, plus a random value from zero through
`jitter`.

Temporary network failures and HTTP 408, 429, 500, 502, 503, and 504 responses
are retried up to `max_retries`. Backoff starts at `backoff_base`, doubles after
each failed attempt, and is capped by `backoff_max`. HTTP 500 is limited to one
retry, while 429 uses a larger backoff and honors a valid `Retry-After` value as
a lower bound. Concurrency permits and response resources are released before
the backoff wait.

Pass `user_agents` to rotate a sequence in round-robin order. One User-Agent is
selected for the complete logical request, including redirect checks and
retries, so robots.txt evaluation and request headers remain consistent.

`RateLimiter` and `RobotsParser` are also public for applications that need the
politeness controls independently. `get_stats()` exposes measured
`requests_per_second`, `average_request_delay`, and `robots_blocked` values.

The self-contained demonstration starts a local site, obeys its crawl delay,
and shows a disallowed URL being blocked without depending on the public
internet:

```bash
python examples/polite_crawl.py
```

## Error handling and retries

`RetryStrategy` runs any asynchronous callable with bounded retries. By
default, it retries `TransientError` and `NetworkError`; `PermanentError` and
`ParseError` fail immediately. Per-type retry limits and backoff factors can be
configured independently:

```python
from crawlforge import (
    NetworkError,
    RetryStrategy,
    TransientError,
)

retry_strategy = RetryStrategy(
    max_retries=3,
    backoff_factor=1.0,
    retry_on=[TransientError, NetworkError],
    retry_limits={TransientError: 3, NetworkError: 2},
    backoff_factors={TransientError: 1.0, NetworkError: 0.5},
)
```

Pass the strategy to `AsyncCrawler(retry_strategy=retry_strategy)` to replace
the compatibility settings `max_retries`, `backoff_base`, and `backoff_max`.
Each failed attempt is stored in `error_history` with its type, URL, HTTP
status, attempt number, retry decision, and next delay. `get_error_stats()`
returns error totals by type, total and successful retries, average scheduled
retry delay, and URLs with permanent errors. The same snapshot is available as
`get_stats()["errors"]`.

HTTP 401, 403, and 404 responses are classified as permanent. HTTP 429, 500,
502, 503, and 504 responses are transient. Timeouts are transient failures,
connection and DNS failures are network errors, and decoding or parsing
failures are parse errors. Task cancellation always propagates.

Connection, socket-read, and total timeouts can be configured independently:

```python
crawler = AsyncCrawler(
    connect_timeout=5.0,
    read_timeout=20.0,
    total_timeout=30.0,
    timeout_backoff_factor=1.5,
)
```

The timeout budget is multiplied by `timeout_backoff_factor` on every retry.
The initial attempt uses the configured values unchanged.

The error demonstration starts a local server with 429, 404, 500, and 503
responses, shows automatic recovery, prints retry statistics, and
asynchronously saves a JSON report:

```bash
python examples/error_retries.py --output error-report.json
```

## Asynchronous HTTP client

```python
import asyncio

from crawlforge import AsyncCrawler


async def main() -> None:
    urls = [
        "https://example.com",
        "https://www.python.org",
    ]

    async with AsyncCrawler(max_concurrent=5) as crawler:
        pages = await crawler.fetch_urls(urls)

    for url, content in pages.items():
        print(url, len(content))


asyncio.run(main())
```

`AsyncCrawler` creates its `aiohttp.ClientSession` lazily and reuses it across
requests. `max_concurrent` limits both active downloads and the connector pool.
Connection, socket-read, and total timeouts can be configured with
`connect_timeout`, `read_timeout`, and `total_timeout`.

HTTP errors, timeouts, and other `aiohttp` client errors are logged and produce
an empty string for the affected URL, allowing the remaining batch to finish.
Because an empty successful response has the same representation, applications
that need richer result metadata should use the log records to distinguish the
outcome. Task cancellation is not converted into an empty result.

Use the asynchronous context manager when possible. For manual lifecycle
management, always call `await crawler.close()` in a `finally` block.

## HTML parsing and data extraction

`HTMLParser` extracts page text, metadata, absolute HTTP links, images,
`h1`–`h3` headings, tables, and ordered or unordered lists. Parsing runs in a
worker thread so malformed or large documents do not block the event loop.

```python
import asyncio

from crawlforge import AsyncCrawler


async def main() -> None:
    async with AsyncCrawler() as crawler:
        page = await crawler.fetch_and_parse("https://example.com")

    print(page["title"])
    print(page["links"])
    print(page["metadata"])


asyncio.run(main())
```

`fetch_and_parse()` returns a stable dictionary with `url`, `title`, `text`,
`links`, `metadata`, `images`, `headings`, `tables`, and `lists`. Relative links
and image sources are resolved against the final response URL after redirects,
while the result's `url` field preserves the requested URL. Empty values,
unsupported schemes, invalid URLs, and duplicate links are excluded; URL
fragments are removed. External HTTP links are retained.

The parser recovers available data from malformed HTML. If document creation or
one extractor fails, it logs a warning and returns the fields it could produce.
A download failure uses the same result shape with empty extracted values.

The HTTP example compares sequential and concurrent downloads and reports the
status of each request:

```bash
python examples/async_fetch.py
```

The parsing example downloads several pages concurrently and prints titles,
links, headings, and extraction statistics as JSON:

```bash
python examples/parse_pages.py
```

The examples access public websites and are intended for manual use. Automated
tests run only against ephemeral local HTTP servers and in-memory HTML fixtures.

## Asynchronous data storage

Pass a storage backend to `AsyncCrawler` to persist every successfully crawled
page as soon as parsing finishes:

```python
import asyncio

from crawlforge import AsyncCrawler, JSONStorage, SQLiteStorage


async def main() -> None:
    json_storage = JSONStorage("results.jsonl")
    async with AsyncCrawler(storage=json_storage) as crawler:
        await crawler.crawl(["https://example.com"])

    database = SQLiteStorage("crawler.db", batch_size=100)
    crawler = AsyncCrawler(storage=database)
    try:
        await crawler.crawl(["https://example.com"])
    finally:
        await crawler.close()


asyncio.run(main())
```

`JSONStorage` writes one complete object per line by default, allowing large
outputs to be processed incrementally. Set `json_lines=False, indent=2` for a
formatted JSON array. `CSVStorage` determines its columns from the standardized
record, quotes delimiters and line breaks with Python's CSV rules, accepts a
custom text encoding, and JSON-encodes nested links and metadata.

`SQLiteStorage` creates a `pages` table lazily, indexes URL and crawl timestamp
columns, and uses configurable `executemany()` batches. A stable hash of the
complete record deduplicates uncertain retries while retaining later crawls of
the same URL as separate rows. A short final batch is committed by `close()`.
The common `DataStorage` interface can be implemented for additional
destinations with asynchronous `save()` and `close()` methods.

Every stored record contains:

- `url`, `title`, `text`, and `links`;
- `metadata`;
- timezone-aware `crawled_at`;
- the final HTTP `status_code` and normalized `content_type`.

File writes are serialized per backend, and SQLite batches are protected from
concurrent mutation. Storage cancellation propagates normally. Other write
errors use bounded exponential-backoff retries, are logged after the last
attempt, and do not turn an otherwise successful page into a crawl failure.
`get_stats()` reports `stored`, `storage_retries`, and `storage_errors`.
`AsyncCrawler.close()` flushes and closes the configured storage along with the
HTTP session. An exhausted storage close error propagates after the HTTP session
has been released so an uncommitted final batch cannot be mistaken for success.
Direct `fetch_url()` and `fetch_and_parse()` calls do not persist data
automatically.

Retries have at-least-once semantics for custom storage implementations. A
backend that completes a side effect and then reports an error must deduplicate
the repeated `save()` call when duplicate output is unacceptable. The built-in
SQLite backend handles this with record-level idempotency keys.

The self-contained demonstration crawls a local two-page site three times,
writes each supported format, reads the records back, and prints saved counts
and titles. Use an empty output directory; the script refuses to overwrite its
three generated data files:

```bash
python examples/storage_crawl.py --output-dir storage-output
```

## Performance benchmark

The deterministic benchmark starts a local threaded HTTP server with a fixed
5 ms response delay, compares sequential fetch-and-parse work with
`AsyncCrawler`, and measures asynchronous peak memory with `tracemalloc`. The
default workloads contain 100, 500, and 1000 unique pages:

```bash
python examples/performance_benchmark.py
```

It reports measurements instead of enforcing machine-specific timing
thresholds. Concurrency correctness and bounded scheduling remain covered by
event-driven automated tests.

## Requirements

- Python 3.12 or newer

## Development setup

```bash
git clone <repository-url>
cd crawlforge
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Development commands

Run the test suite:

```bash
python -m pytest
```

Run linting and formatting checks:

```bash
ruff check .
ruff format --check .
```

Run static type checking:

```bash
mypy src
```

GitHub Actions runs the full test suite on Python 3.12, 3.13, and 3.14. Pull
requests also run linting, formatting, type checking, CLI, dependency, and
package-build validation.

## Command-line interface

```bash
python -m crawlforge --help
python -m crawlforge --version
python -m crawlforge \
  --urls https://example.com \
  --max-pages 100 \
  --max-depth 2 \
  --output results.json \
  --respect-robots \
  --rate-limit 2
```

The console-script entry point is also available after installation:

```bash
crawlforge --help
crawlforge --config examples/advanced_config.json
```

The crawl command accepts `--urls`, `--max-pages`, `--max-depth`,
`--max-concurrent`, `--output`, `--html-report`, `--config`,
`--respect-robots`/`--no-respect-robots`, `--rate-limit`, `--log-file`, and
`--log-level`. At least `--urls` or `--config` is required to start work.
Without either, the command prints help and exits successfully.

## License

Distributed under the [MIT License](LICENSE).

[ci-badge]: https://github.com/MakarenD/crawlforge/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/MakarenD/crawlforge/actions/workflows/ci.yml
