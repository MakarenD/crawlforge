# Crawler and storage

The CrawlForge crawler is a standalone public subsystem as well as the input
layer for the local web-context engine. It combines asynchronous HTTP,
queue-driven discovery, bounded concurrency, robots.txt enforcement, rate
limiting, retries, structured parsing, optional persistence, and operational
reports.

## Contents

- [Integrated crawler](#integrated-crawler)
- [Sitemap discovery](#sitemap-discovery)
- [JSON configuration](#json-configuration)
- [Statistics, reports, and progress](#statistics-reports-and-progress)
- [Queue-driven crawling](#queue-driven-crawling)
- [Rate limiting and robots.txt](#rate-limiting-and-robotstxt)
- [Errors, retries, and timeouts](#errors-retries-and-timeouts)
- [Asynchronous HTTP client](#asynchronous-http-client)
- [HTML parsing](#html-parsing)
- [Asynchronous storage](#asynchronous-storage)
- [Performance benchmark](#performance-benchmark)
- [Crawler CLI](#crawler-cli)

## Integrated crawler

`AdvancedCrawler` composes the queue, concurrency, politeness, retry, parsing,
storage, sitemap, configuration, and reporting components. It accepts either a
validated JSON configuration or direct constructor values.

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
are written in worker threads after the crawl, and HTTP, sitemap, and storage
resources are closed together. `close()` is idempotent.

## Sitemap discovery

`SitemapParser.fetch_sitemap()` accepts both `urlset` documents and recursive
`sitemapindex` trees. It handles XML namespaces, preserves first-seen URL
order, deduplicates page and sitemap URLs, and stops index cycles.

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
it uses the crawler's asynchronous fetch path, so rate limits, robots.txt,
retries, redirects, User-Agent selection, and request timeouts remain
consistent.

Traversal is bounded by configurable sitemap depth, sitemap count, URL count,
and per-document byte limits. Invalid XML, unsupported roots, relative
locations, and non-HTTP locations fail with source-aware errors.

Sitemap page URLs become ordinary crawl seeds after the same-domain, include,
and exclude filters are applied. Explicit `urls` remain starting URLs. A failed
sitemap is recorded in `sitemap_failures` without discarding valid seeds from
other sources. If every configured source is empty, invalid, or filtered out,
the crawl fails before page tasks start.

## JSON configuration

Configuration paths are resolved relative to the configuration file. Unknown
options and invalid values fail before network or output resources are opened.
Explicit CLI values override only the options supplied on the command line.

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

`storage.format` can be `json`, `csv`, or `sqlite`. JSON storage also accepts
`json_lines`, `indent`, and `encoding`; CSV accepts `encoding`; SQLite accepts
`batch_size`. Omit `storage` to keep parsed results only in memory. At least one
absolute HTTP(S) URL must appear in `urls`, `sitemaps`, or both.

See the checked example in
[`examples/advanced_config.json`](../examples/advanced_config.json) and its
Python runner in
[`examples/advanced_crawl.py`](../examples/advanced_crawl.py).

## Statistics, reports, and progress

`CrawlerStats` records:

- total, successful, and failed pages;
- average processing speed and elapsed time;
- HTTP status-code distribution;
- the ten busiest domains;
- queued and active work;
- completion percentage and estimated remaining time.

`AsyncCrawler.get_advanced_stats()` exposes the extended snapshot without
changing the established `get_stats()` contract. `AdvancedCrawler.get_stats()`
returns both sets in one mapping.

Every completed page writes an `INFO` progress record with counts,
pages/second, percentage, ETA, and active tasks. `configure_logging()` installs
timestamped console output plus an optional `RotatingFileHandler`; repeated
configuration replaces only handlers owned by CrawlForge.

`AdvancedCrawler.export_to_json()` writes statistics, successful parsed pages,
and failures. `export_to_html_report()` creates a standalone escaped HTML file
with summary cards, status-code and domain charts, and success/failure tables.
`CrawlerStats` also provides statistics-only JSON and HTML exporters.

## Queue-driven crawling

Use `AsyncCrawler` directly when sitemap configuration and result reporting are
not needed:

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
insertion order when priorities match. It suppresses duplicates across queued,
active, processed, and failed states. `SemaphoreManager` applies a global
request limit and an optional per-domain limit while exposing active and peak
request counts.

The crawler maintains a fixed worker window. It waits for the first active task
to complete, processes the result, enqueues discovered links, and refills the
window. The number of queued URLs therefore does not become the number of live
tasks.

Starting URLs are always processed when valid. Discovered links can be:

- restricted to starting hostnames with `same_domain_only`;
- excluded by regular-expression `exclude_patterns`;
- admitted only by regular-expression `include_patterns`.

Patterns match the complete normalized URL. Fragments are removed before
queueing, and each normalized URL is visited at most once. Depth zero contains
only starting URLs. Links found at `max_depth` are not queued. `max_pages`
limits attempted pages exactly; remaining URLs stay visible in queue
statistics.

The crawler exposes:

- `visited_urls` for every attempted URL;
- `processed_urls` for successful structured `ParsedPage` values;
- `failed_urls` for failed URLs and their error descriptions;
- `get_stats()` for page, request, queue, rate, and robots metrics.

The local demonstration accepts starting URLs and saves results as JSON:

```bash
python examples/crawl_site.py https://example.com \
  --max-depth 2 \
  --max-pages 50 \
  --output crawl-output.json
```

## Rate limiting and robots.txt

`AsyncCrawler` checks robots.txt before a requested URL and before every
redirect destination. Rules are cached by origin for the crawler lifetime.
A missing robots.txt file permits crawling. Authorization failures, server
errors, and network failures fail closed after applicable retries. Blocked URLs
are logged, returned as empty strings by `fetch_url()`, and recorded in
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
interval is the greatest of the configured rate interval, `min_delay`, and a
matching robots.txt `Crawl-delay`, plus bounded jitter.

Rate capacity is acquired before a request slot is recorded so cancellation
does not reserve unused time. Redirect hops and robots.txt requests follow the
same scheduling and network-policy path as ordinary pages.

Pass `user_agents` to rotate a sequence in round-robin order. One User-Agent is
selected for the complete logical request, including robots checks, redirects,
and retries. `RateLimiter` and `RobotsParser` are public for applications that
need these controls independently.

The self-contained demonstration runs against a local site:

```bash
python examples/polite_crawl.py
```

## Errors, retries, and timeouts

`RetryStrategy` runs an asynchronous callable with bounded retries. By default,
it retries `TransientError` and `NetworkError`; `PermanentError` and
`ParseError` fail immediately. Per-type retry limits and backoff factors can be
configured independently.

```python
from crawlforge import NetworkError, RetryStrategy, TransientError

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
returns totals by type, total and successful retries, average scheduled retry
delay, and URLs with permanent errors.

HTTP 401, 403, and 404 responses are permanent. HTTP 429, 500, 502, 503, and
504 responses are transient. Timeouts are transient, connection and DNS
failures are network errors, and decoding or parsing failures are parse errors.
HTTP 500 is limited to one retry. HTTP 429 uses a larger backoff and honors a
valid `Retry-After` value as a lower bound. Response resources and concurrency
permits are released before waiting.

Connection setup, socket reads, and the complete request attempt have separate
timeouts:

```python
crawler = AsyncCrawler(
    connect_timeout=5.0,
    read_timeout=20.0,
    total_timeout=30.0,
    timeout_backoff_factor=1.5,
)
```

The timeout budget is multiplied by `timeout_backoff_factor` on each retry.
Cancellation always propagates and active child tasks are awaited during
cleanup.

The local error demonstration covers 429, 404, 500, and 503 responses:

```bash
python examples/error_retries.py --output error-report.json
```

## Asynchronous HTTP client

```python
import asyncio

from crawlforge import AsyncCrawler


async def main() -> None:
    urls = ["https://example.com", "https://www.python.org"]

    async with AsyncCrawler(max_concurrent=5) as crawler:
        pages = await crawler.fetch_urls(urls)

    for url, content in pages.items():
        print(url, len(content))


asyncio.run(main())
```

`AsyncCrawler` creates one `aiohttp.ClientSession` lazily and reuses it across
requests. `max_concurrent` bounds active downloads and the connector pool.

Handled HTTP errors, timeouts, and `aiohttp` client errors produce an empty
string for that URL so the remaining batch can finish. A successful empty body
has the same compatibility representation; applications that need the
distinction should use crawl statistics or diagnostics. Cancellation is never
converted into an empty result.

Use the asynchronous context manager when possible. For manual lifecycle
management, call `await crawler.close()` in a `finally` block.

## HTML parsing

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

`fetch_and_parse()` returns a stable mapping containing `url`, `title`, `text`,
`links`, `metadata`, `images`, `headings`, `tables`, and `lists`. Relative
links and image sources resolve against the final response URL; the result
keeps the requested URL. Empty values, unsupported schemes, invalid URLs, and
duplicate links are excluded, and URL fragments are removed.

The parser recovers available data from malformed HTML. If one extractor
fails, it logs a warning and returns the fields it could produce. A download
failure uses the same result shape with empty extracted values.

```bash
python examples/async_fetch.py
python examples/parse_pages.py
```

These examples use public websites and are intended for manual use. Automated
tests use only ephemeral local HTTP servers and in-memory HTML fixtures.

## Asynchronous storage

Pass a storage backend to `AsyncCrawler` to persist each successfully crawled
page when parsing finishes.

```python
import asyncio

from crawlforge import AsyncCrawler, JSONStorage, SQLiteStorage


async def main() -> None:
    async with AsyncCrawler(storage=JSONStorage("results.jsonl")) as crawler:
        await crawler.crawl(["https://example.com"])

    database = SQLiteStorage("crawler.db", batch_size=100)
    crawler = AsyncCrawler(storage=database)
    try:
        await crawler.crawl(["https://example.com"])
    finally:
        await crawler.close()


asyncio.run(main())
```

`JSONStorage` writes one complete object per line by default. Set
`json_lines=False, indent=2` for a formatted JSON array. `CSVStorage` derives
columns from the standard record, applies Python CSV quoting, accepts a custom
encoding, and JSON-encodes nested links and metadata.

`SQLiteStorage` creates a `pages` table lazily, indexes URL and crawl timestamp,
and uses configurable `executemany()` batches. A stable record hash
deduplicates uncertain retries while retaining later crawls of the same URL as
separate records. `close()` commits a short final batch.

Every stored record contains:

- `url`, `title`, `text`, and `links`;
- metadata;
- a timezone-aware `crawled_at`;
- final HTTP `status_code` and normalized `content_type`.

Writes are serialized per backend. SQLite batches are protected from concurrent
mutation. Storage cancellation propagates. Other write errors use bounded
retries and are reported after the final attempt. An exhausted storage-close
error propagates after the HTTP session is released so an uncommitted final
batch cannot look successful.

Retries have at-least-once semantics for custom storage implementations. A
backend that completes a side effect before reporting an error must
deduplicate the repeated `save()` call when duplicates are unacceptable.
The built-in SQLite backend uses record-level idempotency keys.

Run the local storage demonstration with an empty output directory:

```bash
python examples/storage_crawl.py --output-dir storage-output
```

## Performance benchmark

The deterministic crawler benchmark starts a local threaded HTTP server with a
fixed 5 ms response delay, compares sequential fetch-and-parse work with
`AsyncCrawler`, and measures asynchronous peak memory with `tracemalloc`.
Default workloads contain 100, 500, and 1000 unique pages.

```bash
python examples/performance_benchmark.py
```

The script reports measurements rather than enforcing machine-specific timing
thresholds. Concurrency correctness and bounded scheduling are covered by
event-driven automated tests.

## Crawler CLI

Run the integrated crawler directly:

```bash
crawlforge \
  --urls https://example.com \
  --max-pages 100 \
  --max-depth 2 \
  --output results.json \
  --respect-robots \
  --rate-limit 2
```

Or load the complete JSON configuration:

```bash
crawlforge --config examples/advanced_config.json
```

At least `--urls` or `--config` is required to start a crawl. Without either,
the command prints help and exits successfully.

| Option | Purpose |
| --- | --- |
| `-h`, `--help` | Show help and exit. |
| `--version` | Show the installed CrawlForge version and exit. |
| `--urls URL [URL ...]` | Set one or more starting URLs. |
| `--max-pages INTEGER` | Limit attempted pages. |
| `--max-depth INTEGER` | Limit discovered-link depth. |
| `--max-concurrent INTEGER` | Limit active requests. |
| `--output PATH` | Write the JSON result report. |
| `--html-report PATH` | Write the standalone HTML report. |
| `--config PATH` | Load JSON configuration. |
| `--respect-robots` | Enable robots.txt enforcement. |
| `--no-respect-robots` | Disable robots.txt enforcement. |
| `--rate-limit FLOAT` | Set the request rate limit. |
| `--log-file PATH` | Write rotating diagnostic logs. |
| `--log-level LEVEL` | Use `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

The same executable also provides:

- `crawlforge index` and `crawlforge search`, documented in
  [Web-context architecture](web-context.md);
- `crawlforge evaluate`, documented in
  [Retrieval evaluation](retrieval-evaluation.md).

Use `crawlforge <command> --help` for the exact installed command syntax.
