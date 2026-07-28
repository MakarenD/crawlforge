# CrawlForge

[![CI][ci-badge]][ci-workflow]

CrawlForge is a high-performance asynchronous web crawler for Python. The
project is being built as a composable foundation for reliable, polite, and
observable web crawling.

## Status

CrawlForge provides queue-driven website crawling, an importable asynchronous
HTTP client, structured HTML parsing, and a command-line interface with help and
version output. Crawls support depth and page limits, URL filters, duplicate
suppression, global and per-domain concurrency, live progress logging, and
structured success and failure state. Requests support per-domain or global
rate limiting, robots.txt enforcement, configurable delays, User-Agent
rotation, classified failures, configurable retries with exponential backoff,
and retry statistics.

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

## Planned capabilities

- Sitemap discovery
- Pluggable storage and crawl reports
- Configuration files and an extensible CLI

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
mypy
```

GitHub Actions runs the full test suite on Python 3.12, 3.13, and 3.14. Pull
requests also run linting, formatting, type checking, CLI, dependency, and
package-build validation.

## Command-line interface

```bash
python -m crawlforge --help
python -m crawlforge --version
```

The console-script entry point is also available after installation:

```bash
crawlforge --help
```

## License

Distributed under the [MIT License](LICENSE).

[ci-badge]: https://github.com/MakarenD/crawlforge/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/MakarenD/crawlforge/actions/workflows/ci.yml
