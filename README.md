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
structured success and failure state.

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
  pages-per-second values.

Progress is logged after every completed page at `INFO` level. The complete
demonstration accepts one or more starting URLs, shows live progress, and saves
successful pages, failures, and final statistics as JSON:

```bash
python examples/crawl_site.py https://example.com \
  --max-depth 2 \
  --max-pages 50 \
  --output crawl-output.json
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
Connection and socket-read timeouts can be configured with `connect_timeout`
and `read_timeout`.

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

- `robots.txt` support, retries, and sitemap discovery
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
