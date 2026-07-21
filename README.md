# CrawlForge

CrawlForge is a high-performance asynchronous web crawler for Python. The
project is being built as a composable foundation for reliable, polite, and
observable web crawling.

## Status

CrawlForge provides an importable asynchronous HTTP client, structured HTML
parsing, and a command-line interface with help and version output. The client
supports pooled connections, configurable connection and read timeouts, bounded
concurrency, and request lifecycle logging.

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
and image sources are resolved against the requested URL. Empty values,
unsupported schemes, invalid URLs, and duplicate links are excluded; URL
fragments are removed. External HTTP links are retained.

The parser recovers available data from malformed HTML. If document creation or
one extractor fails, it logs a warning and returns the fields it could produce.
A download failure uses the same result shape with empty extracted values.

The example script compares sequential and concurrent downloads and reports
the status of each request:

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

- URL queueing and per-host rate limiting
- `robots.txt` support, retries, and sitemap discovery
- Pluggable storage, runtime statistics, and reports
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
