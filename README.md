# CrawlForge

CrawlForge is a high-performance asynchronous web crawler for Python. The
project is being built as a composable foundation for reliable, polite, and
observable web crawling.

## Status

The project is in its initial setup stage. It currently provides an importable
package and a command-line interface with help and version output. Crawling is
not implemented yet.

## Planned capabilities

- Asynchronous HTTP fetching and HTML parsing
- URL queueing, concurrency management, and per-host rate limiting
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
