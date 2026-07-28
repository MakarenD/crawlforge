"""Compare sequential and asynchronous crawling on a local deterministic site."""

from __future__ import annotations

import argparse
import asyncio
import json
import tracemalloc
from collections.abc import Sequence
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import perf_counter, sleep
from typing import TypedDict
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from crawlforge import AsyncCrawler


class BenchmarkResult(TypedDict):
    """Measurements for one local workload size."""

    pages: int
    synchronous_seconds: float
    asynchronous_seconds: float
    asynchronous_peak_memory_mib: float
    asynchronous_pages_per_second: float


class _PageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        """Return a small deterministic HTML page."""
        sleep(0.005)
        body = f"<title>{self.path}</title><p>benchmark</p>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        """Keep benchmark output machine-readable."""


async def run_benchmark(pages: int) -> BenchmarkResult:
    """Measure a sequential baseline and CrawlForge for one page count."""
    if pages <= 0:
        raise ValueError("pages must be greater than zero")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PageHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    host, port = server.server_address
    urls = [f"http://{host}:{port}/page/{index}" for index in range(pages)]
    try:
        sync_started = perf_counter()
        await asyncio.to_thread(_download_sequentially, urls)
        sync_elapsed = perf_counter() - sync_started

        tracemalloc.start()
        async_started = perf_counter()
        async with AsyncCrawler(
            max_concurrent=min(50, pages),
            max_depth=0,
            respect_robots=False,
            requests_per_second=100_000,
        ) as crawler:
            await crawler.crawl(urls, max_pages=pages)
        async_elapsed = perf_counter() - async_started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        await asyncio.to_thread(_stop_server, server, server_thread)

    return {
        "pages": pages,
        "synchronous_seconds": sync_elapsed,
        "asynchronous_seconds": async_elapsed,
        "asynchronous_peak_memory_mib": peak / (1024 * 1024),
        "asynchronous_pages_per_second": (
            pages / async_elapsed if async_elapsed > 0 else 0.0
        ),
    }


async def run_benchmarks(scales: Sequence[int]) -> list[BenchmarkResult]:
    """Run every requested workload size sequentially."""
    return [await run_benchmark(scale) for scale in scales]


def _download_sequentially(urls: Sequence[str]) -> None:
    for url in urls:
        parsed = urlsplit(url)
        connection = HTTPConnection(parsed.hostname, parsed.port, timeout=10)
        try:
            connection.request("GET", parsed.path)
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"unexpected HTTP status {response.status}")
            BeautifulSoup(body, "lxml").get_text(" ", strip=True)
        finally:
            connection.close()


def _stop_server(server: ThreadingHTTPServer, thread: Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the local 100/500/1000-page benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scales",
        nargs="*",
        type=int,
        default=[100, 500, 1000],
        help="page counts to measure",
    )
    arguments = parser.parse_args(argv)
    print(json.dumps(asyncio.run(run_benchmarks(arguments.scales)), indent=2))


if __name__ == "__main__":
    main()
