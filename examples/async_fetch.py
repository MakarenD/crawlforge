"""Compare sequential and concurrent page downloads with CrawlForge."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from crawlforge import AsyncCrawler

URLS = [
    "https://example.com",
    "https://www.python.org",
    "https://docs.aiohttp.org",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/2",
    "https://httpbin.org/status/404",
]


async def main() -> None:
    """Download example URLs sequentially and concurrently."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    async with AsyncCrawler(max_concurrent=5) as crawler:
        sequential_started = perf_counter()
        for url in URLS:
            await crawler.fetch_url(url)
        sequential_elapsed = perf_counter() - sequential_started

        parallel_started = perf_counter()
        parallel_results = await crawler.fetch_urls(URLS)
        parallel_elapsed = perf_counter() - parallel_started

    print("\nRequest results:")
    for url, content in parallel_results.items():
        status = "OK" if content else "ERROR"
        print(f"  {status:5} {url}")

    print(f"\nSequential: {sequential_elapsed:.2f}s")
    print(f"Parallel:   {parallel_elapsed:.2f}s")
    if parallel_elapsed > 0:
        print(f"Speedup:    {sequential_elapsed / parallel_elapsed:.2f}x")
    print(
        "Successful pages: "
        f"{sum(bool(page) for page in parallel_results.values())}/{len(URLS)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
