"""Download pages and print structured HTML extraction statistics."""

from __future__ import annotations

import asyncio
import json
import logging

from crawlforge import AsyncCrawler, ParsedPage

URLS = [
    "https://example.com",
    "https://www.python.org",
    "https://docs.aiohttp.org",
]


def summarize(page: ParsedPage) -> dict[str, object]:
    """Build a compact, JSON-serializable page summary."""
    return {
        "url": page["url"],
        "title": page["title"],
        "text_length": len(page["text"]),
        "links_count": len(page["links"]),
        "links": page["links"],
        "images_count": len(page["images"]),
        "headings": page["headings"],
    }


async def main() -> None:
    """Download several pages concurrently and print their summaries."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    async with AsyncCrawler(max_concurrent=3) as crawler:
        pages = await asyncio.gather(
            *(crawler.fetch_and_parse(url) for url in URLS),
        )

    summaries = [summarize(page) for page in pages]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
