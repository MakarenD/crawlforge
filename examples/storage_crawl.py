"""Crawl a local site and demonstrate JSON, CSV, and SQLite persistence."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import aiofiles
import aiosqlite
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import AsyncCrawler, CSVStorage, DataStorage, JSONStorage, SQLiteStorage


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for the storage demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("storage-output"),
        help="directory for generated JSON, CSV, and SQLite files",
    )
    return parser


def build_demo_site() -> web.Application:
    """Create a deterministic two-page website."""

    async def index(_request: web.Request) -> web.Response:
        return web.Response(
            text=(
                "<title>Storage demo</title>"
                '<meta name="description" content="Index page">'
                '<a href="/about">About</a>'
            ),
            content_type="text/html",
        )

    async def about(_request: web.Request) -> web.Response:
        return web.Response(
            text="<title>About</title><p>Persisted asynchronously.</p>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/about", about)
    return app


async def crawl_to_storage(
    root_url: str,
    storage: DataStorage,
) -> dict[str, object]:
    """Crawl the local site once with one configured storage backend."""
    async with AsyncCrawler(
        storage=storage,
        max_depth=1,
        max_concurrent=2,
        respect_robots=False,
        requests_per_second=1000,
    ) as crawler:
        await crawler.crawl([root_url], same_domain_only=True)
        stats: dict[str, object] = dict(crawler.get_stats())
    return stats


async def read_json(path: Path) -> list[dict[str, object]]:
    """Read JSON Lines records produced by the demonstration."""
    async with aiofiles.open(path, encoding="utf-8") as source:
        content = await source.read()
    return [json.loads(line) for line in content.splitlines()]


async def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV records produced by the demonstration."""
    async with aiofiles.open(path, encoding="utf-8", newline="") as source:
        content = await source.read()
    return list(csv.DictReader(io.StringIO(content, newline="")))


async def read_sqlite(path: Path) -> list[dict[str, object]]:
    """Read SQLite records produced by the demonstration."""
    async with aiosqlite.connect(path) as connection:
        rows = await connection.execute_fetchall(
            "SELECT url, title, status_code, content_type FROM pages ORDER BY id"
        )
    return [
        {
            "url": row[0],
            "title": row[1],
            "status_code": row[2],
            "content_type": row[3],
        }
        for row in rows
    ]


async def run_demo(output_dir: Path) -> None:
    """Run all storage backends and print saved-data summaries."""
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    json_path = output_dir / "pages.jsonl"
    csv_path = output_dir / "pages.csv"
    sqlite_path = output_dir / "pages.sqlite3"
    paths = (json_path, csv_path, sqlite_path)
    existing = [
        path
        for path, exists in zip(
            paths,
            await asyncio.gather(*(asyncio.to_thread(path.exists) for path in paths)),
            strict=True,
        )
        if exists
    ]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing demo files: {names}")

    server = TestServer(build_demo_site())
    await server.start_server()
    try:
        root_url = str(server.make_url("/"))
        backends: list[tuple[str, DataStorage]] = [
            ("json", JSONStorage(json_path)),
            ("csv", CSVStorage(csv_path)),
            ("sqlite", SQLiteStorage(sqlite_path, batch_size=2)),
        ]
        stats = {
            name: await crawl_to_storage(root_url, storage)
            for name, storage in backends
        }
    finally:
        await server.close()

    stored_data = {
        "json": await read_json(json_path),
        "csv": await read_csv(csv_path),
        "sqlite": await read_sqlite(sqlite_path),
    }
    summary = {
        "stats": stats,
        "saved_records": {name: len(records) for name, records in stored_data.items()},
        "sample_titles": {
            name: [str(record["title"]) for record in records]
            for name, records in stored_data.items()
        },
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and run the local storage demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arguments = build_parser().parse_args(argv)
    asyncio.run(run_demo(arguments.output_dir))


if __name__ == "__main__":
    main()
