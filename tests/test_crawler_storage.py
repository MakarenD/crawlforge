"""Integration tests for crawl-time persistence."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import AsyncCrawler, DataStorage, SQLiteStorage


@asynccontextmanager
async def serve(app: web.Application) -> AsyncIterator[TestServer]:
    """Run an aiohttp application on an ephemeral local port."""
    server = TestServer(app)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


class RecordingStorage(DataStorage):
    """Keep defensive copies of saved records for integration assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, object]] = []
        self.closed = False

    async def save(self, data: dict[str, object]) -> None:
        """Record one independent data snapshot."""
        if self.closed:
            raise RuntimeError("storage is closed")
        self.records.append(copy.deepcopy(data))
        self._saved_count += 1

    async def close(self) -> None:
        """Mark the storage closed."""
        self.closed = True


class FlakyStorage(RecordingStorage):
    """Fail a configured number of save or close attempts."""

    def __init__(self, *, save_failures: int = 0, close_failures: int = 0) -> None:
        super().__init__()
        self.save_failures = save_failures
        self.close_failures = close_failures
        self.save_attempts = 0
        self.close_attempts = 0

    async def save(self, data: dict[str, object]) -> None:
        """Fail early attempts before delegating to the recorder."""
        self.save_attempts += 1
        if self.save_attempts <= self.save_failures:
            raise OSError("temporary write failure")
        await super().save(data)

    async def close(self) -> None:
        """Fail early close attempts before completing cleanup."""
        self.close_attempts += 1
        if self.close_attempts <= self.close_failures:
            raise OSError("temporary close failure")
        await super().close()


class BlockingStorage(RecordingStorage):
    """Expose deterministic barriers around save and close operations."""

    def __init__(self) -> None:
        super().__init__()
        self.save_started = asyncio.Event()
        self.release_save = asyncio.Event()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def save(self, data: dict[str, object]) -> None:
        """Wait until the test explicitly releases the save."""
        self.save_started.set()
        await self.release_save.wait()
        await super().save(data)

    async def close(self) -> None:
        """Wait until the test explicitly releases resource cleanup."""
        self.close_started.set()
        await self.release_close.wait()
        await super().close()


class AmbiguousStorage(RecordingStorage):
    """Report one failure after the record has already been accepted."""

    async def save(self, data: dict[str, object]) -> None:
        """Expose at-least-once behavior after an uncertain custom side effect."""
        await super().save(data)
        if len(self.records) == 1:
            raise OSError("write acknowledgement was lost")


@pytest.mark.asyncio
async def test_crawl_persists_standardized_successful_page_data() -> None:
    """A successful page is saved once with response and crawl metadata."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(
            status=201,
            text=(
                "<title>Stored page</title>"
                '<meta name="description" content="description">'
                '<a href="/next">Next</a>'
                "<p>Body</p>"
            ),
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/", page)
    storage = RecordingStorage()

    async with (
        serve(app) as server,
        AsyncCrawler(
            storage=storage,
            max_depth=0,
            respect_robots=False,
            requests_per_second=1000,
        ) as crawler,
    ):
        url = str(server.make_url("/"))
        next_url = str(server.make_url("/next"))
        pages = await crawler.crawl([url])
        stats = crawler.get_stats()

    assert set(pages) == {url}
    assert storage.closed
    assert len(storage.records) == 1
    assert storage.records[0] == {
        "url": url,
        "title": "Stored page",
        "text": "Next Body",
        "links": [next_url],
        "metadata": {
            "title": "Stored page",
            "description": "description",
            "keywords": "",
        },
        "crawled_at": storage.records[0]["crawled_at"],
        "status_code": 201,
        "content_type": "text/html",
    }
    crawled_at = storage.records[0]["crawled_at"]
    assert isinstance(crawled_at, datetime)
    assert crawled_at.tzinfo is not None
    assert stats["stored"] == 1
    assert stats["storage_errors"] == 0
    assert stats["storage_retries"] == 0


@pytest.mark.asyncio
async def test_crawl_saves_each_page_once_with_requested_redirect_url() -> None:
    """Redirected pages and discovered children retain stable requested identity."""

    async def redirect(_request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/dir/page")

    async def page(_request: web.Request) -> web.Response:
        return web.Response(
            text='<title>Redirected</title><a href="child">Child</a>',
            content_type="text/html",
        )

    async def child(_request: web.Request) -> web.Response:
        return web.Response(text="<title>Child</title>", content_type="text/html")

    app = web.Application()
    app.router.add_get("/old", redirect)
    app.router.add_get("/dir/page", page)
    app.router.add_get("/dir/child", child)
    storage = RecordingStorage()

    async with (
        serve(app) as server,
        AsyncCrawler(
            storage=storage,
            max_depth=1,
            respect_robots=False,
            requests_per_second=1000,
        ) as crawler,
    ):
        old_url = str(server.make_url("/old"))
        child_url = str(server.make_url("/dir/child"))
        pages = await crawler.crawl([old_url], same_domain_only=True)

    assert set(pages) == {old_url, child_url}
    assert [record["url"] for record in storage.records] == [old_url, child_url]
    assert storage.records[0]["links"] == [child_url]
    assert storage.records[0]["status_code"] == 200


@pytest.mark.asyncio
async def test_storage_save_retries_then_recovers_without_losing_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient write failures are retried without failing the crawl."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<p>ok</p>", content_type="text/html")

    async def no_wait(_delay: float) -> None:
        return None

    app = web.Application()
    app.router.add_get("/", page)
    storage = FlakyStorage(save_failures=2)
    crawler = AsyncCrawler(
        storage=storage,
        storage_max_retries=2,
        storage_retry_delay=1,
        respect_robots=False,
        requests_per_second=1000,
    )
    monkeypatch.setattr(crawler, "_storage_sleep", no_wait)

    async with serve(app) as server:
        url = str(server.make_url("/"))
        pages = await crawler.crawl([url])
    stats = crawler.get_stats()
    await crawler.close()

    assert set(pages) == {url}
    assert storage.save_attempts == 3
    assert len(storage.records) == 1
    assert stats["stored"] == 1
    assert stats["storage_retries"] == 2
    assert stats["storage_errors"] == 0


@pytest.mark.asyncio
async def test_final_storage_error_is_logged_and_crawl_continues(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final write failure does not turn a parsed page into a crawl failure."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<p>ok</p>", content_type="text/html")

    async def no_wait(_delay: float) -> None:
        return None

    app = web.Application()
    app.router.add_get("/", page)
    storage = FlakyStorage(save_failures=10)
    crawler = AsyncCrawler(
        storage=storage,
        storage_max_retries=1,
        storage_retry_delay=1,
        respect_robots=False,
        requests_per_second=1000,
    )
    monkeypatch.setattr(crawler, "_storage_sleep", no_wait)

    async with serve(app) as server:
        url = str(server.make_url("/"))
        with caplog.at_level(logging.ERROR, logger="crawlforge.crawler"):
            pages = await crawler.crawl([url])
    stats = crawler.get_stats()
    await crawler.close()

    assert set(pages) == {url}
    assert crawler.failed_urls == {}
    assert stats["stored"] == 0
    assert stats["storage_errors"] == 1
    assert stats["storage_retries"] == 1
    assert f"Could not save crawl data for {url} after 2 attempt(s)" in caplog.text


@pytest.mark.asyncio
async def test_custom_storage_retries_have_documented_at_least_once_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom backend must deduplicate an ambiguous successful side effect."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<p>ok</p>")

    async def no_wait(_delay: float) -> None:
        return None

    app = web.Application()
    app.router.add_get("/", page)
    storage = AmbiguousStorage()
    crawler = AsyncCrawler(
        storage=storage,
        storage_max_retries=1,
        storage_retry_delay=1,
        respect_robots=False,
        requests_per_second=1000,
    )
    monkeypatch.setattr(crawler, "_storage_sleep", no_wait)

    async with serve(app) as server:
        url = str(server.make_url("/"))
        pages = await crawler.crawl([url])
    await crawler.close()

    assert set(pages) == {url}
    assert [record["url"] for record in storage.records] == [url, url]
    assert crawler.get_stats()["stored"] == 1
    assert crawler.get_stats()["storage_retries"] == 1


@pytest.mark.asyncio
async def test_failed_download_is_not_sent_to_storage() -> None:
    """Only successfully downloaded and parsed pages are persisted."""
    app = web.Application()
    storage = RecordingStorage()

    async with (
        serve(app) as server,
        AsyncCrawler(
            storage=storage,
            max_retries=0,
            respect_robots=False,
            requests_per_second=1000,
        ) as crawler,
    ):
        url = str(server.make_url("/missing"))
        pages = await crawler.crawl([url])

    assert pages == {}
    assert url in crawler.failed_urls
    assert storage.records == []


@pytest.mark.asyncio
async def test_fetch_and_parse_does_not_implicitly_persist() -> None:
    """Storage integration is scoped to crawl orchestration."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<p>ok</p>")

    app = web.Application()
    app.router.add_get("/", page)
    storage = RecordingStorage()

    async with (
        serve(app) as server,
        AsyncCrawler(
            storage=storage,
            respect_robots=False,
            requests_per_second=1000,
        ) as crawler,
    ):
        result = await crawler.fetch_and_parse(str(server.make_url("/")))

    assert result["text"] == "ok"
    assert storage.records == []


@pytest.mark.asyncio
async def test_cancellation_during_storage_save_cleans_scheduler_state() -> None:
    """Cancelling persistence propagates and leaves no page task active."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<p>ok</p>")

    app = web.Application()
    app.router.add_get("/", page)
    storage = BlockingStorage()
    crawler = AsyncCrawler(
        storage=storage,
        respect_robots=False,
        requests_per_second=1000,
    )

    async with serve(app) as server:
        task = asyncio.create_task(crawler.crawl([str(server.make_url("/"))]))
        await asyncio.wait_for(storage.save_started.wait(), timeout=2)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    assert crawler.get_stats()["active"] == 0
    assert crawler.queue.get_stats()["active"] == 0
    assert storage.records == []
    storage.release_close.set()
    await crawler.close()


@pytest.mark.asyncio
async def test_close_finishes_storage_cleanup_before_propagating_cancellation() -> None:
    """Crawler close shields both owned resources from caller cancellation."""
    storage = BlockingStorage()
    crawler = AsyncCrawler(storage=storage)
    close_task = asyncio.create_task(crawler.close())
    await asyncio.wait_for(storage.close_started.wait(), timeout=2)
    close_task.cancel()
    storage.release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert storage.closed
    await crawler.close()


@pytest.mark.asyncio
async def test_storage_close_is_retried_before_crawler_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owned storage gets bounded close retries and remains observable."""

    async def no_wait(_delay: float) -> None:
        return None

    storage = FlakyStorage(close_failures=2)
    crawler = AsyncCrawler(
        storage=storage,
        storage_max_retries=2,
        storage_retry_delay=1,
    )
    monkeypatch.setattr(crawler, "_storage_sleep", no_wait)

    await crawler.close()

    assert storage.closed
    assert storage.close_attempts == 3
    assert crawler.get_stats()["storage_retries"] == 2
    assert crawler.get_stats()["storage_errors"] == 0


@pytest.mark.asyncio
async def test_exhausted_storage_close_error_propagates_after_http_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final close failure remains visible after the session is released."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def no_wait(_delay: float) -> None:
        return None

    app = web.Application()
    app.router.add_get("/", page)
    storage = FlakyStorage(close_failures=10)
    crawler = AsyncCrawler(
        storage=storage,
        storage_max_retries=1,
        storage_retry_delay=1,
        respect_robots=False,
        requests_per_second=1000,
    )
    monkeypatch.setattr(crawler, "_storage_sleep", no_wait)

    async with serve(app) as server:
        assert await crawler.fetch_url(str(server.make_url("/"))) == "ok"
        with pytest.raises(OSError, match="temporary close failure"):
            await crawler.close()

    assert crawler._session is None
    assert crawler.get_stats()["storage_retries"] == 1
    assert crawler.get_stats()["storage_errors"] == 1


@pytest.mark.asyncio
async def test_crawler_close_flushes_short_sqlite_batch(tmp_path: Path) -> None:
    """Crawler ownership makes a buffered SQLite result durable on context exit."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<title>Durable</title>")

    app = web.Application()
    app.router.add_get("/", page)
    output = tmp_path / "crawler.sqlite3"
    storage = SQLiteStorage(output, batch_size=100)

    async with (
        serve(app) as server,
        AsyncCrawler(
            storage=storage,
            respect_robots=False,
            requests_per_second=1000,
        ) as crawler,
    ):
        url = str(server.make_url("/"))
        await crawler.crawl([url])
        assert storage.pending_count == 1

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall(
            "SELECT url, title, status_code FROM pages"
        )
    assert rows == [(url, "Durable", 200)]
