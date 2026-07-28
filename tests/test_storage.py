"""Tests for asynchronous crawl data storage backends."""

from __future__ import annotations

import asyncio
import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

from crawlforge import CSVStorage, JSONStorage, SQLiteStorage


def crawl_data(
    url: str = "https://example.com/",
    *,
    title: str = 'Example, "quoted"\npage',
) -> dict[str, object]:
    """Build one complete standardized storage record."""
    return {
        "url": url,
        "title": title,
        "text": "Unicode text: Привет 🌊\nsecond line",
        "links": [f"{url}next", "https://example.org/?a=1&b=2"],
        "metadata": {
            "description": 'A "quoted", multilingual value',
            "nested": {"language": "ru"},
        },
        "crawled_at": datetime(2026, 7, 28, 12, 30, tzinfo=UTC),
        "status_code": 200,
        "content_type": "text/html",
    }


@pytest.mark.asyncio
async def test_json_storage_streams_json_lines_and_preserves_data(
    tmp_path: Path,
) -> None:
    """JSON Lines records round-trip without loading the complete output."""
    output = tmp_path / "pages.jsonl"
    storage = JSONStorage(output)

    await storage.save(crawl_data())
    await storage.save(crawl_data("https://example.com/second"))
    await storage.close()

    rendered = await asyncio.to_thread(output.read_text, encoding="utf-8")
    records = [json.loads(line) for line in rendered.splitlines()]
    assert [record["url"] for record in records] == [
        "https://example.com/",
        "https://example.com/second",
    ]
    assert records[0]["text"] == "Unicode text: Привет 🌊\nsecond line"
    assert records[0]["metadata"]["nested"] == {"language": "ru"}
    assert records[0]["crawled_at"] == "2026-07-28T12:30:00+00:00"
    assert storage.saved_count == 2


@pytest.mark.asyncio
async def test_json_storage_supports_formatted_json_array(tmp_path: Path) -> None:
    """Formatted mode produces one valid, indented JSON array."""
    output = tmp_path / "pages.json"
    storage = JSONStorage(output, json_lines=False, indent=2)

    await storage.save(crawl_data())
    await storage.close()

    rendered = await asyncio.to_thread(output.read_text, encoding="utf-8")
    assert rendered.startswith("[\n  {")
    assert json.loads(rendered)[0]["url"] == "https://example.com/"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("json_lines", "indent"),
    [
        (True, None),
        (False, 2),
    ],
)
async def test_json_storage_rolls_back_partial_record_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_lines: bool,
    indent: int | None,
) -> None:
    """A short failed write leaves no fragment before the record is retried."""
    output_path = tmp_path / "partial.json"
    storage = JSONStorage(output_path, json_lines=json_lines, indent=indent)
    async with storage._lock:
        output = await storage._ensure_open_locked()
    original_write = output.write
    failed = False

    async def fail_after_partial_write(value: str) -> int:
        nonlocal failed
        if not failed:
            failed = True
            await original_write(value[: max(1, len(value) // 2)])
            raise OSError("partial write failed")
        return await original_write(value)

    monkeypatch.setattr(output, "write", fail_after_partial_write)

    with pytest.raises(OSError, match="partial write failed"):
        await storage.save(crawl_data())

    assert storage.saved_count == 0
    await storage.save(crawl_data())
    await storage.close()

    rendered = await asyncio.to_thread(output_path.read_text, encoding="utf-8")
    records = (
        [json.loads(line) for line in rendered.splitlines()]
        if json_lines
        else json.loads(rendered)
    )
    assert len(records) == 1
    assert records[0]["url"] == "https://example.com/"


@pytest.mark.asyncio
async def test_json_storage_writes_empty_array_when_closed_without_records(
    tmp_path: Path,
) -> None:
    """Formatted JSON remains readable when no page was saved."""
    output = tmp_path / "empty.json"
    storage = JSONStorage(output, json_lines=False, indent=2)

    await storage.close()
    await storage.close()

    rendered = await asyncio.to_thread(output.read_text, encoding="utf-8")
    assert json.loads(rendered) == []
    with pytest.raises(RuntimeError, match="closed"):
        await storage.save(crawl_data())


@pytest.mark.asyncio
async def test_json_cancellation_after_write_keeps_formatted_array_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed output updates format state before cancellation propagates."""
    output_path = tmp_path / "cancelled.json"
    storage = JSONStorage(output_path, json_lines=False, indent=2)
    async with storage._lock:
        output = await storage._ensure_open_locked()
    original_write = output.write
    written = asyncio.Event()
    release = asyncio.Event()

    async def controlled_write(value: str) -> int:
        count = await original_write(value)
        written.set()
        await release.wait()
        return count

    monkeypatch.setattr(output, "write", controlled_write)
    save_task = asyncio.create_task(storage.save(crawl_data()))
    await asyncio.wait_for(written.wait(), timeout=2)
    save_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await save_task

    assert storage.saved_count == 1
    await storage.close()
    rendered = await asyncio.to_thread(output_path.read_text, encoding="utf-8")
    assert len(json.loads(rendered)) == 1


@pytest.mark.asyncio
async def test_json_failed_partial_append_preserves_previous_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback removes only the failed fragment after an existing JSON line."""
    output_path = tmp_path / "preserved.jsonl"
    storage = JSONStorage(output_path)
    await storage.save(crawl_data())
    async with storage._lock:
        output = await storage._ensure_open_locked()
    original_write = output.write

    async def fail_after_partial_write(value: str) -> int:
        await original_write(value[: max(1, len(value) // 2)])
        raise OSError("partial append failed")

    monkeypatch.setattr(output, "write", fail_after_partial_write)

    with pytest.raises(OSError, match="partial append failed"):
        await storage.save(crawl_data("https://example.com/second"))

    monkeypatch.setattr(output, "write", original_write)
    await storage.close()
    rendered = await asyncio.to_thread(output_path.read_text, encoding="utf-8")
    records = [json.loads(line) for line in rendered.splitlines()]
    assert [record["url"] for record in records] == ["https://example.com/"]


@pytest.mark.asyncio
async def test_csv_storage_detects_one_header_under_concurrent_writes(
    tmp_path: Path,
) -> None:
    """Concurrent saves serialize rows and emit exactly one header."""
    output = tmp_path / "pages.csv"
    storage = CSVStorage(output, encoding="utf-16")
    urls = [f"https://example.com/{index}" for index in range(20)]

    await asyncio.gather(*(storage.save(crawl_data(url)) for url in urls))
    await storage.close()

    rendered = await asyncio.to_thread(output.read_text, encoding="utf-16")
    rows = list(csv.DictReader(io.StringIO(rendered, newline="")))
    assert len(rows) == len(urls)
    assert {row["url"] for row in rows} == set(urls)
    assert rows[0]["title"] == 'Example, "quoted"\npage'
    assert json.loads(rows[0]["links"])[1] == "https://example.org/?a=1&b=2"
    assert json.loads(rows[0]["metadata"])["nested"] == {"language": "ru"}
    assert rendered.count("url,title,text,links") == 1


@pytest.mark.asyncio
async def test_csv_storage_rolls_back_partial_header_and_row_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed first write does not leave a partial or duplicate CSV header."""
    output_path = tmp_path / "partial.csv"
    storage = CSVStorage(output_path)
    async with storage._lock:
        output = await storage._ensure_open_locked()
    original_write = output.write
    failed = False

    async def fail_after_partial_write(value: str) -> int:
        nonlocal failed
        if not failed:
            failed = True
            await original_write(value[: max(1, len(value) // 2)])
            raise OSError("partial CSV write failed")
        return await original_write(value)

    monkeypatch.setattr(output, "write", fail_after_partial_write)

    with pytest.raises(OSError, match="partial CSV write failed"):
        await storage.save(crawl_data())

    assert storage.saved_count == 0
    await storage.save(crawl_data())
    await storage.close()

    rendered = await asyncio.to_thread(output_path.read_text, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(rendered, newline="")))
    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.com/"
    assert rendered.count("url,title,text,links") == 1


@pytest.mark.asyncio
async def test_csv_cancellation_after_write_does_not_repeat_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed cancelled row advances CSV header and count state."""
    output_path = tmp_path / "cancelled.csv"
    storage = CSVStorage(output_path)
    async with storage._lock:
        output = await storage._ensure_open_locked()
    original_write = output.write
    written = asyncio.Event()
    release = asyncio.Event()

    async def controlled_write(value: str) -> int:
        count = await original_write(value)
        written.set()
        await release.wait()
        return count

    monkeypatch.setattr(output, "write", controlled_write)
    save_task = asyncio.create_task(storage.save(crawl_data()))
    await asyncio.wait_for(written.wait(), timeout=2)
    save_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await save_task

    monkeypatch.setattr(output, "write", original_write)
    await storage.save(crawl_data("https://example.com/second"))
    await storage.close()
    rendered = await asyncio.to_thread(output_path.read_text, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(rendered, newline="")))
    assert len(rows) == 2
    assert rendered.count("url,title,text,links") == 1


@pytest.mark.asyncio
async def test_sqlite_storage_batches_rows_and_creates_indexes(tmp_path: Path) -> None:
    """SQLite flushes complete batches and exposes indexed, intact records."""
    output = tmp_path / "pages.sqlite3"
    storage = SQLiteStorage(output, batch_size=2)

    await storage.init_db()
    await storage.save(crawl_data())
    assert storage.pending_count == 1
    await storage.save(crawl_data("https://example.com/second"))
    assert storage.pending_count == 0
    await storage.save(crawl_data("https://example.com/third"))
    assert storage.pending_count == 1
    await storage.flush()
    assert storage.pending_count == 0

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall(
            """
            SELECT url, title, text, links, metadata, crawled_at,
                   status_code, content_type
            FROM pages
            ORDER BY id
            """
        )
        indexes = await connection.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )

    assert len(rows) == 3
    assert rows[0][0] == "https://example.com/"
    assert json.loads(rows[0][3])[0] == "https://example.com/next"
    assert json.loads(rows[0][4])["nested"] == {"language": "ru"}
    assert rows[0][5:] == ("2026-07-28T12:30:00+00:00", 200, "text/html")
    assert {row[0] for row in indexes} >= {
        "idx_pages_url",
        "idx_pages_crawled_at",
        "idx_pages_record_key",
    }
    assert storage.saved_count == 3
    await storage.close()


@pytest.mark.asyncio
async def test_sqlite_storage_concurrent_saves_do_not_lose_rows(
    tmp_path: Path,
) -> None:
    """The backend lock protects its shared batch and connection."""
    output = tmp_path / "concurrent.sqlite3"
    storage = SQLiteStorage(output, batch_size=7)
    urls = [f"https://example.com/{index}" for index in range(50)]

    await asyncio.gather(*(storage.save(crawl_data(url)) for url in urls))
    await storage.close()

    async with aiosqlite.connect(output) as connection:
        row = await connection.execute_fetchall(
            "SELECT COUNT(*), COUNT(DISTINCT url) FROM pages"
        )
    assert row == [(50, 50)]


@pytest.mark.asyncio
async def test_sqlite_failed_batch_can_be_retried_without_duplicate_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed threshold insert does not retain the retried record twice."""
    output = tmp_path / "retry.sqlite3"
    storage = SQLiteStorage(output, batch_size=2)
    await storage.save(crawl_data())
    original_insert = storage._insert_batch
    attempts = 0

    async def fail_once(
        connection: aiosqlite.Connection,
        batch: tuple[tuple[object, ...], ...],
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise aiosqlite.OperationalError("database is temporarily busy")
        await original_insert(connection, batch)

    monkeypatch.setattr(storage, "_insert_batch", fail_once)
    second = crawl_data("https://example.com/second")

    with pytest.raises(aiosqlite.OperationalError):
        await storage.save(second)

    assert storage.pending_count == 1
    await storage.save(second)
    await storage.close()

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall("SELECT url FROM pages ORDER BY id")
    assert rows == [
        ("https://example.com/",),
        ("https://example.com/second",),
    ]


@pytest.mark.asyncio
async def test_sqlite_retry_after_uncertain_commit_upserts_complete_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-commit exception cannot duplicate rows when the batch is retried."""
    output = tmp_path / "uncertain-commit.sqlite3"
    storage = SQLiteStorage(output, batch_size=2)
    await storage.save(crawl_data())
    original_insert = storage._insert_batch
    attempts = 0

    async def fail_after_first_commit(
        connection: aiosqlite.Connection,
        batch: tuple[tuple[object, ...], ...],
    ) -> None:
        nonlocal attempts
        attempts += 1
        await original_insert(connection, batch)
        if attempts == 1:
            raise aiosqlite.OperationalError("commit acknowledgement was lost")

    monkeypatch.setattr(storage, "_insert_batch", fail_after_first_commit)
    second = crawl_data("https://example.com/second")

    with pytest.raises(aiosqlite.OperationalError):
        await storage.save(second)

    await storage.save(second)
    await storage.close()

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall(
            "SELECT url, COUNT(*) FROM pages GROUP BY url ORDER BY url"
        )
    assert rows == [
        ("https://example.com/", 1),
        ("https://example.com/second", 1),
    ]


@pytest.mark.asyncio
async def test_sqlite_retains_distinct_crawls_of_the_same_url(
    tmp_path: Path,
) -> None:
    """Record-level idempotency does not collapse later crawl history."""
    output = tmp_path / "history.sqlite3"
    storage = SQLiteStorage(output, batch_size=1)
    first = crawl_data()
    second = crawl_data(title="Updated title")
    second["crawled_at"] = datetime(2026, 7, 28, 13, 30, tzinfo=UTC)

    await storage.save(first)
    await storage.save(second)
    await storage.close()

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall(
            "SELECT title, crawled_at FROM pages ORDER BY id"
        )
    assert rows == [
        ('Example, "quoted"\npage', "2026-07-28T12:30:00+00:00"),
        ("Updated title", "2026-07-28T13:30:00+00:00"),
    ]


@pytest.mark.asyncio
async def test_sqlite_initialization_migrates_legacy_page_table(
    tmp_path: Path,
) -> None:
    """An existing table without record keys remains readable and writable."""
    output = tmp_path / "legacy.sqlite3"
    async with aiosqlite.connect(output) as connection:
        await connection.executescript(
            """
            CREATE TABLE pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                links TEXT NOT NULL,
                metadata TEXT NOT NULL,
                crawled_at TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                content_type TEXT NOT NULL
            );
            CREATE INDEX idx_pages_url ON pages(url);
            CREATE INDEX idx_pages_crawled_at ON pages(crawled_at);
            INSERT INTO pages (
                url, title, text, links, metadata, crawled_at,
                status_code, content_type
            ) VALUES (
                'https://example.com/', 'Legacy', '', '[]', '{}',
                '2026-07-27T12:00:00+00:00', 200, 'text/html'
            );
            """
        )
        await connection.commit()

    storage = SQLiteStorage(output, batch_size=1)
    await storage.save(crawl_data())
    await storage.close()

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall(
            "SELECT title, record_key FROM pages ORDER BY id"
        )
    assert len(rows) == 2
    assert rows[0][0] == "Legacy"
    assert str(rows[0][1]).startswith("legacy:")
    assert len(str(rows[1][1])) == 64


@pytest.mark.asyncio
async def test_sqlite_failed_initialization_can_be_retried_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema failure closes the partial connection before a later retry."""
    output = tmp_path / "init-retry.sqlite3"
    storage = SQLiteStorage(output)
    original_initialize = storage._initialize_connection
    attempts = 0

    async def fail_once(connection: aiosqlite.Connection) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise aiosqlite.OperationalError("schema is temporarily unavailable")
        await original_initialize(connection)

    monkeypatch.setattr(storage, "_initialize_connection", fail_once)

    with pytest.raises(aiosqlite.OperationalError):
        await storage.save(crawl_data())

    assert storage._connection is None
    assert storage.pending_count == 0
    await storage.save(crawl_data())
    await storage.close()

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall("SELECT COUNT(*) FROM pages")
    assert rows == [(1,)]


@pytest.mark.asyncio
async def test_sqlite_cancelled_failed_initialization_still_closes_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation wins over a concurrent schema error after cleanup."""
    storage = SQLiteStorage(tmp_path / "cancel-init.sqlite3")
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    original_close = storage._close_connection

    async def failing_initialize(_connection: aiosqlite.Connection) -> None:
        started.set()
        await release.wait()
        raise aiosqlite.OperationalError("schema failed")

    async def tracked_close(connection: aiosqlite.Connection) -> None:
        await original_close(connection)
        closed.set()

    monkeypatch.setattr(storage, "_initialize_connection", failing_initialize)
    monkeypatch.setattr(storage, "_close_connection", tracked_close)
    init_task = asyncio.create_task(storage.init_db())
    await asyncio.wait_for(started.wait(), timeout=2)
    init_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await init_task

    assert closed.is_set()
    assert storage._connection is None


@pytest.mark.asyncio
async def test_sqlite_storage_close_flushes_pending_rows_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """Closing commits a short final batch before releasing the connection."""
    output = tmp_path / "close.sqlite3"
    storage = SQLiteStorage(output, batch_size=100)

    await storage.save(crawl_data())
    await storage.close()
    await storage.close()

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall("SELECT url FROM pages")
    assert rows == [("https://example.com/",)]
    with pytest.raises(RuntimeError, match="closed"):
        await storage.save(crawl_data())


@pytest.mark.asyncio
async def test_sqlite_close_can_retry_a_failed_final_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A close-time commit error retains the batch for the next close attempt."""
    output = tmp_path / "close-retry.sqlite3"
    storage = SQLiteStorage(output, batch_size=100)
    await storage.save(crawl_data())
    original_insert = storage._insert_batch
    attempts = 0

    async def fail_once(
        connection: aiosqlite.Connection,
        batch: tuple[tuple[object, ...], ...],
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise aiosqlite.OperationalError("database is temporarily busy")
        await original_insert(connection, batch)

    monkeypatch.setattr(storage, "_insert_batch", fail_once)

    with pytest.raises(aiosqlite.OperationalError):
        await storage.close()

    assert storage.pending_count == 1
    await storage.close()

    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall("SELECT COUNT(*) FROM pages")
    assert rows == [(1,)]


@pytest.mark.asyncio
async def test_sqlite_flush_finishes_commit_before_propagating_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation cannot leave an unknown in-flight batch transaction."""
    output = tmp_path / "cancel.sqlite3"
    storage = SQLiteStorage(output, batch_size=100)
    await storage.save(crawl_data())
    started = asyncio.Event()
    release = asyncio.Event()
    original_insert = storage._insert_batch

    async def delayed_insert(
        connection: aiosqlite.Connection,
        batch: tuple[tuple[object, ...], ...],
    ) -> None:
        started.set()
        await release.wait()
        await original_insert(connection, batch)

    monkeypatch.setattr(storage, "_insert_batch", delayed_insert)
    flush_task = asyncio.create_task(storage.flush())
    await asyncio.wait_for(started.wait(), timeout=2)
    flush_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await flush_task

    assert storage.pending_count == 0
    await storage.close()
    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall("SELECT COUNT(*) FROM pages")
    assert rows == [(1,)]


@pytest.mark.asyncio
async def test_sqlite_cancelled_failed_flush_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation remains primary and retains a concurrently failed batch."""
    output = tmp_path / "cancel-failure.sqlite3"
    storage = SQLiteStorage(output, batch_size=100)
    await storage.save(crawl_data())
    started = asyncio.Event()
    release = asyncio.Event()
    original_insert = storage._insert_batch

    async def delayed_failure(
        _connection: aiosqlite.Connection,
        _batch: tuple[tuple[object, ...], ...],
    ) -> None:
        started.set()
        await release.wait()
        raise aiosqlite.OperationalError("commit failed")

    monkeypatch.setattr(storage, "_insert_batch", delayed_failure)
    flush_task = asyncio.create_task(storage.flush())
    await asyncio.wait_for(started.wait(), timeout=2)
    flush_task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError) as caught:
        await flush_task

    assert isinstance(caught.value.__cause__, aiosqlite.OperationalError)
    assert storage.pending_count == 1
    monkeypatch.setattr(storage, "_insert_batch", original_insert)
    await storage.close()
    async with aiosqlite.connect(output) as connection:
        rows = await connection.execute_fetchall("SELECT COUNT(*) FROM pages")
    assert rows == [(1,)]


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"links": "not-a-list"}, TypeError, "links"),
        ({"crawled_at": "yesterday"}, TypeError, "crawled_at"),
        ({"crawled_at": datetime(2026, 7, 28)}, ValueError, "timezone-aware"),
        ({"status_code": 0}, ValueError, "status_code"),
    ],
)
@pytest.mark.asyncio
async def test_storage_rejects_invalid_standardized_data(
    tmp_path: Path,
    change: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    """Malformed records fail before a file is opened."""
    storage = JSONStorage(tmp_path / "invalid.jsonl")
    data = crawl_data()
    data.update(change)

    with pytest.raises(error, match=message):
        await storage.save(data)

    await storage.close()
    assert not storage.path.exists()
