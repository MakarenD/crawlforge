"""Asynchronous storage backends for standardized crawl data."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TypedDict, cast

import aiofiles
import aiosqlite
from aiofiles.threadpool.text import AsyncTextIOWrapper

_FIELDS = (
    "url",
    "title",
    "text",
    "links",
    "metadata",
    "crawled_at",
    "status_code",
    "content_type",
)


class CrawlData(TypedDict):
    """Standardized data persisted for one successfully crawled page."""

    url: str
    title: str
    text: str
    links: list[str]
    metadata: dict[str, object]
    crawled_at: datetime
    status_code: int
    content_type: str


class DataStorage(ABC):
    """Persist standardized crawl records through an asynchronous interface."""

    def __init__(self) -> None:
        """Initialize common storage state."""
        self._saved_count = 0

    @property
    def saved_count(self) -> int:
        """Return the number of records accepted by this storage instance."""
        return self._saved_count

    @abstractmethod
    async def save(self, data: dict[str, object]) -> None:
        """Persist one record; tolerate retries when duplicates are unacceptable."""

    @abstractmethod
    async def close(self) -> None:
        """Flush pending data and close owned resources."""

    async def __aenter__(self) -> DataStorage:
        """Enter the asynchronous storage context."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the storage when leaving its asynchronous context."""
        await self.close()


class _AsyncTextStorage(DataStorage):
    """Share cancellation-safe lifecycle handling for text file storage."""

    def __init__(self, path: str | Path, *, encoding: str) -> None:
        super().__init__()
        self.path = Path(path)
        self.encoding = encoding
        self._file: AsyncTextIOWrapper | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._close_error: Exception | None = None
        self._write_error: Exception | None = None

    async def _ensure_open_locked(self) -> AsyncTextIOWrapper:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        if self._write_error is not None:
            raise self._write_error
        if self._file is None:
            self._file = await aiofiles.open(
                self.path,
                "w",
                encoding=self.encoding,
                newline="",
            )
        return self._file

    async def _write_locked(
        self,
        value: str,
        *,
        on_written: Callable[[], None] | None = None,
    ) -> None:
        output = await self._ensure_open_locked()
        position = await output.tell()

        async def write_complete_value() -> None:
            written = 0
            while written < len(value):
                count = await output.write(value[written:])
                if count <= 0:
                    raise OSError("text storage write made no progress")
                written += count
            if on_written is not None:
                on_written()

        write_task = asyncio.create_task(write_complete_value())
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError as cancelled:
            try:
                await write_task
            except Exception as write_error:
                await self._rollback_locked(output, position)
                raise cancelled from write_error
            raise
        except Exception:
            await self._rollback_locked(output, position)
            raise

    async def _rollback_locked(
        self,
        output: AsyncTextIOWrapper,
        position: int,
    ) -> None:
        async def rollback() -> None:
            await output.seek(position)
            await output.truncate(position)
            await output.flush()

        rollback_task = asyncio.create_task(rollback())
        try:
            await asyncio.shield(rollback_task)
        except asyncio.CancelledError as cancelled:
            try:
                await rollback_task
            except Exception as rollback_error:
                raise cancelled from rollback_error
            raise
        except Exception as rollback_error:
            self._write_error = OSError(
                "text storage could not roll back a partial record"
            )
            raise self._write_error from rollback_error

    async def _finish_locked(self) -> None:
        """Write any format trailer before the file is closed."""

    async def close(self) -> None:
        """Flush and close the output file; repeated calls are safe."""
        close_task = asyncio.create_task(self._close_impl())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await close_task
            except Exception as close_error:
                raise cancelled from close_error
            raise

    async def _close_impl(self) -> None:
        async with self._lock:
            if self._closed:
                if self._close_error is not None:
                    raise self._close_error
                return
            error = self._write_error
            if error is None:
                try:
                    await self._finish_locked()
                    output = self._file
                    if output is not None:
                        await output.flush()
                except Exception as caught:
                    error = caught
            output = self._file
            if output is not None:
                try:
                    await output.close()
                except Exception as caught:
                    if error is None:
                        error = caught
            self._closed = True
            self._file = None
            self._close_error = error
            if error is not None:
                raise error


class JSONStorage(_AsyncTextStorage):
    """Stream crawl records to JSON Lines or a formatted JSON array."""

    def __init__(
        self,
        path: str | Path,
        *,
        json_lines: bool = True,
        indent: int | None = None,
        ensure_ascii: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        """Configure the JSON representation and output encoding."""
        if json_lines and indent is not None:
            raise ValueError("indent requires json_lines=False")
        if indent is not None and indent < 0:
            raise ValueError("indent must be zero or greater")
        super().__init__(path, encoding=encoding)
        self._json_lines = json_lines
        self._indent = indent
        self._ensure_ascii = ensure_ascii
        self._array_started = False

    async def save(self, data: dict[str, object]) -> None:
        """Append one validated record without loading earlier records."""
        record = _json_record(_normalize_record(data))
        encoded = json.dumps(
            record,
            ensure_ascii=self._ensure_ascii,
            indent=self._indent,
        )
        async with self._lock:
            if self._json_lines:
                output = f"{encoded}\n"
            else:
                prefix = "[\n" if not self._array_started else ",\n"
                indented = "\n".join(f"  {line}" for line in encoded.splitlines())
                output = f"{prefix}{indented}"

            def record_written() -> None:
                self._array_started = self._array_started or not self._json_lines
                self._saved_count += 1

            await self._write_locked(output, on_written=record_written)

    async def _finish_locked(self) -> None:
        if self._json_lines:
            return
        if not self._array_started:
            await self._write_locked("[]\n")
        else:
            await self._write_locked("\n]\n")


class CSVStorage(_AsyncTextStorage):
    """Stream crawl records to CSV with an automatically generated header."""

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Configure the CSV output path and text encoding."""
        super().__init__(path, encoding=encoding)
        self._header_written = False

    async def save(self, data: dict[str, object]) -> None:
        """Append one CSV row with safe quoting for special characters."""
        record = _csv_record(_normalize_record(data), ensure_ascii=False)

        async with self._lock:
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(record))
            if not self._header_written:
                writer.writeheader()
            writer.writerow(record)

            def record_written() -> None:
                self._header_written = True
                self._saved_count += 1

            await self._write_locked(
                buffer.getvalue(),
                on_written=record_written,
            )


class SQLiteStorage(DataStorage):
    """Buffer crawl records and persist them to an indexed SQLite table."""

    def __init__(self, path: str | Path, *, batch_size: int = 100) -> None:
        """Configure the database path and insertion batch size."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        super().__init__()
        self.path: str | Path = ":memory:" if str(path) == ":memory:" else Path(path)
        self.batch_size = batch_size
        self._connection: aiosqlite.Connection | None = None
        self._buffer: list[tuple[object, ...]] = []
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def pending_count(self) -> int:
        """Return the number of buffered records awaiting a database commit."""
        return len(self._buffer)

    async def init_db(self) -> None:
        """Create the page table and search indexes when absent."""
        async with self._lock:
            await self._ensure_db_locked()

    async def save(self, data: dict[str, object]) -> None:
        """Buffer one record and insert a complete batch when ready."""
        record = _json_record(_normalize_record(data))
        record_key = _record_key(record)
        values = (
            record_key,
            record["url"],
            record["title"],
            record["text"],
            json.dumps(record["links"], ensure_ascii=False),
            json.dumps(record["metadata"], ensure_ascii=False),
            record["crawled_at"],
            record["status_code"],
            record["content_type"],
        )
        async with self._lock:
            await self._ensure_db_locked()
            inserted_at = len(self._buffer)
            self._buffer.append(values)
            try:
                if len(self._buffer) >= self.batch_size:
                    await self._flush_locked()
            except Exception:
                del self._buffer[inserted_at]
                raise
            self._saved_count += 1

    async def flush(self) -> None:
        """Commit every currently buffered record."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("SQLiteStorage is closed")
            await self._ensure_db_locked()
            await self._flush_locked()

    async def close(self) -> None:
        """Commit pending records and close the database connection."""
        close_task = asyncio.create_task(self._close_impl())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await close_task
            except Exception as close_error:
                raise cancelled from close_error
            raise

    async def _ensure_db_locked(self) -> aiosqlite.Connection:
        if self._closed:
            raise RuntimeError("SQLiteStorage is closed")
        if self._connection is None:
            connection = await aiosqlite.connect(self.path)
            init_task = asyncio.create_task(self._initialize_connection(connection))
            try:
                await asyncio.shield(init_task)
            except asyncio.CancelledError as cancelled:
                init_error: Exception | None = None
                try:
                    await init_task
                except Exception as caught:
                    init_error = caught
                try:
                    await self._close_connection(connection)
                except Exception as close_error:
                    raise cancelled from close_error
                if init_error is not None:
                    raise cancelled from init_error
                raise
            except Exception:
                await self._close_connection(connection)
                raise
            self._connection = connection
        return self._connection

    async def _initialize_connection(
        self,
        connection: aiosqlite.Connection,
    ) -> None:
        await connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_key TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                links TEXT NOT NULL,
                metadata TEXT NOT NULL,
                crawled_at TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                content_type TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);
            CREATE INDEX IF NOT EXISTS idx_pages_crawled_at ON pages(crawled_at);
            """
        )
        columns = await connection.execute_fetchall("PRAGMA table_info(pages)")
        if "record_key" not in {str(column[1]) for column in columns}:
            await connection.execute("ALTER TABLE pages ADD COLUMN record_key TEXT")
        await connection.execute(
            """
            UPDATE pages
            SET record_key = 'legacy:' || id
            WHERE record_key IS NULL
            """
        )
        await connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_pages_record_key
            ON pages(record_key)
            """
        )
        await connection.commit()

    async def _close_connection(self, connection: aiosqlite.Connection) -> None:
        close_task = asyncio.create_task(connection.close())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await close_task
            except Exception as close_error:
                raise cancelled from close_error
            raise

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        connection = await self._ensure_db_locked()
        batch = tuple(self._buffer)
        flush_task = asyncio.create_task(self._insert_batch(connection, batch))
        try:
            await asyncio.shield(flush_task)
        except asyncio.CancelledError as cancelled:
            try:
                await flush_task
            except Exception as flush_error:
                raise cancelled from flush_error
            del self._buffer[: len(batch)]
            raise
        else:
            del self._buffer[: len(batch)]

    async def _insert_batch(
        self,
        connection: aiosqlite.Connection,
        batch: tuple[tuple[object, ...], ...],
    ) -> None:
        await connection.executemany(
            """
            INSERT INTO pages (
                record_key,
                url,
                title,
                text,
                links,
                metadata,
                crawled_at,
                status_code,
                content_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO NOTHING
            """,
            batch,
        )
        await connection.commit()

    async def _close_impl(self) -> None:
        async with self._lock:
            if self._closed:
                return
            try:
                if self._buffer:
                    await self._flush_locked()
                connection = self._connection
                if connection is not None:
                    await self._close_connection(connection)
            except Exception:
                connection = self._connection
                if connection is not None:
                    await self._close_connection(connection)
                self._connection = None
                raise
            self._connection = None
            self._closed = True


def _normalize_record(data: dict[str, object]) -> CrawlData:
    missing = [field for field in _FIELDS if field not in data]
    if missing:
        raise ValueError(f"missing crawl data fields: {', '.join(missing)}")

    string_fields = ("url", "title", "text", "content_type")
    for field in string_fields:
        if not isinstance(data[field], str):
            raise TypeError(f"{field} must be a string")

    links = data["links"]
    if not isinstance(links, list) or any(not isinstance(link, str) for link in links):
        raise TypeError("links must be a list of strings")
    metadata = data["metadata"]
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) for key in metadata
    ):
        raise TypeError("metadata must be a dictionary with string keys")
    crawled_at = data["crawled_at"]
    if not isinstance(crawled_at, datetime):
        raise TypeError("crawled_at must be a datetime")
    if crawled_at.tzinfo is None or crawled_at.utcoffset() is None:
        raise ValueError("crawled_at must be timezone-aware")
    status_code = data["status_code"]
    if (
        isinstance(status_code, bool)
        or not isinstance(status_code, int)
        or not 100 <= status_code <= 599
    ):
        raise ValueError("status_code must be an integer between 100 and 599")

    return cast(
        CrawlData,
        {
            "url": data["url"],
            "title": data["title"],
            "text": data["text"],
            "links": list(links),
            "metadata": dict(metadata),
            "crawled_at": crawled_at,
            "status_code": status_code,
            "content_type": data["content_type"],
        },
    )


def _json_record(data: CrawlData) -> dict[str, object]:
    return {
        "url": data["url"],
        "title": data["title"],
        "text": data["text"],
        "links": list(data["links"]),
        "metadata": dict(data["metadata"]),
        "crawled_at": data["crawled_at"].isoformat(),
        "status_code": data["status_code"],
        "content_type": data["content_type"],
    }


def _record_key(data: dict[str, object]) -> str:
    canonical = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _csv_record(
    data: CrawlData,
    *,
    ensure_ascii: bool,
) -> dict[str, str | int]:
    return {
        "url": data["url"],
        "title": data["title"],
        "text": data["text"],
        "links": json.dumps(data["links"], ensure_ascii=ensure_ascii),
        "metadata": json.dumps(data["metadata"], ensure_ascii=ensure_ascii),
        "crawled_at": data["crawled_at"].isoformat(),
        "status_code": data["status_code"],
        "content_type": data["content_type"],
    }
