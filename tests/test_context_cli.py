"""CLI tests for local indexing and lexical context retrieval."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


class _DocumentationHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        """Return one deterministic documentation page."""
        body = (
            b"<title>Retry documentation</title>"
            b"<nav>Navigation noise</nav>"
            b"<h1>Retries</h1>"
            b"<p>Configure retry_after and exponential backoff locally.</p>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        """Keep test output quiet."""


@contextmanager
def serve() -> Iterator[str]:
    """Run a deterministic threaded HTTP server."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DocumentationHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_index_and_search_subcommands_have_help() -> None:
    """New command help is available without replacing legacy root flags."""
    root = _run("--help")
    index = _run("index", "--help")
    search = _run("search", "--help")

    assert root.returncode == 0
    assert "--urls" in root.stdout
    assert "index" in root.stdout
    assert "search" in root.stdout
    assert index.returncode == 0
    assert "--database" in index.stdout
    assert "--max-pages" in index.stdout
    assert search.returncode == 0
    assert "--token-budget" in search.stdout


def test_index_and_search_json_stdout_is_machine_readable(tmp_path: Path) -> None:
    """JSON mode emits only the requested payload and includes source provenance."""
    database = tmp_path / "nested" / "context.db"
    with serve() as url:
        indexed = _run(
            "index",
            url,
            "--database",
            str(database),
            "--max-pages",
            "1",
            "--max-depth",
            "0",
            "--rate-limit",
            "1000",
            "--no-respect-robots",
            "--json",
        )
    searched = _run(
        "search",
        "retry_after backoff",
        "--database",
        str(database),
        "--limit",
        "3",
        "--token-budget",
        "100",
        "--json",
    )

    assert indexed.returncode == 0
    indexed_payload = json.loads(indexed.stdout)
    assert indexed_payload["documents_seen"] == 1
    assert indexed_payload["chunks_indexed"] >= 1
    assert searched.returncode == 0
    searched_payload = json.loads(searched.stdout)
    assert searched_payload["results"][0]["url"] == url
    assert searched_payload["results"][0]["rank"] == 1
    assert isinstance(searched_payload["results"][0]["bm25_score"], float)
    assert searched_payload["estimated_tokens"] <= 100


def test_search_human_output_and_expected_error_exit_codes(tmp_path: Path) -> None:
    """Human results show citations while missing indexes fail without traceback."""
    database = tmp_path / "context.db"
    with serve() as url:
        indexed = _run(
            "index",
            url,
            "--database",
            str(database),
            "--max-pages",
            "1",
            "--max-depth",
            "0",
            "--rate-limit",
            "1000",
            "--no-respect-robots",
        )
    searched = _run(
        "search",
        "retry_after",
        "--database",
        str(database),
    )
    missing = _run(
        "search",
        "query",
        "--database",
        str(tmp_path / "missing.db"),
        "--json",
    )

    assert indexed.returncode == 0
    assert "Indexed 1/1 documents" in indexed.stdout
    assert searched.returncode == 0
    assert f"URL: {url}" in searched.stdout
    assert "BM25:" in searched.stdout
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert "context index does not exist" in missing.stderr
    assert "Traceback" not in missing.stderr


def test_search_rejects_corrupted_database_without_traceback(tmp_path: Path) -> None:
    """A damaged local index is an expected CLI error, not an internal traceback."""
    database = tmp_path / "corrupted.db"
    database.write_bytes(b"not a SQLite database")

    result = _run(
        "search",
        "query",
        "--database",
        str(database),
        "--json",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "file is not a database" in result.stderr
    assert "Traceback" not in result.stderr


def test_search_reports_locked_database_without_traceback(tmp_path: Path) -> None:
    """A busy local index follows the same safe CLI error contract."""
    database = tmp_path / "locked.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE marker (value INTEGER)")
        connection.commit()
        connection.execute("BEGIN EXCLUSIVE")
        result = _run(
            "search",
            "query",
            "--database",
            str(database),
            "--json",
        )
    finally:
        connection.rollback()
        connection.close()

    assert result.returncode == 2
    assert result.stdout == ""
    assert "database is locked" in result.stderr
    assert "Traceback" not in result.stderr


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "crawlforge", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
