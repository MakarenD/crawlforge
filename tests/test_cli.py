"""Tests for the command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import __version__
from crawlforge.cli import build_parser, run


@asynccontextmanager
async def serve(app: web.Application) -> AsyncIterator[TestServer]:
    """Run an aiohttp application on an ephemeral local port."""
    server = TestServer(app)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


def test_module_help_is_available() -> None:
    """The module entry point prints its usage information."""
    result = subprocess.run(
        [sys.executable, "-m", "crawlforge", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "High-performance asynchronous web crawler" in result.stdout
    assert "--version" in result.stdout
    assert "--urls" in result.stdout
    assert "--max-pages" in result.stdout
    assert "--max-depth" in result.stdout
    assert "--output" in result.stdout
    assert "--config" in result.stdout
    assert "--respect-robots" in result.stdout
    assert "--rate-limit" in result.stdout


def test_module_version_is_available() -> None:
    """The module entry point reports the package version."""
    result = subprocess.run(
        [sys.executable, "-m", "crawlforge", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"crawlforge {__version__}"


@pytest.mark.asyncio
async def test_cli_config_overrides_run_crawl_and_write_output(
    tmp_path: Path,
) -> None:
    """Explicit CLI values override JSON settings and dispatch the crawler."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(text="<title>CLI page</title>")

    app = web.Application()
    app.router.add_get("/", page)

    async with serve(app) as server:
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "urls": [str(server.make_url("/"))],
                    "crawler": {
                        "max_pages": 50,
                        "rate_limit": 1,
                        "respect_robots": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        output = tmp_path / "cli-results.json"
        arguments = build_parser().parse_args(
            [
                "--config",
                str(config_path),
                "--max-pages",
                "1",
                "--output",
                str(output),
                "--no-respect-robots",
                "--rate-limit",
                "1000",
            ]
        )

        stats = await run(arguments)

    assert stats["total_pages"] == 1
    assert stats["successful"] == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stats"]["total_pages"] == 1
    assert next(iter(payload["pages"].values()))["title"] == "CLI page"
