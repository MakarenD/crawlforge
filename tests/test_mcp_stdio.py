"""Subprocess smoke tests for the packaged stdio MCP executable."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "index_site",
    "search_index",
    "build_context",
    "get_index_info",
}


@pytest.mark.asyncio
async def test_stdio_initialize_tools_list_and_clean_shutdown(
    tmp_path: Path,
) -> None:
    executable = shutil.which("crawlforge-mcp")
    assert executable is not None
    stderr_path = tmp_path / "stderr.log"
    parameters = StdioServerParameters(
        command=executable,
        args=[
            "--database",
            str(tmp_path / "index.db"),
            "--log-level",
            "INFO",
        ],
        cwd=str(tmp_path),
    )

    with stderr_path.open("w+", encoding="utf-8") as stderr:
        async with stdio_client(parameters, errlog=stderr) as streams:
            async with ClientSession(*streams) as session:
                initialized = await session.initialize()
                tools = await session.list_tools()
        stderr.flush()
        stderr.seek(0)
        diagnostics = stderr.read()

    assert initialized.server_info.name == "crawlforge"
    assert {tool.name for tool in tools.tools} == EXPECTED_TOOLS
    assert "MCP startup" in diagnostics
    assert "MCP shutdown" in diagnostics


@pytest.mark.asyncio
async def test_stdio_modern_client_discovers_and_calls_tool(tmp_path: Path) -> None:
    executable = shutil.which("crawlforge-mcp")
    assert executable is not None
    stderr_path = tmp_path / "modern-stderr.log"
    parameters = StdioServerParameters(
        command=executable,
        args=["--database", str(tmp_path / "modern.db")],
        cwd=str(tmp_path),
    )

    with stderr_path.open("w+", encoding="utf-8") as stderr:
        async with Client(stdio_client(parameters, errlog=stderr)) as client:
            tools = await client.list_tools()
            info = await client.call_tool("get_index_info", {})
        stderr.flush()
        stderr.seek(0)
        diagnostics = stderr.read()

    assert {tool.name for tool in tools.tools} == EXPECTED_TOOLS
    assert not info.is_error
    assert info.structured_content is not None
    assert info.structured_content["database_ready"]
    assert "MCP startup" in diagnostics
    assert "MCP shutdown" in diagnostics
