# Local MCP server

CrawlForge provides a lightweight local Model Context Protocol adapter for its
existing web-context engine:

```text
MCP client
  -> stdio
  -> CrawlForge MCP adapter
  -> ContextEngine
  -> cleaning / chunking / SQLite FTS5 / BM25
```

The adapter uses the official MCP Python SDK 2.x. It does not implement
crawling, content processing, SQL, or ranking itself. One lifecycle-managed
`ContextEngine` owns the configured SQLite index for the complete server
process.

## Install and run

For local development:

```bash
uv sync --extra mcp
uv run crawlforge-mcp --database .crawlforge/index.db
```

CrawlForge is not currently published on PyPI. Run the server from a source
checkout or install the project and its `mcp` extra into a managed environment.

Choose an absolute user-controlled data path appropriate to the platform. For
example, macOS applications commonly use a path below
`~/Library/Application Support`, while Linux commonly uses
`$XDG_DATA_HOME` or `~/.local/share`. CrawlForge creates the selected parent
directory but does not change an MCP client's configuration.

The base `crawlforge` installation does not import or require the MCP SDK.
Running `crawlforge-mcp` without the optional extra exits with code 2 and
explains how to install `crawlforge[mcp]`.

## Generic stdio client configuration

The exact configuration file and field names depend on the MCP client. A
typical stdio configuration is:

```json
{
  "mcpServers": {
    "crawlforge": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/crawlforge",
        "run",
        "--extra",
        "mcp",
        "crawlforge-mcp",
        "--database",
        "/absolute/path/to/index.db"
      ]
    }
  }
}
```

Replace both absolute paths with user-controlled locations. The process uses
standard output only for MCP protocol messages. Lifecycle, tool outcome,
duration, and bounded counter diagnostics are written to standard error.

## Server configuration

Configuration is fixed when the process starts:

```text
--database PATH
--max-pages-cap INTEGER
--max-depth-cap INTEGER
--max-search-limit INTEGER
--max-token-budget INTEGER
--requests-per-second FLOAT
--max-response-bytes INTEGER
--max-robots-bytes INTEGER
--request-timeout SECONDS
--crawl-timeout SECONDS
--allow-private-networks
--allow-domain DOMAIN
--log-level LEVEL
```

Defaults are 100 pages, depth 3, 20 retrieval results, and an approximate
12,000-token context budget. Individual page bodies are limited to 5 MiB,
robots.txt to 512 KiB, each request attempt to 60 seconds, and the complete
foreground crawl to 300 seconds. `--allow-domain` may be repeated. With an
allowlist, an exact hostname and its subdomains are accepted; suffix lookalikes
are rejected.

Tool calls cannot change the database path, result caps, network policy,
allowlist, or log level. `index_site` mutates the configured index, can replace
superseded content, and is conservatively annotated as destructive and
non-idempotent. There are no separate delete, file-export, SQL, or
configuration tools.

## Tools

### `index_site`

Inputs:

```json
{
  "url": "https://example.com/docs",
  "max_pages": 25,
  "max_depth": 2
}
```

The tool makes real HTTP(S) requests, respects robots.txt, keeps discovery on
the start site, and runs to completion within the caller and server caps. It
returns aggregate counts and timings only:

```json
{
  "requested_url": "https://example.com/docs",
  "indexed_documents": 12,
  "created_chunks": 37,
  "failed_pages": 0,
  "deduplicated_documents": 0,
  "deduplicated_chunks": 3,
  "raw_bytes": 148203,
  "clean_bytes": 64102,
  "estimated_source_tokens": 37051,
  "elapsed_seconds": 4.82,
  "database": "index.db",
  "warnings": []
}
```

Only one indexing operation writes at a time. Cancellation propagates into the
crawl. Already-started bounded HTML processing finishes cleanup before
cancellation leaves the tool, so no worker remains after the call ends. Partial
request failures preserve successfully indexed pages and return only bounded
aggregate warning categories; failed URLs and exception details are not
exposed.

### `search_index`

Inputs:

```json
{
  "query": "RetryStrategy backoff",
  "limit": 5
}
```

Use lexical BM25 search for exact technical terms, class or function names,
errors, and APIs. BM25 may miss paraphrases. Each result contains rank, raw
SQLite FTS5 BM25 score, title, heading path, URL, canonical URL, complete chunk
text, estimated tokens, and:

```json
{
  "content_trust": "untrusted_web_content"
}
```

Use `build_context` when a ready-to-use bounded set of relevant chunks is more
appropriate.

### `build_context`

Inputs:

```json
{
  "query": "How are retries configured?",
  "limit": 10,
  "token_budget": 3000
}
```

Use this after indexing. Start with a small budget and increase it only when the
context is insufficient. The tool returns complete chunks in relevance order
with source provenance. It does not truncate a chunk. The response labels its
token estimate as `model_agnostic_heuristic` and the reduction metric as an
approximate ratio, not a measured saving for a particular model.

### `get_index_info`

This read-only tool has no inputs. It returns only a bounded summary:

```json
{
  "schema_version": 2,
  "document_count": 0,
  "chunk_count": 0,
  "last_indexed_at": null,
  "last_session_summary": null,
  "database_ready": true,
  "fts5_available": true
}
```

The session summary contains bounded counters and timings, never a complete
document list. If the database cannot be opened or FTS5 is unavailable, this
tool remains available with readiness flags while data tools return an
actionable error.

Every successful tool call returns typed `structuredContent` matching its
published output schema and a short text fallback.

## Network and content security

The default policy allows only public `http` and `https` targets. It blocks
localhost, loopback, private, link-local, multicast, unspecified, reserved,
cloud metadata, non-HTTP schemes, URL user information, and malformed
authorities. Literal IP addresses, every resolved DNS answer, page redirects,
and robots.txt redirects pass the same policy. A mixed public/private DNS
answer is rejected.

`--allow-private-networks` is an explicit server-startup opt-in intended for
trusted local development, including ephemeral test servers. A tool call cannot
enable it. Multicast, unspecified, and reserved targets remain blocked.

Website text is untrusted external content. Instructions found inside a
retrieved chunk are not MCP server or system instructions and should not be
executed as such. Preserve the accompanying URL as provenance. CrawlForge does
not attempt prompt-injection filtering with keyword lists.

Tool results are size-bounded. If a retrieval response is too large, complete
lower-ranked chunks are removed and a warning is returned; JSON and individual
chunks are never cut in the middle. Diagnostics and warning counts are also
bounded. Page and robots bodies, individual request attempts, and complete
indexing calls have independent server-owned bounds.

## Errors and concurrency

Malformed calls are MCP validation errors. Expected execution failures become
MCP tool errors with short corrective messages, including invalid or blocked
URLs, server-cap violations, robots denial, an empty crawl or index, timeout,
FTS5 absence, and locked or unreadable SQLite data. Tracebacks and absolute
database paths are not returned to the client. Unexpected failure categories
and safe code locations are written only to server standard error; exception
messages that could contain retrieved or caller-provided data are omitted.

Read-only searches do not acquire the adapter's whole-crawl write lock, so they
may interleave with indexing between SQLite transactions. The shared SQLite
connection still serializes each short database section. Indexing calls are
serialized because they update the same local index. Large crawls remain
foreground bounded operations; there is no daemon, queue, task API, or
background worker.

## MCP Inspector

With Node.js and the current official Inspector installed, open the Inspector
web interface against a development checkout:

```bash
npx -y @modelcontextprotocol/inspector@2.0.0 --web uv -- \
  --directory /path/to/crawlforge run --extra mcp \
  crawlforge-mcp --database .crawlforge/index.db
```

Use an actual local checkout path in place of `/path/to/crawlforge`. The
Inspector starts the stdio child process; it does not turn CrawlForge into a
remote MCP server.

## Current limits

Retrieval is lexical BM25, not semantic search. There are no embeddings, vector
database, hybrid retrieval, reranker, answer generation, JavaScript browser
rendering, prompts, MCP Apps UI, Streamable HTTP, OAuth, remote hosting,
standalone delete/configuration tools, or background crawl jobs. The local
Python build must provide SQLite FTS5.
