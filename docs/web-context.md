# Local web-context engine

CrawlForge can turn crawled pages into a compact, source-linked lexical context
without sending page content to an external service:

```text
AsyncCrawler
  -> CrawledPage
  -> ContentProcessor
  -> SourceDocument
  -> TextChunker
  -> SQLiteContextIndex
       -> FTS5/BM25
       -> optional float32 embeddings / exact cosine
  -> ContextEngine
       -> Python callers
       -> CLI
       -> local MCP stdio adapter (BM25 only)
```

The crawler remains responsible for HTTP, discovery, politeness, retries, and
resource lifecycle. Content processing, chunking, indexing, retrieval, and
budget selection are separate deterministic components. The CLI is an adapter
over `ContextEngine`; it does not contain retrieval logic.

## Python API

```python
import asyncio

from crawlforge import ContextEngine


async def main() -> None:
    async with ContextEngine(".crawlforge/index.db") as engine:
        indexing = await engine.ingest_url(
            "https://example.com/docs",
            max_pages=100,
            max_depth=2,
        )
        print(indexing.documents_indexed, indexing.chunks_indexed)

        hits = await engine.search("How are retries configured?", limit=5)
        for hit in hits:
            print(hit.rank, hit.bm25_score, hit.source.url)

        context = await engine.build_context(
            "How are retries configured?",
            limit=10,
            token_budget=3000,
        )
        for hit in context.hits:
            print(hit.chunk.text, hit.source.url)


asyncio.run(main())
```

`index_pages()` also accepts finite iterables of cleaned `SourceDocument`
objects or raw `CrawledPage` envelopes. `ingest_url()` processes successful
pages as the bounded crawler produces them, so raw HTML for the complete site
does not need to remain in memory.

`ContextEngine`, `ContentProcessor`, `TextChunker`, and `SQLiteContextIndex`
have asynchronous or synchronous APIs appropriate to their resource ownership.
The engine and index are asynchronous context managers and close their SQLite
connection deterministically.

Optional semantic methods reuse the same engine and provenance boundary:
`index_embeddings()`, `semantic_search()`, and
`build_semantic_context()`. They require an explicit `EmbeddingProvider`;
details are in [Local semantic retrieval](semantic-retrieval.md).

## CLI

Create or update an index:

```bash
crawlforge index https://example.com/docs \
  --database .crawlforge/index.db \
  --max-pages 100 \
  --max-depth 2
```

Retrieve a token-budgeted context:

```bash
crawlforge search "How are retries configured?" \
  --database .crawlforge/index.db \
  --limit 5 \
  --token-budget 3000
```

Both commands support `--json`. Machine-readable results are written to
standard output; crawler progress and diagnostics use standard error. Expected
argument, filesystem, schema, and capability errors use exit code 2 without a
traceback. A search with no lexical matches succeeds with an empty result.

The default database is `.crawlforge/index.db`. It is ordinary local data and
can be removed when no longer needed:

```bash
rm .crawlforge/index.db
```

Only remove a path you selected for the local index. CrawlForge does not provide
a remote index or remote deletion operation.

## Content processing

`ContentProcessor` parses with the project's existing BeautifulSoup/lxml
stack in a worker thread. It:

- removes `script`, `style`, `noscript`, and `template`;
- removes semantic `nav`/`footer` elements and conservatively recognized
  navigation roles or exact class/id markers;
- keeps the rest of the body instead of selecting only conventional
  `main`/`article` containers;
- normalizes ordinary whitespace while preserving Unicode;
- extracts a normalized page title and accepts an HTTP(S) canonical URL only
  when it has the same origin as the fetched page;
- preserves heading paths from `h1` through `h6`;
- keeps paragraphs and useful standalone text in nonstandard containers;
- renders nested ordered and unordered lists;
- keeps code block line structure and fenced Markdown;
- keeps link labels in plain text and link targets in Markdown;
- renders simple tables as rows in plain text and pipe tables in Markdown.

The original crawler `ParsedPage` contract is unchanged. The context layer uses
its own typed `SourceDocument`, so cleaned Markdown, hashes, sizes, transport
metadata, and provenance do not appear unexpectedly in existing crawl reports
or storage formats.

Source and cleaned sizes are UTF-8 byte counts. SHA-256 identifiers and content
hashes use normalized deterministic inputs rather than Python's
process-randomized `hash()`. Cross-origin canonical declarations cannot replace
an existing source identity in a shared local index.

## Chunking

`TextChunker` is configured in Unicode characters:

- `target_chars` is a soft packing target;
- `max_chars` is a hard maximum;
- `overlap_chars` is a small within-section overlap.

The chunker groups content by heading path before packing blocks. Paragraphs,
lists, tables, and code blocks remain intact when they fit. Oversized prose is
split by sentence, then whitespace, then a hard character boundary only when
required. Oversized code is split by lines before a hard split of an individual
line. Empty chunks and heading-only chunks are not emitted.

Chunk identifiers include the document identity, ordinal, and normalized
contextual content. The same input and settings therefore produce the same
chunks. The value called `estimated_tokens` is provided by a `TokenEstimator`;
it is not used as a character-limit name.

## SQLite and FTS5

The context index is intentionally separate from `SQLiteStorage`.
`SQLiteStorage` is an append-oriented crawl record backend, while the context
index owns update, deduplication, stale-chunk cleanup, retrieval, and a
versioned schema.

The schema contains:

- `context_schema_version` — the context schema version;
- `document_contents` — normalized content shared by exact duplicates;
- `documents` — stable URL/provenance records referencing content;
- `chunks` — globally deduplicated contextual chunk bodies;
- `content_chunks` — ordered content-to-chunk relationships;
- `chunk_provenance` — one deterministic, directly addressable source for each
  deduplicated chunk;
- `index_sessions` — counts, sizes, approximate token totals, and timings;
- `embedding_models` — compatible embedding fingerprints and configuration;
- `chunk_embeddings` — normalized float32 vectors for deduplicated chunks;
- `embedding_sessions` — incremental embedding outcomes and timings;
- `chunk_fts` — FTS5 text, document-title, and heading-path columns.

Document and chunk batches are written in explicit transactions with
parameterized SQL. Updating a document changes its content reference, removes
old relationships, and then deletes only content and FTS rows no longer
referenced by another source. Schema initialization is idempotent. A real FTS5
table creation is used as the capability check; unsupported SQLite builds
raise `FTS5UnavailableError`.

Schema version 2 materializes chunk provenance so search hydration is bounded
by the result limit even when identical chunks occur at many source URLs.
Opening a version-1 context index migrates it transactionally and backfills the
deterministic source mapping.

Schema version 3 adds optional semantic storage without changing FTS5 tables or
BM25 behavior. Opening a version-2 index creates the semantic tables
transactionally. Embeddings reference the internal globally deduplicated chunk
identity, are isolated by a configuration fingerprint, and are removed when
their chunk is no longer referenced.

FTS5 `bm25()` returns smaller values for stronger matches. CrawlForge sorts the
raw score in ascending order and exposes it as `SearchHit.bm25_score`; the
1-based `rank` is the convenient public ordering. The score is not converted
into a probability.

## Context budgets and metrics

`build_context()` retrieves BM25 candidates, removes repeated chunk content,
preserves relevance order, and adds complete chunks that fit the configured
budget. A chunk is not cut in the middle. If no chunk fits, the result is
empty even when candidates exist.

The default estimator uses a deterministic characters-per-token heuristic.
It does not reproduce the tokenizer of a particular model. The reported
metric:

```text
estimated_context_reduction =
    1 - returned_estimated_tokens / source_estimated_tokens
```

is an engineering estimate against the raw source sizes represented by the
candidate documents. It is not a measured billable-token saving for a specific
model.

Index sessions also report raw and cleaned UTF-8 bytes, estimated tokens before
and after cleaning, new and duplicate document/chunk counts, cleaning time, and
indexing time. Context results report selected chunks, returned characters and
estimated tokens, candidates considered, search time, the applied limit and
budget, and whether the index produced a hit.

## Current limits

The default retrieval path is still the lexical baseline:

- BM25 matches terms, not semantic intent or paraphrases;
- ranking depends on corpus term statistics and can favor rare literal terms;
- markup heuristics cannot identify every site-specific boilerplate block;
- the token estimator is deliberately model-agnostic and approximate;
- one deterministic source is selected when a globally deduplicated chunk has
  multiple equivalent sources;
- SQLite FTS5 must be available in the active Python SQLite build.

The optional semantic path provides local embeddings and a linear exact scan,
not an approximate vector database. There is no hybrid retrieval, reranker,
hosted embedding call, generated answer, browser rendering, or remote HTTP
service. The local MCP stdio adapter reuses the stable `ContextEngine` boundary
but exposes only BM25; it does not duplicate crawling, cleaning, chunking, SQL,
or ranking. See [`mcp.md`](mcp.md) for its fixed configuration, security model,
and four bounded tools.
