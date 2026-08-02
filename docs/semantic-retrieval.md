# Local semantic retrieval

CrawlForge supports optional local semantic retrieval with pinned Sentence
Transformers embeddings and exact cosine search. BM25 remains the default.
Semantic retrieval is an explicit Python or CLI choice and does not send
documents to an embedding API.

## What the index stores

An embedding is a fixed-length numeric representation of text. The provider
creates document embeddings for indexed chunks and a query embedding for each
search. Texts with related meanings can have similar vectors even when they use
different words.

CrawlForge normalizes every vector to unit length and stores it as a
little-endian float32 blob. It ranks each compatible chunk by exact cosine
similarity:

```text
cosine(query, document) = dot(query, document)
```

The simplified equality is valid because both vectors are normalized. Higher
scores rank first. Search scans every compatible vector, so its CPU and memory
cost are linear in:

```text
number_of_chunks * embedding_dimension
```

This is a transparent baseline for local collections, not a production-scale
vector database or approximate nearest-neighbor index.

## Optional installation

The base package does not install or import PyTorch, Transformers, or Sentence
Transformers:

```bash
uv sync
```

Install semantic inference only where it is needed:

```bash
uv sync --extra semantic
```

Equivalent editable installations are:

```bash
python -m pip install -e .
python -m pip install -e ".[semantic]"
python -m pip install -e ".[mcp,semantic]"
```

CrawlForge itself supports Python 3.12–3.14. On Intel macOS, upstream PyTorch
no longer publishes compatible current wheels for Python 3.13 or 3.14. The
semantic extra therefore uses the last compatible PyTorch line on Python 3.12
there; the base package and non-Intel platforms retain their normal Python
range.

## Pinned model

The default configuration is reproducible:

| Setting | Value |
| --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Dimension | 384 |
| Precision and storage | float32 |
| Normalization | enabled |
| Document method | `encode_document` |
| Query method | `encode_query` |
| Canonical device | CPU |
| Model license | Apache-2.0 |

The model revision is immutable rather than a moving branch. Its pinned
[model-card metadata](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
declares Apache-2.0; CrawlForge itself remains MIT licensed.

Model files are intentionally not included in the package, repository, SQLite
vectors, or built wheel. Sentence Transformers downloads them to its configured
local cache on first use. Use `--cache-directory` to select a cache and
`--local-files-only` to require an already cached pinned revision.

## Deterministic document and query inputs

Document inputs use formatter version
`crawlforge-semantic-document-v1`:

```text
Title: {document title}
Section: {heading 1} > {heading 2}

{complete chunk text}
```

The chunk text retains Unicode and fenced code produced by the existing content
pipeline. Query formatter version `crawlforge-semantic-query-v1` strips only
outer whitespace. Query and document encoding stay separate so a provider can
apply task-specific behavior.

The model fingerprint covers:

- provider implementation;
- model ID and immutable revision;
- dimension, normalization, dtype, and precision;
- document and query formatter versions.

Changing any of those values creates a different fingerprint. Incompatible
vectors are never mixed in one ranking.

## SQLite schema and incremental indexing

Context schema version 3 adds three tables:

- `embedding_models` records fingerprints, model configuration, safe runtime
  metadata, and pending invalidation counts;
- `chunk_embeddings` relates a fingerprint and internal deduplicated chunk ID
  to its content hash and float32 blob;
- `embedding_sessions` records cache hits, failures, invalidations, truncation,
  byte counts, and separated indexing timings.

`chunk_embeddings.chunk_id` references the existing global `chunks.id` with
foreign-key cleanup. A second foreign key references the model fingerprint.
Index writes are parameterized, batched, and transactional.

`crawlforge embed` enumerates missing chunks in bounded batches. An unchanged
chunk and fingerprint is a cache hit. When a document update removes a chunk,
its vector is removed through the same index lifecycle. A stale content hash is
deleted before it can be reused, and the invalidation appears in the next
embedding session. A failed batch rolls back without leaving a partial batch;
the next run retries missing work.

## CLI

First build or update the ordinary context index:

```bash
uv run crawlforge index https://example.com/docs \
  --database .crawlforge/index.db
```

Build the compatible embedding index:

```bash
uv run --extra semantic crawlforge embed \
  --database .crawlforge/index.db \
  --device cpu \
  --batch-size 32
```

The command reports the model fingerprint, new embeddings, cache hits,
invalidations, failures, truncation statistics, timing, and stored bytes. A
second unchanged run should report cache hits rather than recompute vectors.

Search with semantic ranking and the same complete-chunk budget selection:

```bash
uv run --extra semantic crawlforge search \
  "How does the crawler avoid overwhelming a host?" \
  --database .crawlforge/index.db \
  --strategy semantic \
  --limit 5 \
  --token-budget 3000
```

Semantic search does not silently build missing vectors. It returns a clear
error when the index is missing or only an incompatible fingerprint exists.
Omitting `--strategy semantic` preserves the existing BM25 behavior.

## Python API

The public provider is lazy and owns one model per lifecycle:

```python
import asyncio

from crawlforge import ContextEngine, SentenceTransformerEmbeddingProvider


async def main() -> None:
    provider = SentenceTransformerEmbeddingProvider(device="cpu")
    try:
        async with ContextEngine(".crawlforge/index.db") as engine:
            indexing = await engine.index_embeddings(provider, batch_size=32)
            print(indexing.embedded_chunks, indexing.cache_hits)

            hits = await engine.semantic_search(
                "How does the crawler limit host traffic?",
                provider=provider,
                limit=5,
            )
            context = await engine.build_semantic_context(
                "How does the crawler limit host traffic?",
                provider=provider,
                limit=10,
                token_budget=3000,
            )
    finally:
        await provider.close()

    for hit in hits:
        print(hit.rank, hit.cosine_similarity, hit.source.url)
    print(context.estimated_tokens)


asyncio.run(main())
```

`EmbeddingProvider` is the typed boundary for model inference. It exposes
separate document and query methods, dimension and normalization contracts,
safe runtime metadata, aggregate input analysis, and asynchronous cleanup.
Inference runs outside the event loop. Controlled-vector implementations live
only in tests.

`SemanticSearchHit` preserves the complete chunk, stable provenance, rank,
cosine value, model identity, fingerprint, score type, and retrieval strategy.
`SemanticContextResult` uses the same whole-chunk, deduplicated, strict
token-budget selection as BM25.

## Evaluation and measured baseline

The semantic evaluator implements the existing strategy protocol and uses the
same frozen dataset, chunks, relevance judgments, query order, K values, and
token budget as BM25. The dataset signature for both checked reports is:

```text
bb1bf9a8b79f7b47f2850aac362f144d7984196648f592716c7e0d33ff00acfd
```

Run the pinned semantic baseline:

```bash
uv run --extra semantic crawlforge evaluate run \
  --strategy semantic \
  --database .crawlforge/semantic-evaluation.db \
  --output reports/semantic-baseline.json \
  --format json \
  --device cpu
```

Run the paired comparison:

```bash
uv run --extra semantic crawlforge evaluate compare \
  --strategies bm25,semantic \
  --database .crawlforge/evaluation-compare.db \
  --output reports/bm25-vs-semantic.md \
  --format markdown \
  --device cpu
```

Measured aggregate results:

| Metric | BM25 | Semantic | Semantic delta |
| --- | ---: | ---: | ---: |
| Hit Rate@5 | 0.9643 | 0.9821 | +0.0179 |
| Recall@5 | 0.8021 | 0.8229 | +0.0208 |
| MRR | 0.8681 | 0.8563 | -0.0118 |
| MAP@5 | 0.7294 | 0.6970 | -0.0324 |
| NDCG@5 | 0.8100 | 0.8102 | +0.0002 |
| Negative no-result accuracy | 0.1250 | 0.0000 | -0.1250 |

Category MRR shows the trade-off:

| Category | BM25 | Semantic | Delta |
| --- | ---: | ---: | ---: |
| Exact term | 1.0000 | 0.8125 | -0.1875 |
| Code symbol | 1.0000 | 0.8438 | -0.1562 |
| Error lookup | 1.0000 | 1.0000 | 0.0000 |
| Paraphrase | 0.7639 | 0.8542 | +0.0903 |
| Conceptual | 0.7083 | 0.9062 | +0.1979 |
| Ambiguous | 0.6667 | 0.6708 | +0.0042 |
| Multi-relevant | 0.9375 | 0.9062 | -0.0312 |

Semantic retrieval won 15 individual queries and BM25 won 21. Seven negative
queries failed under both strict no-result rules. Semantic scores have no
fitted abstention threshold, so every non-empty semantic index returns nearest
neighbors even when none should be considered relevant.

The paired bootstrap is deterministic and query-level, but its 95% intervals
cross zero for Hit@5, Recall@5, MRR, and NDCG@5. The corpus is too small and
synthetic for claims of statistical significance or generalization. See the
complete [semantic report](../reports/semantic-baseline.md) and
[paired comparison](../reports/bm25-vs-semantic.md).

## Truncation, latency, and storage

`all-MiniLM-L6-v2` truncates inputs beyond 256 word pieces. CrawlForge measures
untruncated tokenizer lengths before encoding without storing tokenizer output.
The canonical corpus recorded:

- 50 document inputs;
- 0 truncated inputs;
- maximum untruncated length 199;
- average untruncated length 133.7.

Chunking is not changed automatically to fit this model. A future chunking
comparison must be a separate experiment.

The canonical CPU JSON run measured:

- model load: 3796.0 ms;
- document encoding: 1227.0 ms, or 40.7 embeddings/second;
- SQLite vector writes: 5.7 ms;
- mean readiness check: 0.13 ms;
- mean query encoding: 5.2 ms;
- mean coherent SQLite snapshot fetch: 0.42 ms;
- mean float32 vector decode: 1.1 ms;
- mean exact scan: 1.3 ms;
- mean provenance materialization from snapshot rows: 0.41 ms;
- mean total semantic retrieval: 8.6 ms;
- stored and loaded vector bytes: 76,800;
- 1,536 vector bytes per chunk;
- model cache size: about 183 MB.

These timings describe one Intel macOS CPU run and are not regression gates or
scalability promises. Python object overhead can exceed the raw vector byte
estimate during a scan. For concurrency safety, SQLite fetches compatible
vectors and complete provenance in one coherent statement; the report therefore
does not pretend that their database transfer costs are independently
measurable.

## Real-model smoke and offline mode

The normal test suite never downloads a model. Run the explicit integration
test when network access is intended:

```bash
CRAWLFORGE_RUN_SEMANTIC_MODEL_TESTS=1 \
CRAWLFORGE_SEMANTIC_CACHE=/path/to/cache \
uv run --extra semantic pytest tests/test_semantic_model_integration.py
```

After the pinned revision is cached, verify offline behavior:

```bash
CRAWLFORGE_RUN_SEMANTIC_MODEL_TESTS=1 \
CRAWLFORGE_SEMANTIC_OFFLINE=1 \
CRAWLFORGE_SEMANTIC_CACHE=/path/to/cache \
uv run --extra semantic pytest tests/test_semantic_model_integration.py
```

No API key is required.

## Limits and decision boundary

- Cosine similarity is a relative ranking score, not calibrated confidence.
- No score threshold, abstention policy, or reranker is fitted. Hybrid search
  uses fixed rank fusion rather than calibrated semantic scores.
- The default model is lightweight and English-focused.
- Model download is required unless the exact revision is already cached.
- Exact scan materializes compatible vectors and is unsuitable for unbounded
  collections.
- Token budgets still use CrawlForge's deterministic approximation, not the
  embedding tokenizer.
- The MCP server remains lexical BM25. Semantic and hybrid strategy selection
  require a separate MCP model-lifecycle and deployment design.

The measured gains on paraphrase and conceptual queries justify keeping
semantic candidates alongside BM25 rather than replacing lexical retrieval.
The fixed RRF implementation and its checked results are documented in
[Hybrid retrieval](hybrid-retrieval.md); the dataset and judgments remain
unchanged from this semantic baseline.
