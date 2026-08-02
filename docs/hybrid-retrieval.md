# Rank-fused hybrid retrieval

CrawlForge can combine the existing lexical BM25 and exact semantic rankings
with deterministic Reciprocal Rank Fusion (RRF). BM25 remains the default.
Hybrid search is an explicit Python or CLI choice and uses the same local
chunks, SQLite index, embeddings, provenance, and complete-chunk context
selection as the component strategies.

## Why combine rankings

The frozen retrieval baseline exposes complementary behavior. BM25 is strongest
when a query contains an exact technical term, error string, or code symbol.
Semantic retrieval is often stronger when a query paraphrases an indexed
concept. Neither raw score can be compared directly:

- SQLite FTS5 BM25 scores are lower-is-better;
- cosine similarities are higher-is-better;
- neither score is calibrated confidence.

Hybrid retrieval therefore combines positions, not raw scores.

```text
BM25 candidates ----\
                     > Reciprocal Rank Fusion -> bounded context
Semantic candidates /
```

## Reciprocal Rank Fusion

For a chunk that appears at lexical rank `r_bm25` and semantic rank
`r_semantic`, CrawlForge computes:

```text
RRF(chunk) =
    bm25_weight / (rrf_k + r_bm25)
    + semantic_weight / (rrf_k + r_semantic)
```

A missing rank contributes zero. The checked baseline uses a configuration
chosen before evaluation:

| Setting | Value |
| --- | ---: |
| `rrf_k` | 60 |
| `bm25_weight` | 1.0 |
| `semantic_weight` | 1.0 |
| `bm25_candidate_limit` | 50 |
| `semantic_candidate_limit` | 50 |

The equal weights are a reproducible baseline, not learned or tuned values.
The final RRF score is a ranking score, not a probability or confidence.

## Candidate identity and deduplication

Both component searches read the same globally deduplicated SQLite chunk
records. Fusion identifies a stored chunk by its full SHA-256 content hash. The
index collision-checks the associated title, heading path, and text, so this is
the stable storage identity rather than a comparison of short text fragments.

When both rankings contain the same chunk, hybrid search returns it once and
keeps:

- both component ranks and RRF contributions;
- the raw BM25 score and cosine similarity as separate evidence;
- complete chunk text and token estimate;
- source URL, canonical URL, title, heading path, and document identity;
- the embedding model fingerprint and fusion configuration.

Raw component scores never participate in fusion or tie-breaking.

## Deterministic ordering

Results sort by descending RRF score. Exact ties use, in order:

1. a chunk present in both component rankings;
2. the best minimum component rank;
3. BM25 rank;
4. semantic rank;
5. stable content hash.

Ranks are recomputed after fusion. Duplicate identities or ranks, non-contiguous
component ranks, non-finite values, invalid weights, and invalid candidate
limits are rejected rather than silently repaired.

## Execution and failure behavior

The canonical implementation executes BM25 first and semantic retrieval second.
Both use one locked SQLite connection, so their database phases cannot run in
parallel. In a direct controlled-vector benchmark over the frozen 64-query
corpus, concurrent orchestration was slower than sequential orchestration
(1.8721 ms versus 1.7484 ms per query). These values are machine-specific and
are not performance gates.

Each component read is internally coherent, but the two public searches do not
form one cross-strategy transaction. A concurrent reindex can therefore make
them observe different committed revisions. Semantic snapshot checks still
fail explicitly if compatible embeddings become incomplete.

Hybrid search requires both strategies. A lexical, semantic, or fusion failure
returns an actionable typed error and no partial result. Missing embeddings are
never built during search. Build them explicitly:

```bash
uv run --extra semantic crawlforge embed \
  --database .crawlforge/index.db \
  --device cpu
```

Cancellation propagates to the caller. `HybridRetriever` borrows its
`ContextEngine` and `EmbeddingProvider`; the caller owns both lifecycles.

## CLI

Search with the fixed production defaults:

```bash
uv run --extra semantic crawlforge search \
  "How does the crawler avoid overwhelming a host?" \
  --database .crawlforge/index.db \
  --strategy hybrid \
  --limit 5 \
  --token-budget 3000
```

Advanced controls are explicit:

```bash
uv run --extra semantic crawlforge search \
  "How does the crawler avoid overwhelming a host?" \
  --database .crawlforge/index.db \
  --strategy hybrid \
  --rrf-k 60 \
  --bm25-weight 1.0 \
  --semantic-weight 1.0 \
  --bm25-candidates 50 \
  --semantic-candidates 50 \
  --limit 5 \
  --json
```

Human-readable output shows the final rank, RRF score, component ranks, source,
heading, and estimated tokens. JSON retains the complete typed fusion evidence.
Omitting `--strategy hybrid` preserves the BM25 default.

## Python API

```python
import asyncio

from crawlforge import (
    ContextEngine,
    HybridRetriever,
    SentenceTransformerEmbeddingProvider,
)


async def main() -> None:
    async with ContextEngine(".crawlforge/index.db") as engine:
        async with SentenceTransformerEmbeddingProvider(device="cpu") as provider:
            hybrid = HybridRetriever(
                context_engine=engine,
                embedding_provider=provider,
            )
            hits = await hybrid.search(
                "How does the crawler avoid overwhelming a host?",
                limit=5,
            )
            context = await hybrid.build_context(
                "How does the crawler avoid overwhelming a host?",
                limit=10,
                token_budget=3000,
            )

    for hit in hits:
        print(hit.rank, hit.rrf_score, hit.bm25_rank, hit.semantic_rank)
    print(context.estimated_tokens)


asyncio.run(main())
```

Document embeddings must already exist. One query embedding is computed for
each hybrid search.

## Context selection

`build_context()` sends the fused ranking through the shared context selector.
It removes repeated stored content, preserves ranking order, skips chunks that
would exceed the remaining token budget, and never truncates a chunk midway.
The result records the retrieval and fusion strategies, RRF settings, candidate
limits, model fingerprint, source-token estimate, and context reduction.

## Evaluation

Run the hybrid strategy against the unchanged version 1.0.0 dataset:

```bash
uv run --extra semantic crawlforge evaluate run \
  --strategy hybrid \
  --device cpu \
  --output reports/hybrid-baseline.json
```

Compare all checked strategies with the same corpus, chunks, query order,
judgments, K values, and token budget:

```bash
uv run --extra semantic crawlforge evaluate compare \
  --strategies bm25,semantic,hybrid \
  --device cpu \
  --bootstrap-samples 5000 \
  --bootstrap-seed 20260729 \
  --output reports/bm25-vs-semantic-vs-hybrid.md
```

The multi-strategy report includes standard IR metrics, per-category MRR,
query-level ranks, component contributions, candidate overlap at K=1, 3, 5,
and 10, unique relevant coverage, fusion recovery, and paired bootstrap
intervals. Its oracle-union recall uses ground truth and is only a diagnostic
upper bound; it is not a deployable retrieval strategy.

See the checked [hybrid baseline](../reports/hybrid-baseline.md) and
[three-strategy comparison](../reports/bm25-vs-semantic-vs-hybrid.md) for the
measured results, including regressions and negative-query behavior.

## Measured fixed baseline

The unchanged version 1.0.0 dataset has signature
`bb1bf9a8b79f7b47f2850aac362f144d7984196648f592716c7e0d33ff00acfd`:

| Metric | BM25 | Semantic | Hybrid |
| --- | ---: | ---: | ---: |
| Hit@5 | 0.9643 | 0.9821 | 0.9821 |
| Precision@5 | 0.2893 | 0.3071 | 0.3250 |
| Recall@5 | 0.8021 | 0.8229 | 0.8705 |
| MRR | 0.8681 | 0.8563 | 0.8810 |
| MAP@5 | 0.7294 | 0.6970 | 0.7608 |
| NDCG@5 | 0.8100 | 0.8102 | 0.8546 |
| Negative no-result accuracy | 0.1250 | 0.0000 | 0.0000 |

Candidate overlap is limited: mean per-query BM25/semantic Jaccard is 0.5312
at K=1, 0.3591 at K=3, 0.3566 at K=5, and 0.3280 at K=10. At K=5, 72 of
114 relevant judgments were found by both components, 9 only by BM25, 14 only
by semantic, and 19 by neither. The ground-truth oracle union recall at K=5 is
0.8333 globally (0.8929 mean per positive query); this is diagnostic only.

RRF recovered 19 of the 23 component-only relevant judgments into final top-5:
9/9 BM25-only and 10/14 semantic-only. Among 320 final top-5 items, 263
(82.19%) had contributions from both lists and 57 (17.81%) were semantic-only;
none were BM25-only because the 50-deep semantic list covered the complete
50-chunk corpus in this benchmark. Average contributions were 0.0129 from BM25
and 0.0156 from semantic. Dual-source promotion was 12.55%. The reported 55.63%
single-source retention compares exclusive identities in the standalone top-5
lists; it is not a retention measure over the deeper hybrid candidate pools.

Hybrid beat both component outcomes on five queries and improved aggregate
quality, but the outcome classifier also records 17 query-level regressions.
All three strategies failed seven of eight strict negative queries, and hybrid
returned candidates for all eight. No threshold was fitted to change that.

Deterministic paired bootstrap intervals use 5,000 samples and seed 20260729.
For hybrid versus BM25, Recall@5 delta is +0.0685 with interval
[+0.0253, +0.1221], while MRR delta is +0.0129 with interval
[-0.0317, +0.0575]. For hybrid versus semantic, Recall@5 delta is +0.0476 with
interval [-0.0045, +0.1071], while MRR delta is +0.0247 with interval
[-0.0435, +0.0967]. These intervals are exploratory; the dataset is small and
synthetic.

One canonical CPU run measured mean warm component boundaries of 1.35 ms for
BM25, 9.31 ms for query encoding, 1.59 ms for exact scan, 0.52 ms for fusion,
2.28 ms for semantic snapshot/decode/provenance hydration, and 15.68 ms total
hybrid retrieval. Model loading and document embedding indexing are excluded.
These machine-dependent values are evidence for this run, not performance
thresholds.

## Limitations and next experiments

- The benchmark corpus is small, synthetic, and English-focused.
- Equal RRF weights and `rrf_k=60` are fixed defaults, not a general optimum.
- Exact semantic search remains linear in chunk count and embedding dimension.
- Warm latency is machine-dependent and excludes model loading and indexing.
- RRF has no calibrated abstention behavior and can return results for negative
  queries.
- There is no query classifier, query rewriting, score normalization, learned
  fusion, reranker, vector database, or approximate nearest-neighbor index.
- The MCP tools remain lexical BM25; adding model lifecycle to MCP requires a
  separate deployment design.

The next step should be chosen from evidence. A reranker is justified only if
the candidate union contains relevant chunks that fixed RRF orders poorly.
Abstention calibration needs a separate calibration set, risk/coverage curves,
and a clarification policy. A larger real-world benchmark is required before
generalizing the checked synthetic results.
