# CrawlForge BM25 Retrieval Baseline

## Dataset

- Name: `crawlforge-retrieval-baseline`
- Version: `1.0.0`
- Retrieval strategy: `bm25-fts5`
- Queries: 64
- Timestamp: `2026-07-29T13:13:53.636111+00:00`

## Corpus statistics

| Measure | Value |
| --- | ---: |
| Documents | 10 |
| Stable sections | 40 |
| Indexed chunks | 50 |
| Source bytes | 33655 |
| Cleaned bytes | 25641 |
| Approximate source tokens | 8407 |
| Approximate cleaned tokens | 6406 |
| Corpus processing and indexing | 26.670 ms |

## Chunking configuration

- `max_chars`: `1600`
- `overlap_chars`: `160`
- `target_chars`: `1200`

## Standard retrieval metrics

Positive-query metrics exclude the explicitly negative queries. Negative queries are evaluated separately with no-result accuracy.

| K | Hit Rate | Precision | Recall | MAP | NDCG |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 78.6% | 78.6% | 54.8% | 0.7857 | 0.7500 |
| 3 | 96.4% | 44.6% | 76.9% | 0.7242 | 0.7975 |
| 5 | 96.4% | 28.9% | 80.2% | 0.7294 | 0.8100 |
| 10 | 100.0% | 17.5% | 91.2% | 0.7645 | 0.8459 |

- MRR: `0.8681`
- No-result accuracy: `12.5%`
- Failed queries: 0

## Metrics by category

| Category | Queries | Hit@5 | Recall@5 | MRR | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exact_term` | 8 | 100.0% | 100.0% | 1.0000 | 0.9792 |
| `code_symbol` | 8 | 100.0% | 100.0% | 1.0000 | 0.9637 |
| `error_lookup` | 8 | 100.0% | 93.8% | 1.0000 | 0.9897 |
| `paraphrase` | 8 | 87.5% | 81.2% | 0.7639 | 0.7625 |
| `conceptual` | 8 | 100.0% | 81.2% | 0.7083 | 0.7127 |
| `ambiguous` | 8 | 87.5% | 40.6% | 0.6667 | 0.5382 |
| `multi_relevant` | 8 | 100.0% | 64.6% | 0.9375 | 0.7238 |
| `negative` | 8 | n/a | n/a | 0.0000 | n/a |

## Strongest queries

| Query | Category | First relevant | Hit@5 | NDCG@5 | Irrelevant token ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `q016` ServerRuntime | `code_symbol` | 1 | 100.0% | 1.0000 | 0.0% |
| `q015` TextChunker | `code_symbol` | 1 | 100.0% | 1.0000 | 0.0% |
| `q013` URLNetworkPolicy | `code_symbol` | 1 | 100.0% | 1.0000 | 0.0% |
| `q012` RateLimiter | `code_symbol` | 1 | 100.0% | 1.0000 | 0.0% |
| `q011` RetryStrategy | `code_symbol` | 1 | 100.0% | 1.0000 | 0.0% |
| `q010` SemaphoreManager | `code_symbol` | 1 | 100.0% | 1.0000 | 0.0% |
| `q009` AsyncCrawler | `code_symbol` | 1 | 100.0% | 1.0000 | 0.0% |
| `q001` Crawl-delay | `exact_term` | 1 | 100.0% | 1.0000 | 0.0% |
| `q002` Retry-After | `exact_term` | 1 | 100.0% | 1.0000 | 50.7% |
| `q006` same-origin canonical | `exact_term` | 1 | 100.0% | 1.0000 | 58.4% |

## Weakest queries

| Query | Category | First relevant | Hit@5 | NDCG@5 | Irrelevant token ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `q063` GPU embedding batch size and CUDA memory | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q061` S3 object versioning and lifecycle retention rules | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q059` OAuth device flow client registration | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q062` Kafka consumer offset commit strategy | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q060` WebSocket streaming endpoint heartbeat interval | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q041` limits | `ambiguous` | 6 | 0.0% | 0.0000 | 72.6% |
| `q046` policy | `ambiguous` | 2 | 100.0% | 0.4702 | 88.1% |
| `q048` errors | `ambiguous` | 3 | 100.0% | 0.3234 | 63.1% |

## False positives

| Query | Category | Rank | Retrieved source | Section |
| --- | --- | ---: | --- | --- |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 1 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Configuration precedence |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 2 | https://benchmark.crawlforge.local/async-http | Async HTTP Timeouts and Lifecycle |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 3 | https://benchmark.crawlforge.local/async-http | Async HTTP Timeouts and Lifecycle > Session lifecycle |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 4 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Command boundaries |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 5 | https://benchmark.crawlforge.local/retrieval-context | BM25 Retrieval and Context Budgets > BM25 ranking semantics |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 6 | https://benchmark.crawlforge.local/retry-errors | Retries, Errors, and Backoff > Retry-After handling |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 7 | https://benchmark.crawlforge.local/async-http | Async HTTP Timeouts and Lifecycle > Empty bodies and failures |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 8 | https://benchmark.crawlforge.local/async-http | Async HTTP Timeouts and Lifecycle > Timeout budgets |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 9 | https://benchmark.crawlforge.local/mcp-adapter | Local MCP Adapter Operations > Stdio lifecycle |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 10 | https://benchmark.crawlforge.local/content-chunking | Cleaning, Chunking, and Token Estimates > Structure preservation |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 1 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Command boundaries |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 2 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 3 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Configuration precedence |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 4 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Storage backends |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 5 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Error and output channels |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 6 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 7 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design > Storage separation |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 8 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design > Schema migrations |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 9 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design > FTS5 virtual table |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 10 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design > Transactions and provenance |

## False negatives

| Query | Category | Missed judgments | First relevant rank |
| --- | --- | --- | ---: |
| `q018` FTS5UnavailableError when SQLite lacks search support | `error_lookup` | q018-j2 | 1 |
| `q041` limits | `ambiguous` | q041-j2, q041-j3 | 6 |
| `q042` cleanup | `ambiguous` | q042-j3 | 1 |
| `q044` storage | `ambiguous` | q044-j3 | 1 |
| `q045` ranking | `ambiguous` | q045-j2 | 1 |
| `q046` policy | `ambiguous` | q046-j2, q046-j3 | 2 |
| `q047` budget | `ambiguous` | q047-j3 | 1 |
| `q048` errors | `ambiguous` | q048-j4 | 3 |
| `q049` How should timeout, cancellation, and retry delays share ownership of a complete fetch? | `multi_relevant` | q049-j2 | 1 |
| `q050` How are concurrency slots, crawl spacing, and robots delay combined for one origin? | `multi_relevant` | q050-j1 | 1 |
| `q051` Which checks must run again when an allowed public URL redirects? | `multi_relevant` | q051-j4 | 2 |
| `q052` Trace cleaned HTML from structured blocks through chunks into the lexical index. | `multi_relevant` | q052-j4 | 1 |
| `q055` Compare lifecycle and failure channels for the local tool server and ordinary CLI. | `multi_relevant` | q055-j4 | 1 |
| `q056` How are stable identities preserved across canonical URLs, chunks, and provenance rows? | `multi_relevant` | q056-j4 | 1 |

## Warm-index retrieval latency

Indexing is excluded. These timings are machine-specific and are not a portable quality gate.

| Samples | Repeats/query | Warm-ups | Mean | Median | P95 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 5 | 3 | 0.507 ms | 0.495 ms | 0.852 ms | 2.388 ms |

## CrawlForge-specific context efficiency

These project-specific measurements describe approximate bounded context, not standardized IR metrics or exact model-token savings.

| Measure | Value |
| --- | ---: |
| Mean candidates before context selection | 7.531 |
| Mean returned estimated tokens | 1077.656 |
| Relevant chunks per 1000 estimated tokens | 1.421 |
| Irrelevant estimated-token ratio | 78.0% |
| Mean relevant-source coverage | 91.2% |
| Mean estimated context reduction | 69.1% |

## Benchmark limitations

- The corpus and judgments are small, synthetic, and designed for transparent regression analysis rather than broad external validity.
- Relevance matching uses stable document, canonical source, section, heading, and optional evidence checks; it does not infer semantics.
- Negative-query abstention is strict because raw SQLite FTS5 BM25 scores are not calibrated confidence values.
- Token counts use CrawlForge's deterministic character heuristic and are not exact for a particular model.
- Retrieval quality does not measure generated-answer correctness, faithfulness, or usefulness.
- Warning: Latency values are warm-index measurements for this machine only.
- Warning: Negative-query no-result accuracy treats any returned lexical candidate as a false positive because BM25 scores are not calibrated.

## Readiness for semantic retrieval

At Hit@5, exact-term/code-symbol queries average 100.0%, while paraphrase/conceptual queries average 93.8%. Semantic retrieval is justified specifically for vocabulary-mismatch cases where the intended mechanism is described without its indexed terms. Observed examples: q031. A future comparison should keep this dataset and add vector/hybrid strategies without changing the judgments.
