# CrawlForge BM25 vs Semantic Retrieval

## Reproducibility

- Dataset: `crawlforge-retrieval-baseline` `1.0.0`
- Dataset signature: `bb1bf9a8b79f7b47f2850aac362f144d7984196648f592716c7e0d33ff00acfd`
- Baseline: `bm25-fts5`
- Candidate: `semantic-exact-cosine`
- Semantic model: `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Semantic vectors: 384 dimensions, float32, normalized=true
- Semantic formatter: `crawlforge-semantic-document-v1` documents, `crawlforge-semantic-query-v1` queries
- Semantic device: `cpu`
- Both runs use the same corpus, chunks, judgments, K values, and token budget.

## Aggregate comparison

| Metric | BM25 | Semantic | Delta |
| --- | ---: | ---: | ---: |
| Hit@5 | 0.9643 | 0.9821 | +0.0179 |
| Precision@5 | 0.2893 | 0.3071 | +0.0179 |
| Recall@5 | 0.8021 | 0.8229 | +0.0208 |
| MRR | 0.8681 | 0.8563 | -0.0118 |
| MAP@5 | 0.7294 | 0.6970 | -0.0324 |
| NDCG@5 | 0.8100 | 0.8102 | +0.0002 |
| Negative no-result accuracy | 0.1250 | 0.0000 | -0.1250 |

## Category comparison

| Category | BM25 MRR | Semantic MRR | Delta |
| --- | ---: | ---: | ---: |
| `exact_term` | 1.0000 | 0.8125 | -0.1875 |
| `code_symbol` | 1.0000 | 0.8438 | -0.1562 |
| `error_lookup` | 1.0000 | 1.0000 | +0.0000 |
| `paraphrase` | 0.7639 | 0.8542 | +0.0903 |
| `conceptual` | 0.7083 | 0.9062 | +0.1979 |
| `ambiguous` | 0.6667 | 0.6708 | +0.0042 |
| `multi_relevant` | 0.9375 | 0.9062 | -0.0312 |
| `negative` | 0.0000 | 0.0000 | +0.0000 |

## Paired bootstrap uncertainty

These deterministic 95% percentile intervals are exploratory. The dataset is small and synthetic; the intervals do not prove generalization to real sites or automatic statistical significance.

| Metric | Mean delta | Lower 95% | Upper 95% | Samples | Seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hit@5 | +0.0179 | -0.0357 | +0.0893 | 5000 | 20260729 |
| Recall@5 | +0.0208 | -0.0595 | +0.0997 | 5000 | 20260729 |
| MRR | -0.0118 | -0.1066 | +0.0759 | 5000 | 20260729 |
| NDCG@5 | +0.0002 | -0.0737 | +0.0721 | 5000 | 20260729 |

## Failure analysis

- Semantic wins: `q014`, `q029`, `q031`, `q032`, `q034`, `q035`, `q037`, `q038`, `q041`, `q043`, `q044`, `q045`, `q048`, `q051`, `q053`
- BM25 wins: `q002`, `q004`, `q007`, `q011`, `q016`, `q023`, `q027`, `q033`, `q039`, `q040`, `q042`, `q046`, `q047`, `q052`, `q054`, `q055`, `q056`, `q059`, `q060`, `q062`, `q064`
- Both succeed: `q001`, `q002`, `q003`, `q004`, `q005`, `q006`, `q008`, `q009`, `q010`, `q011`, `q012`, `q013`, `q014`, `q015`, `q016`, `q017`, `q018`, `q019`, `q020`, `q021`, `q022`, `q023`, `q024`, `q025`, `q026`, `q027`, `q028`, `q029`, `q030`, `q032`, `q033`, `q034`, `q035`, `q036`, `q037`, `q038`, `q039`, `q040`, `q042`, `q043`, `q044`, `q045`, `q046`, `q047`, `q048`, `q049`, `q050`, `q051`, `q052`, `q053`, `q054`, `q055`, `q056`
- Both fail: `q057`, `q058`, `q059`, `q060`, `q061`, `q062`, `q063`
- Semantic regressions: `q002`, `q004`, `q007`, `q011`, `q016`, `q023`, `q027`, `q033`, `q039`, `q040`, `q042`, `q046`, `q047`, `q052`, `q054`, `q055`, `q056`, `q064`

## Selected query analysis

- `q031` (paraphrase): first relevant rank BM25 9, semantic 3; Recall@5 delta +1.0000; NDCG@5 delta +0.5000.
- `q041` (ambiguous): first relevant rank BM25 6, semantic 5; Recall@5 delta +0.2500; NDCG@5 delta +0.2502.
- `q046` (ambiguous): first relevant rank BM25 2, semantic 3; Recall@5 delta +0.0000; NDCG@5 delta -0.0976.
- `q048` (ambiguous): first relevant rank BM25 3, semantic 1; Recall@5 delta +0.5000; NDCG@5 delta +0.3965.

## Query-level paired evidence

| Query | Category | BM25 first relevant | Semantic first relevant | Recall@5 delta | NDCG@5 delta | BM25-only sources | Semantic-only sources |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `q001` Crawl-delay | `exact_term` | 1 | 1 | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/cli-config |
| `q002` Retry-After | `exact_term` | 1 | 2 | -0.5000 | -0.4212 | https://benchmark.crawlforge.local/mcp-adapter | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/async-http |
| `q003` FTS5 virtual table | `exact_term` | 1 | 1 | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/concurrency |
| `q004` BM25 raw score | `exact_term` | 1 | 1 | -0.5000 | -0.2128 | https://benchmark.crawlforge.local/cli-config | https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/mcp-adapter |
| `q005` token budget | `exact_term` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/mcp-adapter | https://benchmark.crawlforge.local/politeness |
| `q006` same-origin canonical | `exact_term` | 1 | 1 | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/sqlite-search |
| `q007` standard output | `exact_term` | 1 | none | -1.0000 | -0.8340 | https://benchmark.crawlforge.local/retrieval-context | https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/url-security, https://benchmark.crawlforge.local/async-http |
| `q008` private and non-public network addresses are blocked | `exact_term` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/concurrency | none |
| `q009` AsyncCrawler | `code_symbol` | 1 | 1 | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/concurrency |
| `q010` SemaphoreManager | `code_symbol` | 1 | 1 | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/cli-config |
| `q011` RetryStrategy | `code_symbol` | 1 | 2 | +0.0000 | -0.3691 | none | https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/concurrency |
| `q012` RateLimiter | `code_symbol` | 1 | 1 | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/mcp-adapter |
| `q013` URLNetworkPolicy | `code_symbol` | 1 | 1 | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/async-http |
| `q014` SQLiteContextIndex | `code_symbol` | 1 | 1 | -0.5000 | +0.2075 | none | https://benchmark.crawlforge.local/retrieval-context |
| `q015` TextChunker | `code_symbol` | 1 | 1 | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/cli-config, https://benchmark.crawlforge.local/async-http |
| `q016` ServerRuntime | `code_symbol` | 1 | 4 | +0.0000 | -0.5693 | none | https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/cli-config, https://benchmark.crawlforge.local/concurrency |
| `q017` token_budget must be greater than zero | `error_lookup` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/url-security, https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/retry-errors | none |
| `q018` FTS5UnavailableError when SQLite lacks search support | `error_lookup` | 1 | 1 | +0.0000 | +0.0000 | none | none |
| `q019` invalid Retry-After header is ignored | `error_lookup` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/cli-config | https://benchmark.crawlforge.local/async-http |
| `q020` robots policy unavailable | `error_lookup` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/cli-config, https://benchmark.crawlforge.local/async-http | https://benchmark.crawlforge.local/url-security |
| `q021` redirect limit exceeded | `error_lookup` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/cli-config, https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/sqlite-search | none |
| `q022` CrawlForge MCP support is not installed | `error_lookup` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/sqlite-search, https://benchmark.crawlforge.local/retrieval-context | none |
| `q023` database is locked with no traceback | `error_lookup` | 1 | 1 | +0.0000 | -0.1660 | https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/async-http | none |
| `q024` empty response body or failed fetch | `error_lookup` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/retrieval-context | none |
| `q025` How should a crawler distinguish connection setup time from a stalled body stream? | `paraphrase` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/url-security, https://benchmark.crawlforge.local/retrieval-context | https://benchmark.crawlforge.local/concurrency |
| `q026` How can the scheduler avoid creating a live task for every discovered link? | `paraphrase` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/url-security | https://benchmark.crawlforge.local/mcp-adapter |
| `q027` What delay should be used when a throttling response supplies a date or number of seconds? | `paraphrase` | 1 | 2 | +0.5000 | -0.2821 | https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/sqlite-search, https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/retrieval-context | none |
| `q028` When does a missing crawler policy file allow requests, and when should access be denied? | `paraphrase` | 1 | 1 | +0.0000 | +0.0000 | none | none |
| `q029` Why must every resolved address be checked even when the hostname is permitted? | `paraphrase` | 2 | 1 | +0.0000 | +0.4826 | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/retrieval-context | https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/cli-config |
| `q030` How can a schema upgrade remain retryable after a failed backfill? | `paraphrase` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/url-security, https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/concurrency | none |
| `q031` Why should context assembly skip an oversized passage instead of slicing it? | `paraphrase` | 9 | 3 | +1.0000 | +0.5000 | https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/cli-config | none |
| `q032` How are navigation clutter and useful technical structure treated during extraction? | `paraphrase` | 2 | 1 | +0.0000 | +0.2671 | https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/concurrency | https://benchmark.crawlforge.local/sqlite-search |
| `q033` Why is one pooled HTTP session preferable to a session per URL? | `conceptual` | 1 | 1 | +0.0000 | -0.0148 | https://benchmark.crawlforge.local/concurrency | https://benchmark.crawlforge.local/content-chunking |
| `q034` What invariants make cancellation safe for bounded worker tasks? | `conceptual` | 2 | 1 | +0.0000 | +0.2671 | https://benchmark.crawlforge.local/sqlite-search, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/mcp-adapter | https://benchmark.crawlforge.local/content-chunking |
| `q035` Why are retries limited to classified temporary failures? | `conceptual` | 2 | 1 | +0.0000 | +0.3386 | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/cli-config, https://benchmark.crawlforge.local/url-security | https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/concurrency |
| `q036` How do request capacity and start-time spacing solve different politeness problems? | `conceptual` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/sqlite-search | https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/content-chunking |
| `q037` Why are hostname allowlists insufficient protection against server-side request forgery? | `conceptual` | 3 | 1 | +0.6667 | +0.6831 | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/retry-errors | https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/async-http |
| `q038` Why should crawl history and the lexical retrieval index use separate schemas? | `conceptual` | 3 | 1 | +0.5000 | +0.5241 | https://benchmark.crawlforge.local/async-http | none |
| `q039` What does a strong context reduction percentage prove and not prove? | `conceptual` | 1 | 1 | -0.5000 | -0.0827 | https://benchmark.crawlforge.local/sqlite-search, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/cli-config | https://benchmark.crawlforge.local/concurrency |
| `q040` Why must retrieved page text remain untrusted at a tool boundary? | `conceptual` | 1 | 4 | +0.0000 | -0.5693 | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/retry-errors | https://benchmark.crawlforge.local/content-chunking |
| `q041` limits | `ambiguous` | 6 | 5 | +0.2500 | +0.2502 | https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/async-http | https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/url-security |
| `q042` cleanup | `ambiguous` | 1 | 3 | -0.3333 | -0.5189 | none | https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/retrieval-context |
| `q043` identity | `ambiguous` | 3 | 2 | +0.3333 | +0.2573 | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/retrieval-context | none |
| `q044` storage | `ambiguous` | 1 | 1 | +0.3333 | +0.2530 | none | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/retrieval-context |
| `q045` ranking | `ambiguous` | 1 | 1 | +0.0000 | +0.0074 | none | https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/concurrency |
| `q046` policy | `ambiguous` | 2 | 3 | +0.0000 | -0.0976 | https://benchmark.crawlforge.local/cli-config | none |
| `q047` budget | `ambiguous` | 1 | 1 | -0.3333 | -0.1597 | https://benchmark.crawlforge.local/async-http | https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/concurrency |
| `q048` errors | `ambiguous` | 3 | 1 | +0.5000 | +0.3965 | https://benchmark.crawlforge.local/politeness | none |
| `q049` How should timeout, cancellation, and retry delays share ownership of a complete fetch? | `multi_relevant` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/cli-config, https://benchmark.crawlforge.local/sqlite-search | none |
| `q050` How are concurrency slots, crawl spacing, and robots delay combined for one origin? | `multi_relevant` | 1 | 1 | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/url-security | none |
| `q051` Which checks must run again when an allowed public URL redirects? | `multi_relevant` | 2 | 1 | +0.2500 | +0.3276 | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/cli-config | https://benchmark.crawlforge.local/retry-errors |
| `q052` Trace cleaned HTML from structured blocks through chunks into the lexical index. | `multi_relevant` | 1 | 1 | +0.2500 | -0.0352 | https://benchmark.crawlforge.local/cli-config | none |
| `q053` How do BM25 ordering, result limits, and token selection produce bounded context? | `multi_relevant` | 1 | 1 | +0.2500 | +0.1810 | https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/politeness | none |
| `q054` How does the command line expose context search without duplicating application logic? | `multi_relevant` | 1 | 4 | +0.0000 | -0.2475 | https://benchmark.crawlforge.local/retry-errors | none |
| `q055` Compare lifecycle and failure channels for the local tool server and ordinary CLI. | `multi_relevant` | 1 | 1 | +0.0000 | -0.1429 | https://benchmark.crawlforge.local/politeness | none |
| `q056` How are stable identities preserved across canonical URLs, chunks, and provenance rows? | `multi_relevant` | 1 | 1 | +0.0000 | -0.2092 | https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/retry-errors | https://benchmark.crawlforge.local/retrieval-context |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | none | none | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/retrieval-context | https://benchmark.crawlforge.local/politeness |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | none | none | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/url-security, https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/mcp-adapter |
| `q059` OAuth device flow client registration | `negative` | none | none | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/url-security, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/retry-errors |
| `q060` WebSocket streaming endpoint heartbeat interval | `negative` | none | none | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/async-http, https://benchmark.crawlforge.local/concurrency |
| `q061` S3 object versioning and lifecycle retention rules | `negative` | none | none | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/url-security | https://benchmark.crawlforge.local/sqlite-search, https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/concurrency |
| `q062` Kafka consumer offset commit strategy | `negative` | none | none | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/sqlite-search | https://benchmark.crawlforge.local/concurrency, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/retrieval-context |
| `q063` GPU embedding batch size and CUDA memory | `negative` | none | none | +0.0000 | +0.0000 | https://benchmark.crawlforge.local/sqlite-search, https://benchmark.crawlforge.local/cli-config, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/retry-errors, https://benchmark.crawlforge.local/async-http | https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/mcp-adapter |
| `q064` PDF OCR language pack installation | `negative` | none | none | +0.0000 | +0.0000 | none | https://benchmark.crawlforge.local/retrieval-context, https://benchmark.crawlforge.local/content-chunking, https://benchmark.crawlforge.local/mcp-adapter, https://benchmark.crawlforge.local/cli-config, https://benchmark.crawlforge.local/politeness, https://benchmark.crawlforge.local/sqlite-search |

## Limitations

- The corpus and judgments are small, synthetic, and English-focused.
- The semantic baseline uses an English lightweight embedding model.
- Exact vector search is linear in chunk count and embedding dimension.
- Model download is required unless the pinned revision is cached.
- Token counts are deterministic approximations, not model-exact counts.
- Semantic scores are cosine similarities, not calibrated confidence.
- No hybrid retrieval, reranking, or negative-query threshold is used.
- Warning: Bootstrap intervals are exploratory estimates from a small synthetic dataset.
- Warning: Intervals do not establish statistical significance or transferability to real websites.
- Warning: Semantic cosine scores are not calibrated confidence and no negative query threshold was fitted.
