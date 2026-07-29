# CrawlForge Semantic Retrieval Baseline

## Dataset

- Name: `crawlforge-retrieval-baseline`
- Version: `1.0.0`
- Signature: `bb1bf9a8b79f7b47f2850aac362f144d7984196648f592716c7e0d33ff00acfd`
- Retrieval strategy: `semantic-exact-cosine`
- Queries: 64
- Timestamp: `2026-07-29T15:30:27.877849+00:00`

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
| Corpus processing and indexing | 26.021 ms |

## Chunking configuration

- `max_chars`: `1600`
- `overlap_chars`: `160`
- `target_chars`: `1200`

## Retrieval configuration

- `average_tokenized_length`: `133.7`
- `average_vector_bytes_per_chunk`: `1536.0`
- `batch_size`: `32`
- `configured_max_sequence_length`: `256`
- `device`: `"cpu"`
- `dimension`: `384`
- `document_encoding_time_ms`: `1210.536874976242`
- `document_format_version`: `"crawlforge-semantic-document-v1"`
- `document_input_count`: `50`
- `embedded_chunks`: `50`
- `embedding_batch_size`: `32`
- `embedding_cache_hits`: `0`
- `embedding_indexing_time_ms`: `4847.404665983049`
- `embeddings_per_second`: `41.303987539397596`
- `exact_scan_complexity`: `"O(number_of_chunks * embedding_dimension)"`
- `exact_vector_scan_mean_ms`: `1.2329409339381583`
- `failed_embedding_chunks`: `0`
- `index`: `"sqlite_float32_blob"`
- `invalidated_embeddings`: `0`
- `limit_values`: `[1,3,5,10]`
- `loaded_vector_bytes`: `76800`
- `loaded_vector_memory_estimate_bytes`: `76800`
- `maximum_tokenized_length`: `199`
- `model_cache_size_bytes`: `183156830`
- `model_fingerprint`: `"99c02b0220fea31cf39e42bcaa37882537882fe061c56eb509db6b0cc61090cf"`
- `model_id`: `"sentence-transformers/all-MiniLM-L6-v2"`
- `model_load_time_ms`: `3621.3732920004986`
- `model_revision`: `"1110a243fdf4706b3f48f1d95db1a4f5529b4d41"`
- `normalized`: `true`
- `precision`: `"float32"`
- `provenance_materialization_mean_ms`: `0.3974788294039701`
- `provider`: `"sentence-transformers"`
- `query_encoding_mean_ms`: `5.2244024672552705`
- `query_format_version`: `"crawlforge-semantic-query-v1"`
- `ranking`: `"exact_cosine_similarity"`
- `readiness_check_mean_ms`: `0.12847792786825737`
- `repeat_latency`: `5`
- `score_order`: `"descending"`
- `score_type`: `"cosine_similarity"`
- `sentence_transformers_version`: `"5.6.1"`
- `sqlite_snapshot_fetch_mean_ms`: `0.4249789910470265`
- `sqlite_snapshot_scope`: `"compatible vectors and complete chunk provenance"`
- `sqlite_vector_write_time_ms`: `6.009416014421731`
- `stored_vector_bytes`: `76800`
- `token_budget`: `3000`
- `torch_version`: `"2.2.2"`
- `total_semantic_retrieval_mean_ms`: `8.457797611634689`
- `transformers_version`: `"4.57.6"`
- `truncated_document_fraction`: `0.0`
- `truncated_document_inputs`: `0`
- `vector_decode_mean_ms`: `0.9824412948943547`
- `warmup_calls`: `3`

## Standard retrieval metrics

Positive-query metrics exclude the explicitly negative queries. Negative queries are evaluated separately with no-result accuracy.

| K | Hit Rate | Precision | Recall | MAP | NDCG |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 78.6% | 78.6% | 51.5% | 0.7857 | 0.7347 |
| 3 | 91.1% | 40.5% | 68.9% | 0.6567 | 0.7545 |
| 5 | 98.2% | 30.7% | 82.3% | 0.6970 | 0.8102 |
| 10 | 98.2% | 17.9% | 91.2% | 0.7282 | 0.8327 |

- MRR: `0.8563`
- No-result accuracy: `0.0%`
- Failed queries: 0

## Metrics by category

| Category | Queries | Hit@5 | Recall@5 | MRR | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exact_term` | 8 | 87.5% | 75.0% | 0.8125 | 0.7957 |
| `code_symbol` | 8 | 100.0% | 93.8% | 0.8438 | 0.8724 |
| `error_lookup` | 8 | 100.0% | 93.8% | 1.0000 | 0.9689 |
| `paraphrase` | 8 | 100.0% | 100.0% | 0.8542 | 0.8835 |
| `conceptual` | 8 | 100.0% | 89.6% | 0.9062 | 0.8559 |
| `ambiguous` | 8 | 100.0% | 50.0% | 0.6708 | 0.5867 |
| `multi_relevant` | 8 | 100.0% | 74.0% | 0.9062 | 0.7080 |
| `negative` | 8 | n/a | n/a | 0.0000 | n/a |

## Strongest queries

| Query | Category | First relevant | Hit@5 | NDCG@5 | Irrelevant token ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `q006` same-origin canonical | `exact_term` | 1 | 100.0% | 1.0000 | 78.4% |
| `q029` Why must every resolved address be checked even when the hostname is permitted? | `paraphrase` | 1 | 100.0% | 1.0000 | 80.0% |
| `q010` SemaphoreManager | `code_symbol` | 1 | 100.0% | 1.0000 | 87.2% |
| `q025` How should a crawler distinguish connection setup time from a stalled body stream? | `paraphrase` | 1 | 100.0% | 1.0000 | 87.3% |
| `q021` redirect limit exceeded | `error_lookup` | 1 | 100.0% | 1.0000 | 88.1% |
| `q003` FTS5 virtual table | `exact_term` | 1 | 100.0% | 1.0000 | 88.3% |
| `q009` AsyncCrawler | `code_symbol` | 1 | 100.0% | 1.0000 | 88.3% |
| `q024` empty response body or failed fetch | `error_lookup` | 1 | 100.0% | 1.0000 | 88.6% |
| `q005` token budget | `exact_term` | 1 | 100.0% | 1.0000 | 88.6% |
| `q022` CrawlForge MCP support is not installed | `error_lookup` | 1 | 100.0% | 1.0000 | 88.6% |

## Weakest queries

| Query | Category | First relevant | Hit@5 | NDCG@5 | Irrelevant token ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `q064` PDF OCR language pack installation | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q063` GPU embedding batch size and CUDA memory | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q062` Kafka consumer offset commit strategy | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q061` S3 object versioning and lifecycle retention rules | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q060` WebSocket streaming endpoint heartbeat interval | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q059` OAuth device flow client registration | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | none | 0.0% | 0.0000 | 100.0% |
| `q007` standard output | `exact_term` | none | 0.0% | 0.0000 | 100.0% |
| `q046` policy | `ambiguous` | 3 | 100.0% | 0.3726 | 88.4% |

## False positives

| Query | Category | Rank | Retrieved source | Section |
| --- | --- | ---: | --- | --- |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 1 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Command boundaries |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 2 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Error and output channels |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 3 | https://benchmark.crawlforge.local/mcp-adapter | Local MCP Adapter Operations > Stable tools |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 4 | https://benchmark.crawlforge.local/content-chunking | Cleaning, Chunking, and Token Estimates > Heuristic token estimate |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 5 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 6 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Storage backends |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 7 | https://benchmark.crawlforge.local/politeness | Rate Limits and Robots Policy |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 8 | https://benchmark.crawlforge.local/politeness | Rate Limits and Robots Policy > Crawl-delay scheduling |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 9 | https://benchmark.crawlforge.local/politeness | Rate Limits and Robots Policy > Robots decisions |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 10 | https://benchmark.crawlforge.local/retry-errors | Retries, Errors, and Backoff > Retry-After handling |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 1 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Command boundaries |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 2 | https://benchmark.crawlforge.local/url-security | URL Identity and Network Safety > Redirect validation |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 3 | https://benchmark.crawlforge.local/retrieval-context | BM25 Retrieval and Context Budgets > BM25 ranking semantics |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 4 | https://benchmark.crawlforge.local/content-chunking | Cleaning, Chunking, and Token Estimates > Heuristic token estimate |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 5 | https://benchmark.crawlforge.local/retrieval-context | BM25 Retrieval and Context Budgets > Search limits |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 6 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 7 | https://benchmark.crawlforge.local/mcp-adapter | Local MCP Adapter Operations > Stable tools |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 8 | https://benchmark.crawlforge.local/url-security | URL Identity and Network Safety > URL normalization |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 9 | https://benchmark.crawlforge.local/retrieval-context | BM25 Retrieval and Context Budgets |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 10 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design > Storage separation |

## False negatives

| Query | Category | Missed judgments | First relevant rank |
| --- | --- | --- | ---: |
| `q007` standard output | `exact_term` | q007-j1, q007-j2 | none |
| `q036` How do request capacity and start-time spacing solve different politeness problems? | `conceptual` | q036-j2 | 1 |
| `q039` What does a strong context reduction percentage prove and not prove? | `conceptual` | q039-j2 | 1 |
| `q041` limits | `ambiguous` | q041-j4 | 5 |
| `q042` cleanup | `ambiguous` | q042-j3 | 3 |
| `q043` identity | `ambiguous` | q043-j3 | 2 |
| `q045` ranking | `ambiguous` | q045-j2 | 1 |
| `q046` policy | `ambiguous` | q046-j2, q046-j3 | 3 |
| `q047` budget | `ambiguous` | q047-j2, q047-j3 | 1 |
| `q048` errors | `ambiguous` | q048-j4 | 1 |
| `q055` Compare lifecycle and failure channels for the local tool server and ordinary CLI. | `multi_relevant` | q055-j4 | 1 |

## Warm-index retrieval latency

Indexing is excluded. These timings are machine-specific and are not a portable quality gate.

| Samples | Repeats/query | Warm-ups | Mean | Median | P95 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 5 | 3 | 8.602 ms | 8.532 ms | 10.779 ms | 12.063 ms |

## CrawlForge-specific context efficiency

These project-specific measurements describe approximate bounded context, not standardized IR metrics or exact model-token savings.

| Measure | Value |
| --- | ---: |
| Mean candidates before context selection | 10.000 |
| Mean returned estimated tokens | 1372.672 |
| Relevant chunks per 1000 estimated tokens | 1.138 |
| Irrelevant estimated-token ratio | 82.4% |
| Mean relevant-source coverage | 91.2% |
| Mean estimated context reduction | 58.0% |

## Benchmark limitations

- The corpus and judgments are small, synthetic, and designed for transparent regression analysis rather than broad external validity.
- Relevance matching uses stable document, canonical source, section, heading, and optional evidence checks; it does not infer semantics.
- Negative-query abstention is strict because retrieval scores are not calibrated confidence values.
- Token counts use CrawlForge's deterministic character heuristic and are not exact for a particular model.
- Retrieval quality does not measure generated-answer correctness, faithfulness, or usefulness.
- Warning: Latency values are warm-index measurements for this machine only.
- Warning: Semantic cosine similarity is not calibrated confidence.
- Warning: No negative-query score threshold is applied; embeddings alone do not provide calibrated abstention.
- Warning: Exact semantic search is O(number_of_chunks × embedding_dimension) and is intended for small local indexes.

## Semantic baseline interpretation

This is an isolated exact-cosine baseline. Its effect should be interpreted only through the paired BM25 comparison on the same dataset and chunks; it does not include fusion, reranking, or a negative-query threshold.
