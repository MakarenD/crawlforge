# CrawlForge Hybrid RRF Retrieval Baseline

## Dataset

- Name: `crawlforge-retrieval-baseline`
- Version: `1.0.0`
- Signature: `bb1bf9a8b79f7b47f2850aac362f144d7984196648f592716c7e0d33ff00acfd`
- Retrieval strategy: `hybrid-rrf`
- Queries: 64
- Timestamp: `2026-08-02T08:31:24.491413+00:00`

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
| Corpus processing and indexing | 36.307 ms |

## Chunking configuration

- `max_chars`: `1600`
- `overlap_chars`: `160`
- `target_chars`: `1200`

## Retrieval configuration

- `average_tokenized_length`: `133.7`
- `average_vector_bytes_per_chunk`: `1536.0`
- `batch_size`: `32`
- `bm25_candidate_count_max`: `50`
- `bm25_candidate_limit`: `50`
- `bm25_retrieval_mean_ms`: `2.3444821060081544`
- `bm25_weight`: `1.0`
- `configured_max_sequence_length`: `256`
- `device`: `"cpu"`
- `dimension`: `384`
- `document_encoding_time_ms`: `1627.3022079840302`
- `document_format_version`: `"crawlforge-semantic-document-v1"`
- `document_input_count`: `50`
- `embedded_chunks`: `50`
- `embedding_batch_size`: `32`
- `embedding_cache_hits`: `0`
- `embedding_indexing_time_ms`: `101994.95383398607`
- `embeddings_per_second`: `30.72570033684283`
- `execution_mode`: `"sequential"`
- `failed_embedding_chunks`: `0`
- `fusion_mean_ms`: `0.8630889136129385`
- `fusion_strategy`: `"reciprocal-rank-fusion"`
- `index`: `"sqlite_fts5+sqlite_float32_blob"`
- `invalidated_embeddings`: `0`
- `limit_values`: `[1,3,5,10]`
- `maximum_tokenized_length`: `199`
- `model_cache_size_bytes`: `null`
- `model_fingerprint`: `"99c02b0220fea31cf39e42bcaa37882537882fe061c56eb509db6b0cc61090cf"`
- `model_id`: `"sentence-transformers/all-MiniLM-L6-v2"`
- `model_load_time_ms`: `100351.13308401196`
- `model_revision`: `"1110a243fdf4706b3f48f1d95db1a4f5529b4d41"`
- `normalized`: `true`
- `overlapping_candidate_count_max`: `50`
- `precision`: `"float32"`
- `provenance_context_hydration_mean_ms`: `4.100312334762875`
- `provider`: `"sentence-transformers"`
- `query_format_version`: `"crawlforge-semantic-query-v1"`
- `ranking`: `"reciprocal_rank_fusion"`
- `repeat_latency`: `5`
- `rrf_k`: `60`
- `score_order`: `"descending"`
- `score_type`: `"rrf_score"`
- `semantic_candidate_count_max`: `50`
- `semantic_candidate_limit`: `50`
- `semantic_exact_scan_mean_ms`: `2.881694925500616`
- `semantic_provenance_materialization_mean_ms`: `0.8961546342731986`
- `semantic_query_encoding_mean_ms`: `15.358949290774474`
- `semantic_readiness_check_mean_ms`: `0.23172346086835274`
- `semantic_retrieval_mean_ms`: `22.84833331563187`
- `semantic_snapshot_fetch_mean_ms`: `1.0827480186960565`
- `semantic_vector_decode_mean_ms`: `2.1214096817936197`
- `semantic_weight`: `1.0`
- `sentence_transformers_version`: `"5.6.1"`
- `sqlite_vector_write_time_ms`: `7.5341670017223805`
- `stored_vector_bytes`: `76800`
- `strict_component_success`: `true`
- `sum_component_durations_mean_ms`: `26.055904335252965`
- `token_budget`: `3000`
- `torch_version`: `"2.2.2"`
- `total_hybrid_retrieval_mean_ms`: `26.625194818562836`
- `transformers_version`: `"4.57.6"`
- `truncated_document_fraction`: `0.0`
- `truncated_document_inputs`: `0`
- `unique_candidate_count_max`: `50`
- `warmup_calls`: `3`

## Standard retrieval metrics

Positive-query metrics exclude the explicitly negative queries. Negative queries are evaluated separately with no-result accuracy.

| K | Hit Rate | Precision | Recall | MAP | NDCG |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 78.6% | 78.6% | 54.3% | 0.7857 | 0.7755 |
| 3 | 98.2% | 47.0% | 79.6% | 0.7356 | 0.8327 |
| 5 | 98.2% | 32.5% | 87.1% | 0.7608 | 0.8546 |
| 10 | 100.0% | 18.2% | 93.3% | 0.7832 | 0.8728 |

- MRR: `0.8810`
- No-result accuracy: `0.0%`
- Failed queries: 0

## Metrics by category

| Category | Queries | Hit@5 | Recall@5 | MRR | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exact_term` | 8 | 100.0% | 100.0% | 0.9167 | 0.9236 |
| `code_symbol` | 8 | 100.0% | 100.0% | 1.0000 | 1.0000 |
| `error_lookup` | 8 | 100.0% | 93.8% | 1.0000 | 0.9897 |
| `paraphrase` | 8 | 100.0% | 100.0% | 0.7917 | 0.8432 |
| `conceptual` | 8 | 100.0% | 89.6% | 0.8125 | 0.7938 |
| `ambiguous` | 8 | 87.5% | 52.1% | 0.7083 | 0.6307 |
| `multi_relevant` | 8 | 100.0% | 74.0% | 0.9375 | 0.8013 |
| `negative` | 8 | n/a | n/a | 0.0000 | n/a |

## Strongest queries

| Query | Category | First relevant | Hit@5 | NDCG@5 | Irrelevant token ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| `q056` How are stable identities preserved across canonical URLs, chunks, and provenance rows? | `multi_relevant` | 1 | 100.0% | 1.0000 | 59.9% |
| `q006` same-origin canonical | `exact_term` | 1 | 100.0% | 1.0000 | 78.3% |
| `q014` SQLiteContextIndex | `code_symbol` | 1 | 100.0% | 1.0000 | 78.5% |
| `q023` database is locked with no traceback | `error_lookup` | 1 | 100.0% | 1.0000 | 78.8% |
| `q010` SemaphoreManager | `code_symbol` | 1 | 100.0% | 1.0000 | 87.2% |
| `q025` How should a crawler distinguish connection setup time from a stalled body stream? | `paraphrase` | 1 | 100.0% | 1.0000 | 87.3% |
| `q003` FTS5 virtual table | `exact_term` | 1 | 100.0% | 1.0000 | 88.3% |
| `q009` AsyncCrawler | `code_symbol` | 1 | 100.0% | 1.0000 | 88.3% |
| `q016` ServerRuntime | `code_symbol` | 1 | 100.0% | 1.0000 | 88.6% |
| `q022` CrawlForge MCP support is not installed | `error_lookup` | 1 | 100.0% | 1.0000 | 88.8% |

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
| `q046` policy | `ambiguous` | 2 | 100.0% | 0.4702 | 88.0% |
| `q041` limits | `ambiguous` | 6 | 0.0% | 0.0000 | 63.1% |

## False positives

| Query | Category | Rank | Retrieved source | Section |
| --- | --- | ---: | --- | --- |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 1 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Command boundaries |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 2 | https://benchmark.crawlforge.local/async-http | Async HTTP Timeouts and Lifecycle |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 3 | https://benchmark.crawlforge.local/async-http | Async HTTP Timeouts and Lifecycle > Session lifecycle |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 4 | https://benchmark.crawlforge.local/retry-errors | Retries, Errors, and Backoff > Retry-After handling |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 5 | https://benchmark.crawlforge.local/content-chunking | Cleaning, Chunking, and Token Estimates > Heuristic token estimate |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 6 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Error and output channels |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 7 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Configuration precedence |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 8 | https://benchmark.crawlforge.local/mcp-adapter | Local MCP Adapter Operations > Stdio lifecycle |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 9 | https://benchmark.crawlforge.local/async-http | Async HTTP Timeouts and Lifecycle > Timeout budgets |
| `q057` Kubernetes Helm chart values for the crawler deployment | `negative` | 10 | https://benchmark.crawlforge.local/async-http | Async HTTP Timeouts and Lifecycle > Empty bodies and failures |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 1 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Command boundaries |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 2 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 3 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 4 | https://benchmark.crawlforge.local/retrieval-context | BM25 Retrieval and Context Budgets > BM25 ranking semantics |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 5 | https://benchmark.crawlforge.local/sqlite-search | SQLite FTS5 Index Design > Storage separation |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 6 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Configuration precedence |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 7 | https://benchmark.crawlforge.local/mcp-adapter | Local MCP Adapter Operations > Stable tools |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 8 | https://benchmark.crawlforge.local/retrieval-context | BM25 Retrieval and Context Budgets > Search limits |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 9 | https://benchmark.crawlforge.local/cli-config | CLI, Configuration, and Storage > Storage backends |
| `q058` Redis vector index configuration and HNSW dimensions | `negative` | 10 | https://benchmark.crawlforge.local/retrieval-context | BM25 Retrieval and Context Budgets |

## False negatives

| Query | Category | Missed judgments | First relevant rank |
| --- | --- | --- | ---: |
| `q018` FTS5UnavailableError when SQLite lacks search support | `error_lookup` | q018-j2 | 1 |
| `q041` limits | `ambiguous` | q041-j2 | 6 |
| `q042` cleanup | `ambiguous` | q042-j3 | 1 |
| `q045` ranking | `ambiguous` | q045-j2 | 1 |
| `q046` policy | `ambiguous` | q046-j2, q046-j3 | 2 |
| `q047` budget | `ambiguous` | q047-j3 | 1 |
| `q048` errors | `ambiguous` | q048-j4 | 2 |
| `q050` How are concurrency slots, crawl spacing, and robots delay combined for one origin? | `multi_relevant` | q050-j1 | 1 |
| `q051` Which checks must run again when an allowed public URL redirects? | `multi_relevant` | q051-j4 | 1 |
| `q052` Trace cleaned HTML from structured blocks through chunks into the lexical index. | `multi_relevant` | q052-j4 | 1 |
| `q055` Compare lifecycle and failure channels for the local tool server and ordinary CLI. | `multi_relevant` | q055-j4 | 1 |

## Warm-index retrieval latency

Indexing is excluded. These timings are machine-specific and are not a portable quality gate.

| Samples | Repeats/query | Warm-ups | Mean | Median | P95 | Maximum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 5 | 3 | 26.837 ms | 26.545 ms | 37.061 ms | 75.027 ms |

## CrawlForge-specific context efficiency

These project-specific measurements describe approximate bounded context, not standardized IR metrics or exact model-token savings.

| Measure | Value |
| --- | ---: |
| Mean candidates before context selection | 10.000 |
| Mean returned estimated tokens | 1397.734 |
| Relevant chunks per 1000 estimated tokens | 1.140 |
| Irrelevant estimated-token ratio | 82.4% |
| Mean relevant-source coverage | 93.3% |
| Mean estimated context reduction | 62.2% |

## Benchmark limitations

- The corpus and judgments are small, synthetic, and designed for transparent regression analysis rather than broad external validity.
- Relevance matching uses stable document, canonical source, section, heading, and optional evidence checks; it does not infer semantics.
- Negative-query abstention is strict because retrieval scores are not calibrated confidence values.
- Token counts use CrawlForge's deterministic character heuristic and are not exact for a particular model.
- Retrieval quality does not measure generated-answer correctness, faithfulness, or usefulness.
- Warning: Latency values are warm-index measurements for this machine only.
- Warning: RRF scores are rank-fusion values, not calibrated confidence.
- Warning: Hybrid retrieval requires both BM25 and semantic search to succeed; it does not silently fall back to one component.
- Warning: No negative-query score threshold is applied; hybrid retrieval does not provide calibrated abstention.
- Warning: The canonical hybrid baseline executes BM25 and semantic search sequentially.

## Hybrid baseline interpretation

This is a fixed equal-weight Reciprocal Rank Fusion baseline. It combines BM25 and semantic ranks without treating either raw score as calibrated confidence. Interpret improvements and regressions through the checked three-strategy comparison; this run does not include reranking or a negative-query threshold.
