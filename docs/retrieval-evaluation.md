# Retrieval evaluation

CrawlForge evaluates retrieval separately from answer generation. A retrieval
benchmark asks whether the index returned the right source sections, how early
they appeared, and how much irrelevant context was included. It does not claim
that a downstream generated answer is correct, faithful, or useful.

The benchmark compares deterministic SQLite FTS5/BM25, optional local semantic
retrieval, and fixed rank-fused hybrid retrieval:

```text
versioned local corpus
    -> ContentProcessor
    -> TextChunker
    -> ContextEngine indexing
    -> public BM25, semantic, or hybrid application service
    -> stable relevance matching
    -> retrieval and context metrics
    -> JSON or Markdown report
```

Evaluation lives in `crawlforge.evaluation`. It does not execute retrieval SQL
or reimplement any ranking. `BM25ContextEngineStrategy`,
`SemanticContextEngineStrategy`, and `HybridContextEngineStrategy` adapt public
results into a strategy-neutral `RetrievedItem`. The hybrid evaluator receives
the already fused production ranking; it does not implement RRF. Future
reranked implementations can use the same `RetrievalStrategy` protocol,
dataset, judgments, metrics, and reporters.

## Offline dataset

The versioned baseline is in `benchmarks/retrieval`:

```text
benchmarks/retrieval/
├── manifest.json
├── documents/
├── queries.jsonl
└── README.md
```

All documents and queries were written for this benchmark. Automated tests and
the full run use no external network access.

Reports record a SHA-256 dataset signature over the exact manifest, query, and
document bytes. Filtering queries retains the complete frozen-dataset
signature, and paired comparison refuses different signatures, chunk settings,
query order, K values, or token budgets.

Installed wheels contain the same baseline as package data. Both `evaluate
validate` and `evaluate run` use it when `--dataset` is omitted, regardless of
the current working directory.

`manifest.json` declares the dataset schema, name, version, documents, synthetic
source URLs, and stable sections. A document entry has this shape:

```json
{
  "document_id": "concurrency",
  "path": "documents/concurrency.html",
  "url": "https://benchmark.crawlforge.local/concurrency",
  "title": "Bounded Concurrency and Cancellation",
  "sections": [
    {
      "section_id": "semaphore-hierarchy",
      "heading_path": [
        "Bounded Concurrency and Cancellation",
        "Semaphore hierarchy"
      ]
    }
  ]
}
```

`queries.jsonl` contains one JSON object per non-empty line:

```json
{
  "query_id": "q017",
  "query": "token_budget must be greater than zero",
  "category": "error_lookup",
  "relevant_sources": [
    {
      "judgment_id": "q017-j1",
      "document_id": "retrieval-context",
      "section_id": "token-budgets",
      "heading_path": [
        "BM25 Retrieval and Context Budgets",
        "Token budgets"
      ],
      "relevance": 3,
      "evidence": "token_budget must be greater than zero"
    }
  ]
}
```

The checked-in dataset covers `exact_term`, `code_symbol`, `error_lookup`,
`paraphrase`, `conceptual`, `ambiguous`, `multi_relevant`, and `negative`
queries. Deliberately similar sections distinguish connect and read timeouts,
global and per-domain limits, temporary and permanent errors, search limits and
token budgets, and private-network and allowed-domain policies.

## Relevance judgments and matching

A relevance judgment is human-authored ground truth, not a score inferred from
the retrieved text. Grades are:

- `0`: irrelevant;
- `1`: partially useful;
- `2`: relevant supporting material;
- `3`: a direct and complete source for the query.

Version 1 stores positive judgments. Grade zero is represented by an unjudged
retrieval result.

Matching is deterministic and follows this order:

1. match the stable dataset `document_id`, or its canonical source URL;
2. when a section is specified, require the stable `section_id` and exact
   `heading_path`;
3. when an evidence span is specified, require its normalized visible text in
   the retrieved chunk.

A document match cannot substitute for a missing section match. Each judgment
is credited at most once in a ranking, so repeated chunks from one judged
section cannot inflate recall, AP, or NDCG. Ground truth never depends on a
generated chunk ID.

## Standard retrieval metrics

Positive-query metrics are reported at each configured K, normally 1, 3, 5,
and 10.

### Hit Rate@K

The fraction of positive queries with at least one relevant result in the
first K positions.

### Precision@K

The number of credited relevant results among the first K divided by K.
Returning fewer than K results does not change the denominator.

### Recall@K

The fraction of all known positive judgments credited in the first K results.
Multiple chunks cannot credit the same judgment twice.

### MRR

Mean reciprocal rank averages `1 / rank` for the first relevant result. A query
with no relevant result contributes zero.

### MAP@K

Average precision accumulates precision at each credited relevant rank and
normalizes by the smaller of K and the number of known positive judgments.
MAP@K is the mean AP@K across positive queries.

### NDCG@K

NDCG uses the graded gain `2^relevance - 1`, discounts lower ranks by
`log2(rank + 1)`, and divides by the ideal ordering of the known judgments.

### No-result accuracy

Negative queries have no positive ground truth. Raw SQLite FTS5 BM25 scores are
not calibrated confidence values, so the baseline uses a strict interpretation:
a negative query is correct only when search returns no candidate. Any returned
candidate is recorded as a false positive. This measures lexical abstention,
not whether the topic is absent from external knowledge.

Standard positive-query metrics exclude negative queries. No-result accuracy
is calculated only across negative queries.

## CrawlForge-specific context measurements

These measurements are reported separately and are not presented as standard
information-retrieval metrics:

- candidate count before bounded context selection;
- returned estimated tokens;
- relevant chunks per 1000 estimated tokens;
- irrelevant estimated-token ratio;
- relevant-source coverage inside the bounded context;
- estimated context reduction relative to unique retrieved source documents.

The bounded selection preserves retrieval order, removes repeated content
hashes, and includes only complete chunks that fit the budget. Token counts use
the existing deterministic character heuristic. They are approximate and are
not exact savings for any particular model.

## Latency method

Corpus processing and indexing are measured separately. Search latency uses the
already built index:

1. run three warm-up searches;
2. run each selected query `--repeat-latency` times;
3. record mean, median, nearest-rank p95, and maximum latency.

Latency is machine-specific and is never a hard CI quality floor. The CI subset
asserts deterministic retrieval-quality floors only.

## Validate and run

Validate the complete dataset before indexing:

```bash
uv run crawlforge evaluate validate \
  --dataset benchmarks/retrieval
```

Add `--json` for a single machine-readable validation summary on standard
output. Validation checks:

- unique document, section, query, and judgment IDs;
- one-to-one section heading paths and unique relevance targets per query;
- supported manifest version;
- safe relative document paths;
- non-empty corpus and queries;
- valid categories and relevance grades;
- existing relevant documents and sections;
- exact manifest heading paths present in HTML;
- evidence spans present in visible document text;
- no positive ground truth on negative queries;
- positive ground truth on every other category.

Run the full BM25 baseline:

```bash
uv run crawlforge evaluate run \
  --dataset benchmarks/retrieval \
  --database .crawlforge/evaluation.db \
  --output reports/bm25-baseline.json \
  --format json \
  --limit-values 1,3,5,10 \
  --token-budget 3000 \
  --repeat-latency 5
```

The selected evaluation database is disposable. `run` removes that exact
database and its SQLite sidecars before indexing so stale corpus data cannot
contaminate the result. The database and output must be outside the dataset and
must not be the same path.

Use `--format markdown` for the human-readable analysis. `--category` selects
one validated category, and repeatable `--query-id` options select individual
queries. Custom datasets, filtered runs, and non-baseline metric settings
require an explicit `--output` path so they cannot replace the canonical
baseline report by accident. Filtering never skips validation of the complete
dataset. `--json` changes only the concise standard-output summary; the
complete report is still written to `--output`.

Successful validation and evaluation return exit code 0. Invalid input,
configuration, paths, or dataset structure use the established CLI error code
2 with no traceback. A completed run that records a per-query retrieval failure
writes the report and returns code 1. Diagnostic output is not mixed into JSON
standard output.

## Semantic and hybrid baselines

The semantic baseline uses the production provider and exact search path, not
test vectors. Install the optional runtime and run the pinned model on CPU:

```bash
uv run --extra semantic crawlforge evaluate run \
  --strategy semantic \
  --dataset benchmarks/retrieval \
  --database .crawlforge/semantic-evaluation.db \
  --output reports/semantic-baseline.json \
  --format json \
  --limit-values 1,3,5,10 \
  --token-budget 3000 \
  --repeat-latency 5 \
  --device cpu
```

The paired command builds one corpus and evaluates BM25 before semantic search
against the same chunks:

```bash
uv run --extra semantic crawlforge evaluate compare \
  --strategies bm25,semantic \
  --dataset benchmarks/retrieval \
  --database .crawlforge/evaluation-compare.db \
  --output reports/bm25-vs-semantic.md \
  --format markdown \
  --limit-values 1,3,5,10 \
  --token-budget 3000 \
  --repeat-latency 5 \
  --bootstrap-samples 5000 \
  --bootstrap-seed 20260729 \
  --device cpu
```

The comparison reports aggregate and category deltas, first relevant ranks,
different retrieved sources, semantic wins, BM25 wins, both-success and
both-fail queries, and semantic regressions. Query failures remain isolated and
visible instead of aborting the other strategy. Paired bootstrap intervals use
query-level metric deltas from non-failed positive queries and a fixed seed.
They are exploratory and are not CI gates or claims of statistical
significance.

The current frozen run found higher semantic Hit@5 and Recall@5, but lower MRR,
MAP@5, and negative-query no-result accuracy. Semantic improved paraphrase and
conceptual category MRR while regressing exact terms and code symbols. See the
[semantic details](semantic-retrieval.md),
[semantic report](../reports/semantic-baseline.md), and
[paired report](../reports/bm25-vs-semantic.md).

Build the fixed hybrid baseline through the production `HybridRetriever`:

```bash
uv run --extra semantic crawlforge evaluate run \
  --strategy hybrid \
  --dataset benchmarks/retrieval \
  --database .crawlforge/hybrid-evaluation.db \
  --output reports/hybrid-baseline.json \
  --format json \
  --limit-values 1,3,5,10 \
  --token-budget 3000 \
  --repeat-latency 5 \
  --device cpu
```

The generic comparison accepts two or more unique strategies. The exact legacy
`bm25,semantic` invocation retains its original schema; the canonical triple
uses the versioned multi-comparison schema:

```bash
uv run --extra semantic crawlforge evaluate compare \
  --strategies bm25,semantic,hybrid \
  --dataset benchmarks/retrieval \
  --database .crawlforge/evaluation-compare.db \
  --output reports/bm25-vs-semantic-vs-hybrid.md \
  --format markdown \
  --limit-values 1,3,5,10 \
  --token-budget 3000 \
  --repeat-latency 5 \
  --bootstrap-samples 5000 \
  --bootstrap-seed 20260729 \
  --device cpu
```

| Metric | BM25 | Semantic | Hybrid |
| --- | ---: | ---: | ---: |
| Hit@5 | 0.9643 | 0.9821 | 0.9821 |
| Precision@5 | 0.2893 | 0.3071 | 0.3250 |
| Recall@5 | 0.8021 | 0.8229 | 0.8705 |
| MRR | 0.8681 | 0.8563 | 0.8810 |
| MAP@5 | 0.7294 | 0.6970 | 0.7608 |
| NDCG@5 | 0.8100 | 0.8102 | 0.8546 |
| Negative no-result accuracy | 0.1250 | 0.0000 | 0.0000 |

The multi-strategy report keeps complete per-query final rankings and adds
diagnostic, non-standard measures: BM25/semantic overlap at K=1, 3, 5, and 10;
unique relevant coverage; a ground-truth oracle union; fusion recovery;
component contribution and retention summaries; named query outcome buckets;
and paired bootstrap intervals. The oracle uses judgments and cannot be used at
search time. The fixed RRF baseline recovered 19 of 23 component-only relevant
judgments at K=5, while all eight negative queries still received results.
See the [hybrid implementation](hybrid-retrieval.md),
[hybrid report](../reports/hybrid-baseline.md), and
[three-strategy comparison](../reports/bm25-vs-semantic-vs-hybrid.md).

## Reading reports

The JSON report contains:

- dataset and strategy identities;
- retrieval and chunking configuration;
- corpus and indexing statistics;
- every query, expected source, retrieved item, rank, grade, matched judgment,
  missed judgment, latency sample, and context measurement;
- aggregate and per-category metrics;
- warm-index latency and context summaries;
- `worst_queries`, failures, and warnings.

The Markdown strategy reports present the same run as overall and category
tables, strongest and weakest queries, false positives, false negatives,
latency, context efficiency, limitations, and a bounded strategy-specific
conclusion. Pair and multi-strategy reports keep comparison evidence separate
from each strategy's own ranking. Reports omit dataset, database, cache, and
output machine paths.

## Extending the benchmark

To add a query:

1. choose a unique query and judgment ID;
2. use one of the supported categories;
3. reference an existing manifest document and section;
4. assign grades by manual source review;
5. add an evidence span when an exact visible fact is important;
6. run `uv run crawlforge evaluate validate`;
7. inspect the query's complete ranking and category metrics.

To add a document, write original local HTML, add its safe relative path and
complete section inventory to the manifest, then add independently reviewed
queries. Do not tune document wording merely to raise a metric.

To compare another retrieval strategy, implement:

```python
class RetrievalStrategy(Protocol):
    @property
    def name(self) -> str: ...

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[RetrievedItem]: ...
```

Return items in strategy rank order with stable source provenance. The
evaluator does not sort by score. Keep the same dataset version and judgments
when comparing BM25, vector, hybrid, or reranked results.

## Decision boundary after fixed RRF

This stage adds deterministic equal-weight rank fusion, but no reranking,
learned weights, answer generation, LLM judge, or calibrated abstention. Hybrid
improved aggregate Recall@5, MRR, MAP@5, and NDCG@5 on this frozen corpus, yet
it regressed individual BM25 and semantic wins and returned candidates for all
negative queries. The small synthetic sample does not establish generalization.

The next evidence-producing step should be a larger real-world benchmark.
Reranking is justified only if candidate-union analysis shows relevant chunks
that RRF consistently orders poorly. Abstention needs a separate calibration
dataset, risk/coverage analysis, and a clarification policy rather than a
threshold fitted to these eight negative queries.
