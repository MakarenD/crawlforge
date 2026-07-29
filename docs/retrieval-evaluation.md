# Retrieval evaluation

CrawlForge evaluates retrieval separately from answer generation. A retrieval
benchmark asks whether the index returned the right source sections, how early
they appeared, and how much irrelevant context was included. It does not claim
that a downstream generated answer is correct, faithful, or useful.

The current benchmark is a deterministic SQLite FTS5/BM25 baseline:

```text
versioned local corpus
    -> ContentProcessor
    -> TextChunker
    -> ContextEngine indexing
    -> public ContextEngine.search()
    -> stable relevance matching
    -> retrieval and context metrics
    -> JSON or Markdown report
```

Evaluation lives in `crawlforge.evaluation`. It does not execute FTS5 SQL or
reimplement BM25 ranking. `BM25ContextEngineStrategy` adapts the existing public
search result into a strategy-neutral `RetrievedItem`. Future vector, hybrid, or
reranked implementations can implement `RetrievalStrategy` and use the same
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
crawlforge evaluate validate \
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
crawlforge evaluate run \
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

The Markdown report presents the same run as overall and category tables,
strongest and weakest queries, false positives, false negatives, latency,
context efficiency, limitations, and a bounded conclusion about whether the
observed vocabulary mismatches justify semantic retrieval. Reports omit the
dataset, database, and output machine paths.

## Extending the benchmark

To add a query:

1. choose a unique query and judgment ID;
2. use one of the supported categories;
3. reference an existing manifest document and section;
4. assign grades by manual source review;
5. add an evidence span when an exact visible fact is important;
6. run `crawlforge evaluate validate`;
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

## Semantic-search decision boundary

This stage does not include embeddings, vector storage, hybrid retrieval,
fusion, reranking, answer generation, or an LLM judge. The baseline should
justify a semantic stage only through concrete failures: relevant sections
missed because the query and source use different vocabulary, relevant sources
consistently ranked below lexical distractors, or bounded context dominated by
irrelevant lexical matches. Short ambiguous queries and false positives on
negative queries remain separate problems; semantic retrieval does not
automatically solve abstention or intent ambiguity.
