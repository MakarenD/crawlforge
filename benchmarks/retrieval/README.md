# CrawlForge retrieval baseline

This directory contains a deterministic, offline, section-level retrieval
benchmark. All ten HTML documents and all query text were written specifically
for this fixture. They do not copy third-party documentation and require no
network access.

## Files

- `manifest.json` declares the corpus, document metadata, and the authoritative
  section inventory.
- `documents/*.html` contains the original technical source pages.
- `queries.jsonl` contains one query record per non-empty line.

`manifest.json` uses schema version 1. Every document has a stable
`document_id`, a path relative to this directory, a synthetic HTTPS URL, a
title, and three or four declared sections. Each section has a stable
`section_id` and an ordered `heading_path`.

Each query record has this shape:

```json
{
  "query_id": "q001",
  "query": "example",
  "category": "exact_term",
  "relevant_sources": [
    {
      "judgment_id": "q001-j1",
      "document_id": "example-document",
      "section_id": "example-section",
      "heading_path": ["Page title", "Section heading"],
      "relevance": 3,
      "evidence": "optional exact source span"
    }
  ]
}
```

## Judgment matching

Consumers should resolve a judgment in this order:

1. match `document_id` to a manifest document;
2. match `section_id` within that document;
3. require `heading_path` to equal the manifest path exactly;
4. when `evidence` is present, verify it as an exact visible-text span within
   that section after HTML text extraction.

The identifiers are authoritative. URLs, titles, and evidence are useful audit
material but are not substitutes for the document-and-section key.

Relevance grades follow a four-point scale:

- `3`: directly answers the query or contains the exact requested fact;
- `2`: substantially useful supporting material;
- `1`: related context that may help disambiguate or complete an answer;
- `0`: not relevant. Version 1 stores only positive grades, so grade zero is
  represented by the absence of a judgment.

## Query categories

There are exactly 64 queries and exactly eight queries in each category:

- `exact_term`
- `code_symbol`
- `error_lookup`
- `paraphrase`
- `conceptual`
- `ambiguous`
- `multi_relevant`
- `negative`

Several ambiguous and multi-relevant queries deliberately have more than one
graded source. Deliberate lexical overlap across documents makes the benchmark
exercise section selection rather than simple document vocabulary matching.

Negative queries use terms absent from the corpus and have an empty
`relevant_sources` array. They measure whether a retriever can avoid asserting
a supported in-corpus answer; they do not prove that the subject is false,
unimportant, or absent from external knowledge.
