"""Transparent stable-source relevance matching without semantic inference."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import NamedTuple

from crawlforge.evaluation.models import (
    EvaluationDataset,
    EvaluationDocument,
    EvaluationQuery,
    RelevanceJudgment,
    RetrievedItem,
)


class RelevanceMatchResult(NamedTuple):
    """Annotated ranking and the set of uniquely credited judgments."""

    items: tuple[RetrievedItem, ...]
    matched_judgment_ids: frozenset[str]


def match_retrieved_items(
    dataset: EvaluationDataset,
    query: EvaluationQuery,
    retrieved: tuple[RetrievedItem, ...],
) -> RelevanceMatchResult:
    """Credit each source/section judgment at most once in retrieval order."""
    available = {
        judgment.judgment_id: judgment
        for judgment in query.relevant_sources
        if judgment.relevance > 0
    }
    documents = {document.document_id: document for document in dataset.documents}
    annotated: list[RetrievedItem] = []
    matched: set[str] = set()

    for item in retrieved:
        candidates = [
            judgment
            for judgment in available.values()
            if judgment.judgment_id not in matched
            and _matches(item, judgment, documents)
        ]
        if not candidates:
            annotated.append(
                replace(
                    item,
                    relevance_grade=0,
                    matched_judgment_id=None,
                )
            )
            continue
        selected = min(
            candidates,
            key=lambda judgment: (-judgment.relevance, judgment.judgment_id),
        )
        matched.add(selected.judgment_id)
        annotated.append(
            replace(
                item,
                relevance_grade=selected.relevance,
                matched_judgment_id=selected.judgment_id,
            )
        )
    return RelevanceMatchResult(tuple(annotated), frozenset(matched))


def judgment_matches(
    dataset: EvaluationDataset,
    item: RetrievedItem,
    judgment: RelevanceJudgment,
) -> bool:
    """Expose the documented single-item matching rule for focused tests."""
    documents = {document.document_id: document for document in dataset.documents}
    return _matches(item, judgment, documents)


def _matches(
    item: RetrievedItem,
    judgment: RelevanceJudgment,
    documents: Mapping[str, EvaluationDocument],
) -> bool:
    document = documents.get(judgment.document_id)
    document_url = document.url if document is not None else None
    canonical_source = judgment.canonical_source or document_url
    source_matches = item.document_id == judgment.document_id or (
        isinstance(canonical_source, str) and item.canonical_url == canonical_source
    )
    if not source_matches:
        return False

    expected_heading = judgment.heading_path
    if judgment.section_id is not None:
        sections = document.sections if document is not None else ()
        section = next(
            (
                candidate
                for candidate in sections
                if candidate.section_id == judgment.section_id
            ),
            None,
        )
        if section is None:
            return False
        expected_heading = section.heading_path
        if item.section_id != judgment.section_id:
            return False
    if expected_heading and item.heading_path != expected_heading:
        return False
    if judgment.evidence is not None and _normalize(
        judgment.evidence
    ) not in _normalize(item.text):
        return False
    return True


def _normalize(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return re.sub(r"\s+([.,;:!?])", r"\1", normalized)
