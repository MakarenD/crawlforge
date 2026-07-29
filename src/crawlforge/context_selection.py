"""Shared complete-chunk selection for lexical and semantic retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from crawlforge.context_models import SourceReference, TextChunk


class ContextCandidate(Protocol):
    """Minimal ranked hit boundary used by bounded context selection."""

    @property
    def chunk(self) -> TextChunk:
        """Return the complete candidate chunk."""

    @property
    def source(self) -> SourceReference:
        """Return the candidate's source provenance."""


@dataclass(frozen=True, slots=True)
class BoundedContext[T: ContextCandidate]:
    """Selected complete chunks and aggregate token measurements."""

    hits: tuple[T, ...]
    candidate_count: int
    total_size_chars: int
    estimated_tokens: int
    source_estimated_tokens: int
    estimated_context_reduction: float


def select_bounded_context[T: ContextCandidate](
    candidates: Sequence[T],
    *,
    token_budget: int,
) -> BoundedContext[T]:
    """Deduplicate by content and select complete chunks within a strict budget."""
    if token_budget <= 0:
        raise ValueError("token_budget must be greater than zero")

    deduplicated: list[T] = []
    seen_hashes: set[str] = set()
    for hit in candidates:
        if hit.chunk.content_hash in seen_hashes:
            continue
        seen_hashes.add(hit.chunk.content_hash)
        deduplicated.append(hit)

    selected: list[T] = []
    estimated_tokens = 0
    total_size_chars = 0
    for hit in deduplicated:
        next_tokens = estimated_tokens + hit.chunk.estimated_tokens
        if next_tokens > token_budget:
            continue
        selected.append(hit)
        estimated_tokens = next_tokens
        total_size_chars += hit.chunk.size_chars

    source_estimated_tokens = sum(
        hit.source.source_estimated_tokens for hit in _unique_sources(deduplicated)
    )
    reduction = (
        max(0.0, min(1.0, 1 - estimated_tokens / source_estimated_tokens))
        if source_estimated_tokens
        else 0.0
    )
    return BoundedContext(
        hits=tuple(selected),
        candidate_count=len(candidates),
        total_size_chars=total_size_chars,
        estimated_tokens=estimated_tokens,
        source_estimated_tokens=source_estimated_tokens,
        estimated_context_reduction=reduction,
    )


def _unique_sources[T: ContextCandidate](hits: Sequence[T]) -> list[T]:
    unique: list[T] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.source.document_id in seen:
            continue
        seen.add(hit.source.document_id)
        unique.append(hit)
    return unique
