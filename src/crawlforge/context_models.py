"""Typed models shared by content processing, indexing, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

BlockKind = Literal["heading", "paragraph", "list", "code", "table"]


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    """One ordered semantic block extracted from a source document."""

    kind: BlockKind
    text: str
    markdown: str
    heading_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Cleaned content and provenance for one fetched web document."""

    id: str
    url: str
    canonical_url: str
    title: str
    text: str
    markdown: str | None
    status_code: int
    content_type: str
    fetched_at: datetime
    content_hash: str
    metadata: dict[str, object]
    source_size_bytes: int
    cleaned_size_bytes: int
    source_estimated_tokens: int
    cleaned_estimated_tokens: int
    blocks: tuple[DocumentBlock, ...]


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A bounded, searchable text unit derived from a source document."""

    id: str
    document_id: str
    ordinal: int
    source_url: str
    document_title: str
    heading_path: tuple[str, ...]
    text: str
    size_chars: int
    estimated_tokens: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Compact source provenance attached to a search result."""

    document_id: str
    url: str
    canonical_url: str
    title: str
    status_code: int
    content_type: str
    fetched_at: datetime
    source_size_bytes: int
    source_estimated_tokens: int


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A ranked text chunk and its source."""

    chunk: TextChunk
    rank: int
    bm25_score: float
    source: SourceReference


@dataclass(frozen=True, slots=True)
class ContextResult:
    """Bounded retrieval result and its measurable reduction."""

    query: str
    hits: tuple[SearchHit, ...]
    total_size_chars: int
    estimated_tokens: int
    candidates_considered: int
    search_time_ms: float
    limit: int
    token_budget: int
    source_estimated_tokens: int
    estimated_context_reduction: float
    index_hit: bool


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Aggregate outcome and timing for one indexing session."""

    session_id: str
    documents_seen: int
    documents_indexed: int
    duplicate_documents: int
    chunks_indexed: int
    duplicate_chunks: int
    source_size_bytes: int
    cleaned_size_bytes: int
    source_estimated_tokens: int
    cleaned_estimated_tokens: int
    cleaning_time_ms: float
    indexing_time_ms: float
    failed_pages: int = 0
    failure_categories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexSessionSummary:
    """Bounded counters for the most recently started indexing session."""

    session_id: str
    started_at: datetime
    finished_at: datetime | None
    documents_seen: int
    documents_indexed: int
    duplicate_documents: int
    chunks_indexed: int
    duplicate_chunks: int
    source_size_bytes: int
    cleaned_size_bytes: int
    source_estimated_tokens: int
    cleaned_estimated_tokens: int
    cleaning_time_ms: float
    indexing_time_ms: float


@dataclass(frozen=True, slots=True)
class IndexInfo:
    """Bounded readiness and size summary for one local context index."""

    schema_version: int
    document_count: int
    chunk_count: int
    last_indexed_at: datetime | None
    last_session_summary: IndexSessionSummary | None
    database_ready: bool
    fts5_available: bool


@runtime_checkable
class TokenEstimator(Protocol):
    """Estimate the number of model tokens in text."""

    def count(self, text: str) -> int:
        """Return a deterministic non-negative token estimate."""


@dataclass(frozen=True, slots=True)
class HeuristicTokenEstimator:
    """Estimate roughly one token per configurable number of characters."""

    chars_per_token: int = 4

    def __post_init__(self) -> None:
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be greater than zero")

    def count(self, text: str) -> int:
        """Return the ceiling of normalized character count per token."""
        if not text:
            return 0
        return (len(text) + self.chars_per_token - 1) // self.chars_per_token
