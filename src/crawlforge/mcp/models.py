"""Typed structured-output models exposed by CrawlForge MCP tools."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ContentTrust = Literal["untrusted_web_content"]


class MCPOutputModel(BaseModel):
    """Strict immutable base for deterministic MCP structured outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class IndexSiteOutput(MCPOutputModel):
    """Bounded aggregate result of one site indexing operation."""

    requested_url: str
    indexed_documents: int = Field(ge=0)
    created_chunks: int = Field(ge=0)
    failed_pages: int = Field(ge=0)
    deduplicated_documents: int = Field(ge=0)
    deduplicated_chunks: int = Field(ge=0)
    raw_bytes: int = Field(ge=0)
    clean_bytes: int = Field(ge=0)
    estimated_source_tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    database: str
    warnings: tuple[str, ...]


class RetrievedChunk(MCPOutputModel):
    """One complete lexical result with provenance and trust metadata."""

    rank: int = Field(ge=1)
    bm25_score: float
    title: str
    heading_path: tuple[str, ...]
    url: str
    canonical_url: str
    chunk_text: str
    estimated_tokens: int = Field(ge=0)
    content_trust: ContentTrust = "untrusted_web_content"


class SearchIndexOutput(MCPOutputModel):
    """Compact lexical BM25 results from the existing local index."""

    query: str
    results: tuple[RetrievedChunk, ...]
    returned_results: int = Field(ge=0)
    database: str
    result_limited: bool
    warnings: tuple[str, ...]


class BuildContextOutput(MCPOutputModel):
    """Complete ranked chunks selected within an approximate token budget."""

    query: str
    chunks: tuple[RetrievedChunk, ...]
    returned_chunks: int = Field(ge=0)
    total_size_chars: int = Field(ge=0)
    estimated_tokens: int = Field(ge=0)
    source_estimated_tokens: int = Field(ge=0)
    candidates_considered: int = Field(ge=0)
    token_budget: int = Field(ge=1)
    estimated_context_reduction: float = Field(ge=0, le=1)
    token_estimate: Literal["model_agnostic_heuristic"] = "model_agnostic_heuristic"
    context_reduction_interpretation: Literal[
        "approximate_ratio_not_model_specific_savings"
    ] = "approximate_ratio_not_model_specific_savings"
    database: str
    result_limited: bool
    warnings: tuple[str, ...]


class IndexSessionSummaryOutput(MCPOutputModel):
    """Bounded counters for the most recently started indexing session."""

    session_id: str
    started_at: datetime
    finished_at: datetime | None
    documents_seen: int = Field(ge=0)
    documents_indexed: int = Field(ge=0)
    duplicate_documents: int = Field(ge=0)
    chunks_indexed: int = Field(ge=0)
    duplicate_chunks: int = Field(ge=0)
    source_size_bytes: int = Field(ge=0)
    cleaned_size_bytes: int = Field(ge=0)
    source_estimated_tokens: int = Field(ge=0)
    cleaned_estimated_tokens: int = Field(ge=0)
    cleaning_time_ms: float = Field(ge=0)
    indexing_time_ms: float = Field(ge=0)


class IndexInfoOutput(MCPOutputModel):
    """Bounded readiness and size summary for the configured local index."""

    schema_version: int | None = Field(default=None, ge=1)
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    last_indexed_at: datetime | None
    last_session_summary: IndexSessionSummaryOutput | None
    database_ready: bool
    fts5_available: bool
