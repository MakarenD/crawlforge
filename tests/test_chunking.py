"""Tests for deterministic, heading-aware text chunking."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from crawlforge.chunking import ChunkingConfig, TextChunker
from crawlforge.context_models import (
    BlockKind,
    DocumentBlock,
    HeuristicTokenEstimator,
    SourceDocument,
)

FETCHED_AT = datetime(2026, 7, 28, 10, 0, tzinfo=UTC)


def block(
    kind: BlockKind,
    text: str,
    *,
    heading_path: tuple[str, ...] = (),
) -> DocumentBlock:
    """Create a compact test block."""
    return DocumentBlock(
        kind=kind,
        text=text,
        markdown=text,
        heading_path=heading_path,
    )


def document(
    *blocks: DocumentBlock,
    title: str = "Test document",
    document_id: str = "document-id",
) -> SourceDocument:
    """Create a source document with fixed provenance."""
    text = "\n\n".join(item.text for item in blocks)
    return SourceDocument(
        id=document_id,
        url="https://example.com/requested",
        canonical_url="https://example.com/canonical",
        title=title,
        text=text,
        markdown=text or None,
        status_code=200,
        content_type="text/html",
        fetched_at=FETCHED_AT,
        content_hash="source-content-hash",
        metadata={},
        source_size_bytes=len(text.encode("utf-8")),
        cleaned_size_bytes=len(text.encode("utf-8")),
        source_estimated_tokens=len(text) // 4,
        cleaned_estimated_tokens=len(text) // 4,
        blocks=blocks,
    )


def test_heuristic_token_estimator_is_deterministic_and_validated() -> None:
    """The lightweight estimator uses a documented character ceiling."""
    estimator = HeuristicTokenEstimator(chars_per_token=4)

    assert estimator.count("") == 0
    assert estimator.count("abcd") == 1
    assert estimator.count("abcde") == 2
    assert estimator.count("Привет") == 2
    with pytest.raises(ValueError):
        HeuristicTokenEstimator(chars_per_token=0)


@pytest.mark.parametrize(
    "config",
    [
        ChunkingConfig(target_chars=1, max_chars=1, overlap_chars=0),
        ChunkingConfig(target_chars=10, max_chars=20, overlap_chars=9),
    ],
)
def test_valid_character_configs_are_accepted(config: ChunkingConfig) -> None:
    """Chunk sizes are explicitly valid character counts."""
    assert TextChunker(config).chunk(document()) == ()


@pytest.mark.parametrize(
    "arguments",
    [
        {"target_chars": 0},
        {"max_chars": 0},
        {"target_chars": 20, "max_chars": 10},
        {"overlap_chars": -1},
        {"target_chars": 10, "overlap_chars": 10},
    ],
)
def test_invalid_character_configs_are_rejected(
    arguments: dict[str, int],
) -> None:
    """Invalid maxima and overlaps fail early."""
    with pytest.raises(ValueError):
        ChunkingConfig(**arguments)


def test_small_document_produces_one_non_heading_only_chunk() -> None:
    """A heading is attached to useful body content rather than emitted alone."""
    source = document(
        block("heading", "Introduction", heading_path=("Introduction",)),
        block(
            "paragraph",
            "A short useful paragraph.",
            heading_path=("Introduction",),
        ),
    )

    chunks = TextChunker(
        ChunkingConfig(target_chars=100, max_chars=120, overlap_chars=0)
    ).chunk(source)

    assert len(chunks) == 1
    assert chunks[0].text == "Introduction\n\nA short useful paragraph."
    assert chunks[0].heading_path == ("Introduction",)
    assert chunks[0].ordinal == 0
    assert chunks[0].size_chars == len(chunks[0].text)
    assert chunks[0].estimated_tokens > 0


def test_large_document_respects_heading_sections() -> None:
    """Chunks never combine body blocks from different heading paths."""
    source = document(
        block("heading", "Alpha", heading_path=("Alpha",)),
        block("paragraph", "alpha one " * 5, heading_path=("Alpha",)),
        block("paragraph", "alpha two " * 5, heading_path=("Alpha",)),
        block("heading", "Beta", heading_path=("Beta",)),
        block("paragraph", "beta only " * 5, heading_path=("Beta",)),
    )

    chunks = TextChunker(
        ChunkingConfig(target_chars=70, max_chars=100, overlap_chars=0)
    ).chunk(source)

    assert len(chunks) == 3
    assert [item.heading_path for item in chunks] == [
        ("Alpha",),
        ("Alpha",),
        ("Beta",),
    ]
    assert "beta" not in chunks[0].text
    assert "beta" not in chunks[1].text
    assert "alpha" not in chunks[2].text


def test_small_overlap_is_added_only_within_the_same_section() -> None:
    """A bounded suffix repeats in the next same-section chunk, not a new section."""
    source = document(
        block("heading", "Section", heading_path=("Section",)),
        block(
            "paragraph",
            "alpha beta gamma delta epsilon",
            heading_path=("Section",),
        ),
        block(
            "paragraph",
            "zeta eta theta iota kappa",
            heading_path=("Section",),
        ),
        block("heading", "Other", heading_path=("Other",)),
        block(
            "paragraph",
            "brand new material",
            heading_path=("Other",),
        ),
    )

    chunks = TextChunker(
        ChunkingConfig(target_chars=45, max_chars=90, overlap_chars=16)
    ).chunk(source)

    assert chunks[1].text.startswith("Section\n\ndelta epsilon\n\nzeta eta theta")
    assert chunks[2].text == "Other\n\nbrand new material"
    assert "iota kappa" not in chunks[2].text


def test_oversized_prose_uses_sentences_words_and_hard_character_splits() -> None:
    """Long prose and an unbroken word remain deterministic under the hard max."""
    prose = (
        "First sentence has several words. "
        "Second sentence also has several words. " + ("unbreakable" * 12)
    )
    source = document(block("paragraph", prose))
    chunker = TextChunker(
        ChunkingConfig(target_chars=45, max_chars=55, overlap_chars=0)
    )

    first = chunker.chunk(source)
    second = chunker.chunk(source)

    assert first == second
    assert len(first) > 3
    assert all(0 < item.size_chars <= 55 for item in first)
    assert "".join(item.text.replace(" ", "") for item in first) == prose.replace(
        " ",
        "",
    )


def test_paragraphs_at_or_below_maximum_are_not_split() -> None:
    """Ordinary semantic blocks remain whole even above the soft target."""
    paragraph = "p" * 80
    source = document(block("paragraph", paragraph))

    chunks = TextChunker(
        ChunkingConfig(target_chars=40, max_chars=80, overlap_chars=0)
    ).chunk(source)

    assert [item.text for item in chunks] == [paragraph]


def test_code_preserves_indentation_and_splits_by_lines_before_characters() -> None:
    """Code keeps first/inner indentation and hard-splits only an oversized line."""
    code = "    first_line()\n" + ("x" * 75) + "\n    final_line()"
    source = document(block("code", code))

    chunks = TextChunker(
        ChunkingConfig(target_chars=25, max_chars=30, overlap_chars=0)
    ).chunk(source)

    assert chunks[0].text == "    first_line()"
    assert chunks[-1].text == "    final_line()"
    assert "".join(item.text for item in chunks[1:-1]) == "x" * 75
    assert all(item.size_chars <= 30 for item in chunks)


def test_no_empty_or_heading_only_chunks_are_emitted() -> None:
    """Documents without useful body blocks produce no chunks."""
    source = document(
        block("heading", "Lonely heading", heading_path=("Lonely heading",)),
        block("paragraph", "   ", heading_path=("Lonely heading",)),
    )

    assert TextChunker().chunk(source) == ()


def test_chunk_ids_and_contextual_hashes_are_stable() -> None:
    """IDs repeat exactly while title and heading context affect global hashes."""
    text = "z" * 40
    config = ChunkingConfig(target_chars=40, max_chars=40, overlap_chars=0)
    chunker = TextChunker(config)
    alpha = document(
        block("paragraph", text, heading_path=("Alpha",)),
        title="Title",
    )
    beta = document(
        block("paragraph", text, heading_path=("Beta",)),
        title="Title",
    )
    renamed = document(
        block("paragraph", text, heading_path=("Alpha",)),
        title="Renamed",
    )

    first = chunker.chunk(alpha)
    repeated = chunker.chunk(alpha)
    other_heading = chunker.chunk(beta)
    other_title = chunker.chunk(renamed)

    assert first == repeated
    assert first[0].id == repeated[0].id
    assert first[0].text == other_heading[0].text == other_title[0].text
    assert first[0].content_hash != other_heading[0].content_hash
    assert first[0].content_hash != other_title[0].content_hash
