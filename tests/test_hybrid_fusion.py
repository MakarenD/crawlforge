from __future__ import annotations

from datetime import UTC, datetime

import pytest

from crawlforge.context_models import SourceReference, TextChunk
from crawlforge.hybrid import ReciprocalRankFusion
from crawlforge.hybrid_models import (
    HybridSearchConfig,
    RankedCandidate,
    RankedCandidateList,
)


def _candidate(identity: str, rank: int, *, score: float = 1.0) -> RankedCandidate:
    chunk = TextChunk(
        id=f"chunk-{identity}",
        document_id=f"document-{identity}",
        ordinal=0,
        source_url=f"https://example.test/{identity}",
        document_title=f"Document {identity}",
        heading_path=(f"Section {identity}",),
        text=f"Content for {identity}",
        size_chars=13,
        estimated_tokens=4,
        content_hash=identity,
    )
    source = SourceReference(
        document_id=chunk.document_id,
        url=chunk.source_url,
        canonical_url=chunk.source_url,
        title=chunk.document_title,
        status_code=200,
        content_type="text/html",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_size_bytes=13,
        source_estimated_tokens=4,
    )
    return RankedCandidate(
        identity=identity,
        rank=rank,
        chunk=chunk,
        source=source,
        raw_score=score,
        score_type="test-score",
    )


def _ranking(
    strategy: str,
    identities: tuple[str, ...],
    *,
    weight: float = 1.0,
    scores: tuple[float, ...] | None = None,
) -> RankedCandidateList:
    raw_scores = scores or tuple(float(rank) for rank in range(1, len(identities) + 1))
    return RankedCandidateList(
        strategy=strategy,
        weight=weight,
        candidates=tuple(
            _candidate(identity, rank, score=score)
            for rank, (identity, score) in enumerate(
                zip(identities, raw_scores, strict=True),
                start=1,
            )
        ),
    )


def test_rrf_uses_manually_verified_rank_contributions_and_order() -> None:
    fusion = ReciprocalRankFusion(rrf_k=60)
    bm25 = _ranking("bm25", ("A", "B", "C"), scores=(9.0, 4.0, 2.0))
    semantic = _ranking(
        "semantic",
        ("C", "A", "D"),
        scores=(0.99, 0.75, 0.5),
    )

    fused = fusion.fuse(rankings=(bm25, semantic), limit=4)

    assert [candidate.identity for candidate in fused] == ["A", "C", "B", "D"]
    assert [candidate.rank for candidate in fused] == [1, 2, 3, 4]
    assert fused[0].rrf_score == pytest.approx(
        0.01639344262295082 + 0.016129032258064516
    )
    assert fused[1].rrf_score == pytest.approx(
        0.015873015873015872 + 0.01639344262295082
    )
    assert fused[2].rrf_score == pytest.approx(0.016129032258064516)
    assert fused[3].rrf_score == pytest.approx(0.015873015873015872)
    assert fused[0].bm25_rank == 1
    assert fused[0].semantic_rank == 2
    assert fused[0].bm25_score == 9.0
    assert fused[0].cosine_similarity == 0.75
    assert [entry.contribution for entry in fused[0].contributions] == pytest.approx(
        [0.01639344262295082, 0.016129032258064516]
    )
    assert fused[2].semantic_rank is None
    assert fused[3].bm25_rank is None


def test_rrf_preserves_empty_rankings_limit_and_custom_weights() -> None:
    fusion = ReciprocalRankFusion(rrf_k=10)
    empty = _ranking("semantic", (), weight=2.0)
    bm25 = _ranking("bm25", ("A", "B"), weight=2.0)

    assert fusion.fuse(rankings=(), limit=5) == ()
    fused = fusion.fuse(rankings=(bm25, empty), limit=1)

    assert len(fused) == 1
    assert fused[0].identity == "A"
    assert fused[0].rrf_score == pytest.approx(0.18181818181818182)
    assert fused[0].contributions[0].weight == 2.0


def test_rrf_ties_are_stable_and_prefer_bm25_when_source_ranks_match() -> None:
    fusion = ReciprocalRankFusion()
    bm25 = _ranking("bm25", ("B",))
    semantic = _ranking("semantic", ("D",))

    expected = ["B", "D"]
    for rankings in ((bm25, semantic), (semantic, bm25)):
        assert [
            candidate.identity for candidate in fusion.fuse(rankings=rankings, limit=2)
        ] == expected


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"rrf_k": 0}, "rrf_k"),
        ({"rrf_k": True}, "rrf_k"),
        ({"bm25_weight": -1.0}, "bm25_weight"),
        ({"semantic_weight": float("nan")}, "semantic_weight"),
        ({"bm25_weight": float("inf")}, "bm25_weight"),
        ({"bm25_weight": 0.0, "semantic_weight": 0.0}, "at least one"),
        ({"bm25_candidate_limit": 0}, "bm25_candidate_limit"),
        ({"semantic_candidate_limit": -1}, "semantic_candidate_limit"),
    ],
)
def test_hybrid_config_rejects_invalid_values(
    updates: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "rrf_k": 60,
        "bm25_weight": 1.0,
        "semantic_weight": 1.0,
        "bm25_candidate_limit": 50,
        "semantic_candidate_limit": 50,
    }
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        HybridSearchConfig(**values)  # type: ignore[arg-type]


def test_rrf_rejects_duplicate_candidates_ranks_and_non_finite_scores() -> None:
    fusion = ReciprocalRankFusion()
    duplicate_candidate = RankedCandidateList(
        strategy="bm25",
        weight=1.0,
        candidates=(_candidate("A", 1), _candidate("A", 2)),
    )
    duplicate_rank = RankedCandidateList(
        strategy="semantic",
        weight=1.0,
        candidates=(_candidate("A", 1), _candidate("B", 1)),
    )

    with pytest.raises(ValueError, match="duplicate candidate"):
        fusion.fuse(rankings=(duplicate_candidate,), limit=2)
    with pytest.raises(ValueError, match="contiguous ranks"):
        fusion.fuse(rankings=(duplicate_rank,), limit=2)
    with pytest.raises(ValueError, match="raw score must be finite"):
        _candidate("A", 1, score=float("nan"))
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        fusion.fuse(rankings=(_ranking("bm25", ("A",)),), limit=0)
    with pytest.raises(ValueError, match="rrf_k"):
        ReciprocalRankFusion(rrf_k=-1)


def test_rrf_rejects_conflicting_metadata_for_one_identity() -> None:
    left = _candidate("A", 1)
    right = _candidate("A", 1)
    conflicting_chunk = TextChunk(
        id=right.chunk.id,
        document_id="different-document",
        ordinal=right.chunk.ordinal,
        source_url=right.chunk.source_url,
        document_title=right.chunk.document_title,
        heading_path=right.chunk.heading_path,
        text=right.chunk.text,
        size_chars=right.chunk.size_chars,
        estimated_tokens=right.chunk.estimated_tokens,
        content_hash=right.chunk.content_hash,
    )
    right = RankedCandidate(
        identity=right.identity,
        rank=right.rank,
        chunk=conflicting_chunk,
        source=right.source,
        raw_score=right.raw_score,
        score_type=right.score_type,
    )

    with pytest.raises(ValueError, match="disagree on chunk metadata"):
        ReciprocalRankFusion().fuse(
            rankings=(
                RankedCandidateList("bm25", 1.0, (left,)),
                RankedCandidateList("semantic", 1.0, (right,)),
            ),
            limit=1,
        )
