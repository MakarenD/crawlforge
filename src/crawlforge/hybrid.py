"""Production Reciprocal Rank Fusion over lexical and semantic APIs."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from crawlforge.context_engine import ContextEngine
from crawlforge.context_selection import select_bounded_context
from crawlforge.hybrid_models import (
    FusedCandidate,
    HybridContextResult,
    HybridIndexReadiness,
    HybridRetrievalError,
    HybridSearchConfig,
    HybridSearchHit,
    HybridSearchMetrics,
    HybridSearchResult,
    RankContribution,
    RankedCandidate,
    RankedCandidateList,
    RankFusionStrategy,
)
from crawlforge.semantic_models import EmbeddingProvider, embedding_model_info


@dataclass(slots=True)
class _FusionAccumulator:
    representative: RankedCandidate
    contributions: list[RankContribution]


class ReciprocalRankFusion:
    """Fuse finite weighted ranks without comparing component scores."""

    name = "reciprocal-rank-fusion"

    def __init__(self, *, rrf_k: int = 60) -> None:
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool):
            raise ValueError("rrf_k must be an integer")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self.rrf_k = rrf_k

    def fuse(
        self,
        *,
        rankings: Sequence[RankedCandidateList],
        limit: int,
    ) -> tuple[FusedCandidate, ...]:
        """Return deterministic RRF order and preserve raw scores as evidence."""
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        copied = tuple(rankings)
        if not copied:
            return ()
        strategies = tuple(ranking.strategy for ranking in copied)
        if len(set(strategies)) != len(strategies):
            raise ValueError("fusion rankings must use unique strategy names")
        if not any(ranking.weight > 0 for ranking in copied):
            raise ValueError("at least one fusion weight must be greater than zero")

        accumulators: dict[str, _FusionAccumulator] = {}
        ordered_rankings = sorted(
            copied,
            key=lambda ranking: (_strategy_order(ranking.strategy), ranking.strategy),
        )
        for ranking in ordered_rankings:
            self._validate_ranking(ranking)
            for candidate in ranking.candidates:
                contribution_value = ranking.weight / (self.rrf_k + candidate.rank)
                if not math.isfinite(contribution_value):
                    raise ValueError("RRF contribution must be finite")
                contribution = RankContribution(
                    strategy=ranking.strategy,
                    rank=candidate.rank,
                    weight=ranking.weight,
                    raw_score=candidate.raw_score,
                    score_type=candidate.score_type,
                    contribution=contribution_value,
                )
                accumulator = accumulators.get(candidate.identity)
                if accumulator is None:
                    accumulators[candidate.identity] = _FusionAccumulator(
                        representative=candidate,
                        contributions=[contribution],
                    )
                    continue
                _validate_same_candidate(accumulator.representative, candidate)
                accumulator.contributions.append(contribution)
                if _strategy_order(ranking.strategy) < _representative_order(
                    accumulator.contributions[:-1]
                ):
                    accumulator.representative = candidate

        scored: list[
            tuple[
                float,
                int,
                int,
                float,
                float,
                str,
                RankedCandidate,
                tuple[RankContribution, ...],
            ]
        ] = []
        for identity, accumulator in accumulators.items():
            contributions = tuple(
                sorted(
                    accumulator.contributions,
                    key=lambda item: (_strategy_order(item.strategy), item.strategy),
                )
            )
            rrf_score = math.fsum(item.contribution for item in contributions)
            if not math.isfinite(rrf_score):
                raise ValueError("RRF score must be finite")
            by_strategy = {item.strategy: item for item in contributions}
            bm25_rank = by_strategy["bm25"].rank if "bm25" in by_strategy else None
            semantic_rank = (
                by_strategy["semantic"].rank if "semantic" in by_strategy else None
            )
            ranks = tuple(item.rank for item in contributions)
            scored.append(
                (
                    -rrf_score,
                    -len(contributions),
                    min(ranks),
                    bm25_rank if bm25_rank is not None else math.inf,
                    semantic_rank if semantic_rank is not None else math.inf,
                    identity,
                    accumulator.representative,
                    contributions,
                )
            )

        scored.sort(key=lambda item: item[:6])
        fused: list[FusedCandidate] = []
        for rank, item in enumerate(scored[:limit], start=1):
            negative_score, _, _, _, _, identity, representative, contributions = item
            by_strategy = {entry.strategy: entry for entry in contributions}
            bm25 = by_strategy.get("bm25")
            semantic = by_strategy.get("semantic")
            fused.append(
                FusedCandidate(
                    identity=identity,
                    rank=rank,
                    chunk=representative.chunk,
                    source=representative.source,
                    rrf_score=-negative_score,
                    contributions=contributions,
                    bm25_rank=bm25.rank if bm25 is not None else None,
                    semantic_rank=(semantic.rank if semantic is not None else None),
                    bm25_score=bm25.raw_score if bm25 is not None else None,
                    cosine_similarity=(
                        semantic.raw_score if semantic is not None else None
                    ),
                )
            )
        return tuple(fused)

    @staticmethod
    def _validate_ranking(ranking: RankedCandidateList) -> None:
        ranks = tuple(candidate.rank for candidate in ranking.candidates)
        if ranks != tuple(range(1, len(ranking.candidates) + 1)):
            raise ValueError(
                f"{ranking.strategy} ranking must use contiguous ranks from one"
            )
        identities = tuple(candidate.identity for candidate in ranking.candidates)
        if len(set(identities)) != len(identities):
            raise ValueError(
                f"{ranking.strategy} ranking contains a duplicate candidate"
            )


class HybridRetriever:
    """Strict application service composing production BM25 and semantic search."""

    def __init__(
        self,
        *,
        context_engine: ContextEngine,
        embedding_provider: EmbeddingProvider,
        config: HybridSearchConfig | None = None,
        fusion: RankFusionStrategy | None = None,
    ) -> None:
        self._context_engine = context_engine
        self._embedding_provider = embedding_provider
        self.config = config or HybridSearchConfig()
        self._fusion = fusion or ReciprocalRankFusion(rrf_k=self.config.rrf_k)

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[HybridSearchHit]:
        """Return fused results while preserving both component ranks."""
        return list((await self.search_with_metrics(query, limit=limit)).hits)

    async def search_with_metrics(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> HybridSearchResult:
        """Run both required strategies sequentially and measure their boundaries."""
        self._validate_limit(limit)
        total_started = perf_counter()

        bm25_started = perf_counter()
        try:
            bm25_hits = await self._context_engine.search(
                query,
                limit=self.config.bm25_candidate_limit,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise HybridRetrievalError("bm25", error) from error
        bm25_time_ms = (perf_counter() - bm25_started) * 1000

        try:
            semantic_result = await self._context_engine.semantic_search_with_metrics(
                query,
                provider=self._embedding_provider,
                limit=self.config.semantic_candidate_limit,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise HybridRetrievalError("semantic", error) from error

        bm25_ranking = RankedCandidateList(
            strategy="bm25",
            weight=self.config.bm25_weight,
            candidates=tuple(
                RankedCandidate(
                    identity=hit.chunk.content_hash,
                    rank=hit.rank,
                    chunk=hit.chunk,
                    source=hit.source,
                    raw_score=hit.bm25_score,
                    score_type="bm25",
                )
                for hit in bm25_hits
            ),
        )
        semantic_ranking = RankedCandidateList(
            strategy="semantic",
            weight=self.config.semantic_weight,
            candidates=tuple(
                RankedCandidate(
                    identity=hit.chunk.content_hash,
                    rank=hit.rank,
                    chunk=hit.chunk,
                    source=hit.source,
                    raw_score=hit.cosine_similarity,
                    score_type=hit.score_type,
                )
                for hit in semantic_result.hits
            ),
        )
        fusion_started = perf_counter()
        try:
            fused = tuple(
                self._fusion.fuse(
                    rankings=(bm25_ranking, semantic_ranking),
                    limit=limit,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise HybridRetrievalError("fusion", error) from error
        fusion_time_ms = (perf_counter() - fusion_started) * 1000

        model = embedding_model_info(self._embedding_provider)
        hits = tuple(
            _hybrid_hit(
                item,
                config=self.config,
                model_id=model.model_id,
                model_revision=model.model_revision,
                model_fingerprint=model.fingerprint,
                fusion_strategy=self._fusion.name,
            )
            for item in fused
        )
        bm25_identities = {candidate.identity for candidate in bm25_ranking.candidates}
        semantic_identities = {
            candidate.identity for candidate in semantic_ranking.candidates
        }
        hydration_time_ms = (
            semantic_result.sqlite_snapshot_fetch_time_ms
            + semantic_result.vector_decode_time_ms
            + semantic_result.provenance_materialization_time_ms
        )
        total_time_ms = (perf_counter() - total_started) * 1000
        metrics = HybridSearchMetrics(
            bm25_retrieval_time_ms=bm25_time_ms,
            semantic_readiness_check_time_ms=(semantic_result.readiness_check_time_ms),
            semantic_query_encoding_time_ms=semantic_result.query_encoding_time_ms,
            semantic_snapshot_fetch_time_ms=(
                semantic_result.sqlite_snapshot_fetch_time_ms
            ),
            semantic_vector_decode_time_ms=semantic_result.vector_decode_time_ms,
            semantic_exact_scan_time_ms=semantic_result.exact_scan_time_ms,
            semantic_provenance_materialization_time_ms=(
                semantic_result.provenance_materialization_time_ms
            ),
            semantic_retrieval_time_ms=semantic_result.total_retrieval_time_ms,
            fusion_time_ms=fusion_time_ms,
            provenance_context_hydration_time_ms=hydration_time_ms,
            total_retrieval_time_ms=total_time_ms,
            sum_component_durations_ms=(
                bm25_time_ms + semantic_result.total_retrieval_time_ms + fusion_time_ms
            ),
            bm25_candidate_count=len(bm25_ranking.candidates),
            semantic_candidate_count=len(semantic_ranking.candidates),
            overlapping_candidate_count=len(bm25_identities & semantic_identities),
            unique_candidate_count=len(bm25_identities | semantic_identities),
        )
        return HybridSearchResult(
            query=query,
            hits=hits,
            metrics=metrics,
            limit=limit,
            model_id=model.model_id,
            model_revision=model.model_revision,
            model_fingerprint=model.fingerprint,
            fusion_configuration=self.config,
            fusion_strategy=self._fusion.name,
        )

    async def build_context(
        self,
        query: str,
        *,
        limit: int = 10,
        token_budget: int = 3000,
    ) -> HybridContextResult:
        """Reuse the shared complete-chunk, strict-budget selection."""
        if isinstance(token_budget, bool) or token_budget <= 0:
            raise ValueError("token_budget must be greater than zero")
        result = await self.search_with_metrics(query, limit=limit)
        selection_started = perf_counter()
        selection = select_bounded_context(result.hits, token_budget=token_budget)
        selection_time_ms = (perf_counter() - selection_started) * 1000
        return HybridContextResult(
            query=result.query,
            hits=selection.hits,
            total_size_chars=selection.total_size_chars,
            estimated_tokens=selection.estimated_tokens,
            candidates_considered=selection.candidate_count,
            search_time_ms=result.metrics.total_retrieval_time_ms,
            context_selection_time_ms=selection_time_ms,
            limit=limit,
            token_budget=token_budget,
            source_estimated_tokens=selection.source_estimated_tokens,
            estimated_context_reduction=selection.estimated_context_reduction,
            index_hit=bool(result.hits),
            model_id=result.model_id,
            model_revision=result.model_revision,
            model_fingerprint=result.model_fingerprint,
            fusion_configuration=result.fusion_configuration,
            metrics=result.metrics,
            fusion_strategy=result.fusion_strategy,
        )

    async def get_index_readiness(self) -> HybridIndexReadiness:
        """Return strict readiness through the two public index-info APIs."""
        try:
            lexical = await self._context_engine.get_index_info()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise HybridRetrievalError("bm25", error) from error
        try:
            semantic = await self._context_engine.get_semantic_index_info(
                self._embedding_provider
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise HybridRetrievalError("semantic", error) from error
        lexical_ready = lexical.database_ready and lexical.fts5_available
        semantic_ready = semantic.ready
        return HybridIndexReadiness(
            lexical=lexical,
            semantic=semantic,
            lexical_ready=lexical_ready,
            semantic_ready=semantic_ready,
            ready=lexical_ready and semantic_ready,
            model_id=semantic.model.model_id,
            model_revision=semantic.model.model_revision,
            model_fingerprint=semantic.model.fingerprint,
        )

    def _validate_limit(self, limit: int) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if self.config.bm25_candidate_limit < limit:
            raise ValueError("bm25 candidate limit must be at least the final limit")
        if self.config.semantic_candidate_limit < limit:
            raise ValueError(
                "semantic candidate limit must be at least the final limit"
            )


def _hybrid_hit(
    item: FusedCandidate,
    *,
    config: HybridSearchConfig,
    model_id: str,
    model_revision: str | None,
    model_fingerprint: str,
    fusion_strategy: str,
) -> HybridSearchHit:
    contributions = {entry.strategy: entry for entry in item.contributions}
    return HybridSearchHit(
        identity=item.identity,
        chunk=item.chunk,
        rank=item.rank,
        rrf_score=item.rrf_score,
        source=item.source,
        bm25_rank=item.bm25_rank,
        semantic_rank=item.semantic_rank,
        bm25_contribution=(
            contributions["bm25"].contribution if "bm25" in contributions else 0.0
        ),
        semantic_contribution=(
            contributions["semantic"].contribution
            if "semantic" in contributions
            else 0.0
        ),
        bm25_score=item.bm25_score,
        cosine_similarity=item.cosine_similarity,
        contributions=item.contributions,
        model_id=model_id,
        model_revision=model_revision,
        model_fingerprint=model_fingerprint,
        fusion_configuration=config,
        fusion_strategy=fusion_strategy,
    )


def _strategy_order(strategy: str) -> int:
    if strategy == "bm25":
        return 0
    if strategy == "semantic":
        return 1
    return 2


def _representative_order(contributions: Sequence[RankContribution]) -> int:
    return min(
        (_strategy_order(contribution.strategy) for contribution in contributions),
        default=3,
    )


def _validate_same_candidate(
    left: RankedCandidate,
    right: RankedCandidate,
) -> None:
    if left.identity != right.identity:
        raise ValueError("cannot merge different candidate identities")
    if left.chunk != right.chunk or left.source != right.source:
        raise ValueError(
            "component rankings disagree on chunk metadata for one identity"
        )
