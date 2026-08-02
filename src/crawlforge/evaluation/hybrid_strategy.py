"""Hybrid retrieval adapter for the strategy-neutral evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import fmean

from crawlforge.context_engine import ContextEngine
from crawlforge.evaluation.models import EvaluationDataset, RetrievedItem
from crawlforge.hybrid import HybridRetriever
from crawlforge.hybrid_models import HybridSearchConfig, HybridSearchMetrics
from crawlforge.semantic_models import (
    EmbeddingProvider,
    JSONValue,
    SemanticIndexInfo,
    SemanticIndexingResult,
)


class HybridContextEngineStrategy:
    """Map production RRF hits into evaluator retrieval items."""

    name = "hybrid-rrf"
    retrieval_strategy = name

    def __init__(
        self,
        engine: ContextEngine,
        provider: EmbeddingProvider,
        dataset: EvaluationDataset,
        *,
        config: HybridSearchConfig | None = None,
        indexing_result: SemanticIndexingResult | None = None,
        index_info: SemanticIndexInfo | None = None,
    ) -> None:
        self._provider = provider
        self._config = config or HybridSearchConfig()
        self._indexing_result = indexing_result
        self._index_info = index_info
        self._retriever = HybridRetriever(
            context_engine=engine,
            embedding_provider=provider,
            config=self._config,
        )
        self._document_ids = {
            document.url: document.document_id for document in dataset.documents
        }
        self._section_ids = {
            (document.document_id, section.heading_path): section.section_id
            for document in dataset.documents
            for section in document.sections
        }
        self._metrics: list[HybridSearchMetrics] = []

    @property
    def warnings(self) -> Sequence[str]:
        """Return stable methodological limitations for generated reports."""
        return (
            "RRF scores are rank-fusion values, not calibrated confidence.",
            (
                "Hybrid retrieval requires both BM25 and semantic search to "
                "succeed; it does not silently fall back to one component."
            ),
            (
                "No negative-query score threshold is applied; hybrid retrieval "
                "does not provide calibrated abstention."
            ),
            (
                "The canonical hybrid baseline executes BM25 and semantic search "
                "sequentially."
            ),
        )

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[RetrievedItem]:
        """Use production rank fusion and preserve its final order unchanged."""
        result = await self._retriever.search_with_metrics(query, limit=limit)
        self._metrics.append(result.metrics)
        config_metadata = _config_metadata(result.fusion_configuration)

        items: list[RetrievedItem] = []
        for hit in result.hits:
            document_id = self._document_ids.get(
                hit.source.canonical_url,
                self._document_ids.get(
                    hit.source.url,
                    hit.source.document_id,
                ),
            )
            items.append(
                RetrievedItem(
                    rank=hit.rank,
                    document_id=document_id,
                    url=hit.source.url,
                    canonical_url=hit.source.canonical_url,
                    title=hit.source.title,
                    section_id=self._section_ids.get(
                        (document_id, hit.chunk.heading_path)
                    ),
                    heading_path=hit.chunk.heading_path,
                    text=hit.chunk.text,
                    score=hit.rrf_score,
                    estimated_tokens=hit.chunk.estimated_tokens,
                    source_estimated_tokens=hit.source.source_estimated_tokens,
                    content_hash=hit.chunk.content_hash,
                    strategy_metadata={
                        "retrieval_strategy": hit.retrieval_strategy,
                        "fusion_strategy": hit.fusion_strategy,
                        "score_type": hit.score_type,
                        "bm25_rank": hit.bm25_rank,
                        "semantic_rank": hit.semantic_rank,
                        "bm25_contribution": hit.bm25_contribution,
                        "semantic_contribution": hit.semantic_contribution,
                        "bm25_score": hit.bm25_score,
                        "cosine_similarity": hit.cosine_similarity,
                        **config_metadata,
                        "model_id": hit.model_id,
                        "model_revision": hit.model_revision,
                        "model_fingerprint": hit.model_fingerprint,
                    },
                )
            )
        return tuple(items)

    def performance_metadata(self) -> dict[str, JSONValue]:
        """Return JSON-safe aggregate indexing and hybrid retrieval evidence."""
        runtime = self._provider.runtime_info
        indexing = self._indexing_result
        info = self._index_info
        metrics = self._metrics
        payload: dict[str, JSONValue] = {
            "score_type": "rrf_score",
            "score_order": "descending",
            "ranking": "reciprocal_rank_fusion",
            "execution_mode": "sequential",
            "strict_component_success": True,
            **_config_metadata(self._config),
            "bm25_retrieval_mean_ms": _mean(
                item.bm25_retrieval_time_ms for item in metrics
            ),
            "semantic_readiness_check_mean_ms": _mean(
                item.semantic_readiness_check_time_ms for item in metrics
            ),
            "semantic_query_encoding_mean_ms": _mean(
                item.semantic_query_encoding_time_ms for item in metrics
            ),
            "semantic_snapshot_fetch_mean_ms": _mean(
                item.semantic_snapshot_fetch_time_ms for item in metrics
            ),
            "semantic_vector_decode_mean_ms": _mean(
                item.semantic_vector_decode_time_ms for item in metrics
            ),
            "semantic_exact_scan_mean_ms": _mean(
                item.semantic_exact_scan_time_ms for item in metrics
            ),
            "semantic_provenance_materialization_mean_ms": _mean(
                item.semantic_provenance_materialization_time_ms for item in metrics
            ),
            "semantic_retrieval_mean_ms": _mean(
                item.semantic_retrieval_time_ms for item in metrics
            ),
            "fusion_mean_ms": _mean(item.fusion_time_ms for item in metrics),
            "provenance_context_hydration_mean_ms": _mean(
                item.provenance_context_hydration_time_ms for item in metrics
            ),
            "sum_component_durations_mean_ms": _mean(
                item.sum_component_durations_ms for item in metrics
            ),
            "total_hybrid_retrieval_mean_ms": _mean(
                item.total_retrieval_time_ms for item in metrics
            ),
            "bm25_candidate_count_max": max(
                (item.bm25_candidate_count for item in metrics), default=0
            ),
            "semantic_candidate_count_max": max(
                (item.semantic_candidate_count for item in metrics), default=0
            ),
            "overlapping_candidate_count_max": max(
                (item.overlapping_candidate_count for item in metrics), default=0
            ),
            "unique_candidate_count_max": max(
                (item.unique_candidate_count for item in metrics), default=0
            ),
        }
        if runtime is not None:
            payload.update(
                {
                    "device": runtime.device,
                    "sentence_transformers_version": (
                        runtime.sentence_transformers_version
                    ),
                    "transformers_version": runtime.transformers_version,
                    "torch_version": runtime.torch_version,
                    "model_load_time_ms": runtime.model_load_time_ms,
                    "configured_max_sequence_length": runtime.max_sequence_length,
                    "model_cache_size_bytes": runtime.model_cache_size_bytes,
                }
            )
        if indexing is not None:
            payload.update(
                {
                    "embedding_indexing_time_ms": indexing.elapsed_time_ms,
                    "document_encoding_time_ms": indexing.document_encoding_time_ms,
                    "sqlite_vector_write_time_ms": indexing.sqlite_write_time_ms,
                    "embedding_batch_size": indexing.batch_size,
                    "embedded_chunks": indexing.embedded_chunks,
                    "embedding_cache_hits": indexing.cache_hits,
                    "invalidated_embeddings": indexing.invalidated_embeddings,
                    "failed_embedding_chunks": indexing.failed_chunks,
                    "embeddings_per_second": indexing.embeddings_per_second,
                }
            )
            statistics = indexing.input_statistics
            if statistics is not None:
                payload.update(
                    {
                        "document_input_count": statistics.input_count,
                        "truncated_document_inputs": statistics.truncated_input_count,
                        "truncated_document_fraction": statistics.truncated_fraction,
                        "maximum_tokenized_length": (
                            statistics.maximum_tokenized_length
                        ),
                        "average_tokenized_length": statistics.average_tokenized_length,
                    }
                )
        if info is not None:
            payload.update(
                {
                    "stored_vector_bytes": info.stored_vector_bytes,
                    "average_vector_bytes_per_chunk": (
                        info.stored_vector_bytes / info.embedded_chunks
                        if info.embedded_chunks
                        else 0.0
                    ),
                }
            )
        return payload


def _config_metadata(config: HybridSearchConfig) -> dict[str, JSONValue]:
    return {
        "rrf_k": config.rrf_k,
        "bm25_weight": config.bm25_weight,
        "semantic_weight": config.semantic_weight,
        "bm25_candidate_limit": config.bm25_candidate_limit,
        "semantic_candidate_limit": config.semantic_candidate_limit,
    }


def _mean(values: Iterable[float]) -> float:
    copied = tuple(values)
    return fmean(copied) if copied else 0.0
