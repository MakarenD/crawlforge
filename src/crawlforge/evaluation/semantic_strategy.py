"""Semantic retrieval adapter for the strategy-neutral evaluator."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from crawlforge.context_engine import ContextEngine
from crawlforge.evaluation.models import EvaluationDataset, RetrievedItem
from crawlforge.semantic_models import (
    EmbeddingProvider,
    JSONValue,
    SemanticIndexInfo,
    SemanticIndexingResult,
)


class SemanticContextEngineStrategy:
    """Map production exact-cosine hits into evaluator retrieval items."""

    name = "semantic-exact-cosine"

    def __init__(
        self,
        engine: ContextEngine,
        provider: EmbeddingProvider,
        dataset: EvaluationDataset,
        *,
        indexing_result: SemanticIndexingResult | None = None,
        index_info: SemanticIndexInfo | None = None,
    ) -> None:
        self._engine = engine
        self._provider = provider
        self._indexing_result = indexing_result
        self._index_info = index_info
        self._document_ids = {
            document.url: document.document_id for document in dataset.documents
        }
        self._section_ids = {
            (document.document_id, section.heading_path): section.section_id
            for document in dataset.documents
            for section in document.sections
        }
        self._readiness_ms: list[float] = []
        self._query_encoding_ms: list[float] = []
        self._snapshot_fetch_ms: list[float] = []
        self._vector_decode_ms: list[float] = []
        self._scan_ms: list[float] = []
        self._provenance_materialization_ms: list[float] = []
        self._total_ms: list[float] = []
        self._loaded_vector_bytes: list[int] = []
        self._loaded_memory_bytes: list[int] = []

    @property
    def warnings(self) -> Sequence[str]:
        return (
            "Semantic cosine similarity is not calibrated confidence.",
            (
                "No negative-query score threshold is applied; embeddings alone "
                "do not provide calibrated abstention."
            ),
            (
                "Exact semantic search is O(number_of_chunks × "
                "embedding_dimension) and is intended for small local indexes."
            ),
        )

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[RetrievedItem]:
        """Use production semantic ranking without evaluator-side resorting."""
        result = await self._engine.semantic_search_with_metrics(
            query,
            provider=self._provider,
            limit=limit,
        )
        self._readiness_ms.append(result.readiness_check_time_ms)
        self._query_encoding_ms.append(result.query_encoding_time_ms)
        self._snapshot_fetch_ms.append(result.sqlite_snapshot_fetch_time_ms)
        self._vector_decode_ms.append(result.vector_decode_time_ms)
        self._scan_ms.append(result.exact_scan_time_ms)
        self._provenance_materialization_ms.append(
            result.provenance_materialization_time_ms
        )
        self._total_ms.append(result.total_retrieval_time_ms)
        self._loaded_vector_bytes.append(result.loaded_vector_bytes)
        self._loaded_memory_bytes.append(result.loaded_vector_memory_estimate_bytes)

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
                    score=hit.cosine_similarity,
                    estimated_tokens=hit.chunk.estimated_tokens,
                    source_estimated_tokens=hit.source.source_estimated_tokens,
                    content_hash=hit.chunk.content_hash,
                )
            )
        return tuple(items)

    def performance_metadata(self) -> dict[str, JSONValue]:
        """Return safe aggregate semantic indexing and retrieval measurements."""
        runtime = self._provider.runtime_info
        indexing = self._indexing_result
        info = self._index_info
        payload: dict[str, JSONValue] = {
            "score_type": "cosine_similarity",
            "score_order": "descending",
            "exact_scan_complexity": "O(number_of_chunks * embedding_dimension)",
            "readiness_check_mean_ms": _mean(self._readiness_ms),
            "query_encoding_mean_ms": _mean(self._query_encoding_ms),
            "sqlite_snapshot_fetch_mean_ms": _mean(self._snapshot_fetch_ms),
            "vector_decode_mean_ms": _mean(self._vector_decode_ms),
            "exact_vector_scan_mean_ms": _mean(self._scan_ms),
            "provenance_materialization_mean_ms": _mean(
                self._provenance_materialization_ms
            ),
            "sqlite_snapshot_scope": (
                "compatible vectors and complete chunk provenance"
            ),
            "total_semantic_retrieval_mean_ms": _mean(self._total_ms),
            "loaded_vector_bytes": max(self._loaded_vector_bytes, default=0),
            "loaded_vector_memory_estimate_bytes": max(
                self._loaded_memory_bytes,
                default=0,
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
                    "configured_max_sequence_length": (runtime.max_sequence_length),
                    "model_cache_size_bytes": runtime.model_cache_size_bytes,
                }
            )
        if indexing is not None:
            payload.update(
                {
                    "embedding_indexing_time_ms": indexing.elapsed_time_ms,
                    "document_encoding_time_ms": (indexing.document_encoding_time_ms),
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
                        "truncated_document_inputs": (statistics.truncated_input_count),
                        "truncated_document_fraction": (statistics.truncated_fraction),
                        "maximum_tokenized_length": (
                            statistics.maximum_tokenized_length
                        ),
                        "average_tokenized_length": (
                            statistics.average_tokenized_length
                        ),
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


def _mean(values: Sequence[float]) -> float:
    return fmean(values) if values else 0.0
