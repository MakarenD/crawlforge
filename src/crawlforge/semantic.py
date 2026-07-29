"""Production application service for optional local semantic retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter

from crawlforge.async_utils import run_lifecycle_owned_thread
from crawlforge.context_engine import ContextEngine
from crawlforge.context_selection import select_bounded_context
from crawlforge.semantic_models import (
    EmbeddingInputStatistics,
    EmbeddingModelInfo,
    EmbeddingProvider,
    EmbeddingRuntimeInfo,
    EmbeddingVector,
    SemanticContextResult,
    SemanticDependencyError,
    SemanticIndexIncompatibleError,
    SemanticIndexInfo,
    SemanticIndexingResult,
    SemanticIndexNotReadyError,
    SemanticSearchHit,
    SemanticSearchResult,
    StoredChunkEmbedding,
    embedding_model_info,
    exact_cosine_similarity,
    format_semantic_document,
    format_semantic_query,
    validate_embedding_batch,
)


@dataclass(frozen=True, slots=True)
class _RankedVector:
    embedding: StoredChunkEmbedding
    score: float


class SemanticContextEngine:
    """Coordinate embedding inference, SQLite storage, and exact cosine search."""

    def __init__(
        self,
        engine: ContextEngine,
        provider: EmbeddingProvider,
    ) -> None:
        self._engine = engine
        self._provider = provider

    async def index(self, *, batch_size: int = 32) -> SemanticIndexingResult:
        """Incrementally embed existing chunks in bounded inference batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        started = perf_counter()
        model = embedding_model_info(self._provider)
        plan = await self._engine.index.prepare_semantic_index(model)
        session_id = await self._engine.index.start_embedding_session(model)
        embedded_chunks = 0
        failed_chunks = 0
        written_bytes = 0
        encoding_time_ms = 0.0
        sqlite_write_time_ms = 0.0
        warnings: list[str] = []
        input_statistics: list[EmbeddingInputStatistics] = []
        after_storage_id = 0

        try:
            while True:
                records = await self._engine.index.list_chunks_missing_embedding(
                    model.fingerprint,
                    after_storage_id=after_storage_id,
                    limit=batch_size,
                )
                if not records:
                    break
                after_storage_id = records[-1].storage_id
                inputs = tuple(
                    format_semantic_document(record.chunk) for record in records
                )
                try:
                    input_statistics.append(
                        await self._provider.analyze_document_inputs(inputs)
                    )
                    encoding_started = perf_counter()
                    vectors = validate_embedding_batch(
                        await self._provider.embed_documents(inputs),
                        expected_count=len(inputs),
                        dimension=model.dimension,
                        normalized=model.normalized,
                    )
                    encoding_time_ms += (perf_counter() - encoding_started) * 1000
                    runtime_model = embedding_model_info(self._provider)
                    if runtime_model.fingerprint != model.fingerprint:
                        raise RuntimeError(
                            "embedding provider fingerprint changed after model load"
                        )
                    model = runtime_model
                    write_started = perf_counter()
                    written_bytes += await self._engine.index.store_chunk_embeddings(
                        model,
                        tuple(zip(records, vectors, strict=True)),
                    )
                    sqlite_write_time_ms += (perf_counter() - write_started) * 1000
                    embedded_chunks += len(records)
                except asyncio.CancelledError:
                    raise
                except SemanticDependencyError:
                    raise
                except Exception as error:
                    failed_chunks += len(records)
                    warnings.append(
                        "Embedding batch ending at chunk "
                        f"{after_storage_id} failed with {type(error).__name__}."
                    )
        except BaseException:
            partial = _indexing_result(
                session_id=session_id,
                model=model,
                considered_chunks=plan.considered_chunks,
                embedded_chunks=embedded_chunks,
                cache_hits=plan.cache_hits,
                invalidated_embeddings=plan.invalidated_embeddings,
                failed_chunks=failed_chunks,
                elapsed_time_ms=(perf_counter() - started) * 1000,
                document_encoding_time_ms=encoding_time_ms,
                sqlite_write_time_ms=sqlite_write_time_ms,
                stored_vector_bytes=written_bytes,
                batch_size=batch_size,
                input_statistics=_merge_input_statistics(input_statistics),
                runtime=self._provider.runtime_info,
                warnings=tuple(warnings),
            )
            await self._finish_session_cancellation_safe(partial)
            raise

        result = _indexing_result(
            session_id=session_id,
            model=model,
            considered_chunks=plan.considered_chunks,
            embedded_chunks=embedded_chunks,
            cache_hits=plan.cache_hits,
            invalidated_embeddings=plan.invalidated_embeddings,
            failed_chunks=failed_chunks,
            elapsed_time_ms=(perf_counter() - started) * 1000,
            document_encoding_time_ms=encoding_time_ms,
            sqlite_write_time_ms=sqlite_write_time_ms,
            stored_vector_bytes=written_bytes,
            batch_size=batch_size,
            input_statistics=_merge_input_statistics(input_statistics),
            runtime=self._provider.runtime_info,
            warnings=tuple(warnings),
        )
        await self._finish_session_cancellation_safe(result)
        info = await self._engine.index.get_semantic_index_info(model)
        return replace(
            result,
            total_stored_vector_bytes=info.stored_vector_bytes,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[SemanticSearchHit]:
        """Return exact cosine-ranked semantic hits."""
        result = await self.search_with_metrics(query, limit=limit)
        return list(result.hits)

    async def search_with_metrics(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> SemanticSearchResult:
        """Return semantic hits with separated query, scan, and hydration timing."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        total_started = perf_counter()
        formatted_query = format_semantic_query(query)
        model = embedding_model_info(self._provider)
        readiness_started = perf_counter()
        info = await self._engine.index.get_semantic_index_info(model)
        readiness_check_time_ms = (perf_counter() - readiness_started) * 1000
        if info.total_chunks == 0:
            return SemanticSearchResult(
                query=formatted_query,
                hits=(),
                readiness_check_time_ms=readiness_check_time_ms,
                query_encoding_time_ms=0.0,
                sqlite_snapshot_fetch_time_ms=0.0,
                vector_decode_time_ms=0.0,
                exact_scan_time_ms=0.0,
                provenance_materialization_time_ms=0.0,
                total_retrieval_time_ms=(perf_counter() - total_started) * 1000,
                loaded_vector_bytes=0,
                loaded_vector_memory_estimate_bytes=0,
            )
        if not info.compatible_model_registered:
            if info.other_model_count:
                raise SemanticIndexIncompatibleError(
                    "Semantic embeddings exist, but not for the requested model "
                    "fingerprint. Build a separate compatible embedding index first."
                )
            raise SemanticIndexNotReadyError(
                "Semantic index is not ready. Run 'crawlforge embed' first."
            )
        if not info.ready:
            raise SemanticIndexNotReadyError(
                "Semantic index is incomplete for this model. "
                "Run 'crawlforge embed' first."
            )

        query_started = perf_counter()
        query_vectors = validate_embedding_batch(
            await self._provider.embed_queries((formatted_query,)),
            expected_count=1,
            dimension=model.dimension,
            normalized=model.normalized,
        )
        query_encoding_time_ms = (perf_counter() - query_started) * 1000

        snapshot = await self._engine.index.load_semantic_snapshot(model)
        scan_started = perf_counter()
        ranked = await run_lifecycle_owned_thread(
            _rank_vectors,
            query_vectors[0],
            snapshot.embeddings,
            model,
            limit,
        )
        exact_scan_time_ms = (perf_counter() - scan_started) * 1000

        hits = tuple(
            SemanticSearchHit(
                chunk=ranked_item.embedding.record.chunk,
                rank=rank,
                cosine_similarity=ranked_item.score,
                source=ranked_item.embedding.record.source,
                model_id=model.model_id,
                model_revision=model.model_revision,
                model_fingerprint=model.fingerprint,
            )
            for rank, ranked_item in enumerate(
                ranked,
                start=1,
            )
        )
        return SemanticSearchResult(
            query=formatted_query,
            hits=hits,
            readiness_check_time_ms=readiness_check_time_ms,
            query_encoding_time_ms=query_encoding_time_ms,
            sqlite_snapshot_fetch_time_ms=(snapshot.sqlite_snapshot_fetch_time_ms),
            vector_decode_time_ms=snapshot.vector_decode_time_ms,
            exact_scan_time_ms=exact_scan_time_ms,
            provenance_materialization_time_ms=(
                snapshot.provenance_materialization_time_ms
            ),
            total_retrieval_time_ms=(perf_counter() - total_started) * 1000,
            loaded_vector_bytes=snapshot.stored_vector_bytes,
            loaded_vector_memory_estimate_bytes=snapshot.stored_vector_bytes,
        )

    async def build_context(
        self,
        query: str,
        *,
        limit: int = 10,
        token_budget: int = 3000,
    ) -> SemanticContextResult:
        """Select complete semantic hits through the shared bounded selector."""
        result = await self.search_with_metrics(query, limit=limit)
        selection = select_bounded_context(
            result.hits,
            token_budget=token_budget,
        )
        model = embedding_model_info(self._provider)
        return SemanticContextResult(
            query=result.query,
            hits=selection.hits,
            total_size_chars=selection.total_size_chars,
            estimated_tokens=selection.estimated_tokens,
            candidates_considered=selection.candidate_count,
            search_time_ms=result.total_retrieval_time_ms,
            limit=limit,
            token_budget=token_budget,
            source_estimated_tokens=selection.source_estimated_tokens,
            estimated_context_reduction=selection.estimated_context_reduction,
            index_hit=bool(result.hits),
            model_id=model.model_id,
            model_revision=model.model_revision,
            model_fingerprint=model.fingerprint,
        )

    async def get_index_info(self) -> SemanticIndexInfo:
        """Return readiness for the configured model fingerprint."""
        return await self._engine.index.get_semantic_index_info(
            embedding_model_info(self._provider)
        )

    async def _finish_session_cancellation_safe(
        self,
        result: SemanticIndexingResult,
    ) -> None:
        finish_task = asyncio.create_task(
            self._engine.index.finish_embedding_session(result)
        )
        try:
            await asyncio.shield(finish_task)
        except asyncio.CancelledError as cancelled:
            try:
                await finish_task
            except Exception as finish_error:
                raise cancelled from finish_error
            raise


def _rank_vectors(
    query: EmbeddingVector,
    stored: Sequence[StoredChunkEmbedding],
    model: EmbeddingModelInfo,
    limit: int,
) -> tuple[_RankedVector, ...]:
    ranked = [
        _RankedVector(
            embedding=item,
            score=exact_cosine_similarity(
                query,
                item.vector,
                normalized=model.normalized,
            ),
        )
        for item in stored
    ]
    ranked.sort(
        key=lambda item: (
            -item.score,
            item.embedding.record.chunk.content_hash,
            item.embedding.record.storage_id,
        )
    )
    return tuple(ranked[:limit])


def _indexing_result(
    *,
    session_id: str,
    model: EmbeddingModelInfo,
    considered_chunks: int,
    embedded_chunks: int,
    cache_hits: int,
    invalidated_embeddings: int,
    failed_chunks: int,
    elapsed_time_ms: float,
    document_encoding_time_ms: float,
    sqlite_write_time_ms: float,
    stored_vector_bytes: int,
    batch_size: int,
    input_statistics: EmbeddingInputStatistics | None,
    runtime: EmbeddingRuntimeInfo | None,
    warnings: tuple[str, ...],
) -> SemanticIndexingResult:
    return SemanticIndexingResult(
        session_id=session_id,
        model=model,
        considered_chunks=considered_chunks,
        embedded_chunks=embedded_chunks,
        cache_hits=cache_hits,
        invalidated_embeddings=invalidated_embeddings,
        failed_chunks=failed_chunks,
        elapsed_time_ms=elapsed_time_ms,
        model_load_time_ms=runtime.model_load_time_ms if runtime is not None else 0.0,
        document_encoding_time_ms=document_encoding_time_ms,
        sqlite_write_time_ms=sqlite_write_time_ms,
        stored_vector_bytes=stored_vector_bytes,
        total_stored_vector_bytes=stored_vector_bytes,
        batch_size=batch_size,
        input_statistics=input_statistics,
        warnings=warnings,
    )


def _merge_input_statistics(
    values: Sequence[EmbeddingInputStatistics],
) -> EmbeddingInputStatistics | None:
    if not values:
        return None
    limits = {value.configured_max_sequence_length for value in values}
    if len(limits) != 1:
        raise RuntimeError("embedding model sequence length changed during indexing")
    input_count = sum(value.input_count for value in values)
    weighted_lengths = sum(
        value.average_tokenized_length * value.input_count for value in values
    )
    return EmbeddingInputStatistics(
        configured_max_sequence_length=next(iter(limits)),
        input_count=input_count,
        truncated_input_count=sum(value.truncated_input_count for value in values),
        maximum_tokenized_length=max(
            value.maximum_tokenized_length for value in values
        ),
        average_tokenized_length=(
            weighted_lengths / input_count if input_count else 0.0
        ),
    )
