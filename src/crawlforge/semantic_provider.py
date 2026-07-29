"""Optional Sentence Transformers embedding provider."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Protocol, cast

from crawlforge.async_utils import run_lifecycle_owned_thread
from crawlforge.semantic_models import (
    DEFAULT_SEMANTIC_DIMENSION,
    DEFAULT_SEMANTIC_MODEL_ID,
    DEFAULT_SEMANTIC_MODEL_REVISION,
    EMBEDDING_PRECISION,
    DeviceName,
    EmbeddingInputStatistics,
    EmbeddingRuntimeInfo,
    EmbeddingVector,
    JSONValue,
    SemanticDependencyError,
)

_DEPENDENCY_MESSAGE = (
    "Semantic retrieval requires the 'semantic' extra:\n"
    'pip install "crawlforge[semantic]"'
)


class _ArrayLike(Protocol):
    def tolist(self) -> object:
        """Return nested Python numeric sequences."""


class _Tokenizer(Protocol):
    def __call__(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> Mapping[str, object]:
        """Tokenize text without framework tensors."""


class _SentenceTransformerModel(Protocol):
    @property
    def device(self) -> object:
        """Return the selected inference device."""

    @property
    def max_seq_length(self) -> int:
        """Return the configured maximum input sequence length."""

    @property
    def tokenizer(self) -> _Tokenizer:
        """Return the underlying tokenizer."""

    def get_embedding_dimension(self) -> int | None:
        """Return the actual embedding dimension."""

    def encode_document(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
        precision: str,
    ) -> _ArrayLike:
        """Encode documents."""

    def encode_query(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
        precision: str,
    ) -> _ArrayLike:
        """Encode queries."""


class SentenceTransformerEmbeddingProvider:
    """Lazy local embeddings through the optional sentence-transformers package."""

    implementation = "sentence-transformers"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
        revision: str | None = DEFAULT_SEMANTIC_MODEL_REVISION,
        dimension: int = DEFAULT_SEMANTIC_DIMENSION,
        device: DeviceName = "auto",
        batch_size: int = 32,
        cache_directory: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if revision is not None and not revision.strip():
            raise ValueError("revision must be non-empty when provided")
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, mps, cuda")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._model_id = model_id
        self._model_revision = revision
        self._dimension = dimension
        self._device = device
        self._batch_size = batch_size
        self._cache_directory = (
            Path(cache_directory).expanduser() if cache_directory is not None else None
        )
        self._local_files_only = local_files_only
        self._model: _SentenceTransformerModel | None = None
        self._load_task: asyncio.Task[_SentenceTransformerModel] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._runtime_info: EmbeddingRuntimeInfo | None = None
        self._closed = False

    async def __aenter__(self) -> SentenceTransformerEmbeddingProvider:
        """Enter without loading the model."""
        if self._closed:
            raise RuntimeError("embedding provider is closed")
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Release provider-owned references."""
        await self.close()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str | None:
        return self._model_revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalized(self) -> bool:
        return True

    @property
    def precision(self) -> str:
        return EMBEDDING_PRECISION

    @property
    def batch_size(self) -> int:
        """Return the configured inference batch size."""
        return self._batch_size

    @property
    def requested_device(self) -> DeviceName:
        """Return the configured device policy."""
        return self._device

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        metadata: dict[str, JSONValue] = {
            "batch_size": self._batch_size,
            "requested_device": self._device,
            "local_files_only": self._local_files_only,
            "trust_remote_code": False,
        }
        runtime = self._runtime_info
        if runtime is not None:
            metadata.update(
                {
                    "actual_device": runtime.device,
                    "sentence_transformers_version": (
                        runtime.sentence_transformers_version
                    ),
                    "transformers_version": runtime.transformers_version,
                    "torch_version": runtime.torch_version,
                    "max_sequence_length": runtime.max_sequence_length,
                }
            )
        return metadata

    @property
    def runtime_info(self) -> EmbeddingRuntimeInfo | None:
        return self._runtime_info

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        """Encode document inputs outside the event loop."""
        return await self._embed(texts, document=True)

    async def embed_queries(
        self,
        texts: Sequence[str],
    ) -> Sequence[EmbeddingVector]:
        """Encode query inputs outside the event loop."""
        return await self._embed(texts, document=False)

    async def analyze_document_inputs(
        self,
        texts: Sequence[str],
    ) -> EmbeddingInputStatistics:
        """Measure untruncated tokenizer lengths outside the event loop."""
        model = await self._get_model()
        copied = tuple(texts)
        if any(not text for text in copied):
            raise ValueError("embedding inputs must not be empty")
        async with self._inference_lock:
            return await run_lifecycle_owned_thread(
                self._analyze_inputs_sync,
                model,
                copied,
            )

    async def close(self) -> None:
        """Wait for owned work and release the loaded model reference."""
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        """Complete cleanup even if the caller awaiting close is cancelled."""
        task = self._load_task
        if task is not None:
            try:
                await asyncio.shield(task)
            except (Exception, asyncio.CancelledError):
                pass
        async with self._inference_lock:
            self._model = None

    async def _embed(
        self,
        texts: Sequence[str],
        *,
        document: bool,
    ) -> tuple[EmbeddingVector, ...]:
        copied = tuple(texts)
        if not copied:
            return ()
        if any(not text for text in copied):
            raise ValueError("embedding inputs must not be empty")
        model = await self._get_model()
        async with self._inference_lock:
            return await run_lifecycle_owned_thread(
                self._encode_sync,
                model,
                copied,
                document,
            )

    async def _get_model(self) -> _SentenceTransformerModel:
        if self._closed:
            raise RuntimeError("embedding provider is closed")
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._load_task is None:
                self._load_task = asyncio.create_task(
                    run_lifecycle_owned_thread(self._load_model_sync)
                )
            task = self._load_task
        model = await asyncio.shield(task)
        self._model = model
        return model

    def _load_model_sync(self) -> _SentenceTransformerModel:
        started = perf_counter()
        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as error:
            raise SemanticDependencyError(_DEPENDENCY_MESSAGE) from error

        factory = cast(
            Callable[..., _SentenceTransformerModel],
            module.__dict__["SentenceTransformer"],
        )
        model = factory(
            self._model_id,
            revision=self._model_revision,
            device=None if self._device == "auto" else self._device,
            cache_folder=(
                str(self._cache_directory)
                if self._cache_directory is not None
                else None
            ),
            trust_remote_code=False,
            local_files_only=self._local_files_only,
        )
        actual_dimension = model.get_embedding_dimension()
        if actual_dimension is None:
            raise RuntimeError("embedding model did not report its dimension")
        if actual_dimension != self._dimension:
            raise ValueError(
                "embedding model dimension mismatch: "
                f"expected {self._dimension}, received {actual_dimension}"
            )
        maximum_length = model.max_seq_length
        if maximum_length <= 0:
            raise RuntimeError("embedding model reported an invalid sequence length")
        self._runtime_info = EmbeddingRuntimeInfo(
            device=str(model.device),
            sentence_transformers_version=_package_version("sentence-transformers"),
            transformers_version=_package_version("transformers"),
            torch_version=_package_version("torch"),
            model_load_time_ms=(perf_counter() - started) * 1000,
            max_sequence_length=maximum_length,
            model_cache_size_bytes=_directory_size(self._cache_directory),
        )
        return model

    def _encode_sync(
        self,
        model: _SentenceTransformerModel,
        texts: tuple[str, ...],
        document: bool,
    ) -> tuple[EmbeddingVector, ...]:
        encode = model.encode_document if document else model.encode_query
        encoded = encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=True,
            precision=EMBEDDING_PRECISION,
        )
        return _embedding_matrix(encoded)

    @staticmethod
    def _analyze_inputs_sync(
        model: _SentenceTransformerModel,
        texts: tuple[str, ...],
    ) -> EmbeddingInputStatistics:
        tokenized = model.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        input_ids = tokenized.get("input_ids")
        if not isinstance(input_ids, Sequence) or isinstance(input_ids, str | bytes):
            raise RuntimeError("embedding tokenizer returned invalid input_ids")
        lengths: list[int] = []
        for row in input_ids:
            if not isinstance(row, Sequence) or isinstance(row, str | bytes):
                raise RuntimeError("embedding tokenizer returned invalid token rows")
            lengths.append(len(row))
        if len(lengths) != len(texts):
            raise RuntimeError("embedding tokenizer returned an unexpected batch size")
        maximum = model.max_seq_length
        return EmbeddingInputStatistics(
            configured_max_sequence_length=maximum,
            input_count=len(lengths),
            truncated_input_count=sum(length > maximum for length in lengths),
            maximum_tokenized_length=max(lengths, default=0),
            average_tokenized_length=(
                math.fsum(lengths) / len(lengths) if lengths else 0.0
            ),
        )


def _embedding_matrix(value: _ArrayLike) -> tuple[EmbeddingVector, ...]:
    rows = value.tolist()
    if not isinstance(rows, list):
        raise RuntimeError("embedding provider returned an invalid array")
    vectors: list[EmbeddingVector] = []
    for row in rows:
        if not isinstance(row, list):
            raise RuntimeError("embedding provider returned a non-matrix array")
        vectors.append(EmbeddingVector.from_sequence(row))
    return tuple(vectors)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _directory_size(path: Path | None) -> int | None:
    if path is None or not path.is_dir():
        return None
    total = 0
    try:
        for candidate in path.rglob("*"):
            if candidate.is_file():
                total += candidate.stat().st_size
    except OSError:
        return None
    return total
