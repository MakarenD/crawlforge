from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

import crawlforge.semantic_provider as provider_module
from crawlforge.semantic_models import SemanticDependencyError
from crawlforge.semantic_provider import SentenceTransformerEmbeddingProvider


class _Array:
    def __init__(self, values: object) -> None:
        self._values = values

    def tolist(self) -> object:
        return self._values


class _Tokenizer:
    def __call__(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, object]:
        assert add_special_tokens
        assert not padding
        assert not truncation
        return {
            "input_ids": [
                list(range(300 if "long" in text else len(text.split()) + 2))
                for text in texts
            ]
        }


class _Model:
    def __init__(
        self,
        *,
        dimension: int = 3,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
        invalid: float | None = None,
    ) -> None:
        self.dimension = dimension
        self.device = "cpu"
        self.max_seq_length = 256
        self.tokenizer = _Tokenizer()
        self.started = started
        self.release = release
        self.invalid = invalid
        self.document_calls: list[tuple[tuple[str, ...], int]] = []
        self.query_calls: list[tuple[tuple[str, ...], int]] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

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
    ) -> _Array:
        self._assert_options(
            show_progress_bar,
            convert_to_numpy,
            convert_to_tensor,
            normalize_embeddings,
            precision,
        )
        copied = tuple(texts)
        self.document_calls.append((copied, batch_size))
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=5)
        value = self.invalid if self.invalid is not None else 1.0
        return _Array([[value, 0.0, 0.0] for _ in copied])

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
    ) -> _Array:
        self._assert_options(
            show_progress_bar,
            convert_to_numpy,
            convert_to_tensor,
            normalize_embeddings,
            precision,
        )
        copied = tuple(texts)
        self.query_calls.append((copied, batch_size))
        return _Array([[1.0, 0.0, 0.0] for _ in copied])

    @staticmethod
    def _assert_options(
        show_progress_bar: bool,
        convert_to_numpy: bool,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
        precision: str,
    ) -> None:
        assert not show_progress_bar
        assert convert_to_numpy
        assert not convert_to_tensor
        assert normalize_embeddings
        assert precision == "float32"


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    model: _Model,
) -> list[dict[str, object]]:
    construction_calls: list[dict[str, object]] = []

    def factory(_model_id: str, **kwargs: object) -> _Model:
        construction_calls.append(kwargs)
        return model

    module = SimpleNamespace(SentenceTransformer=factory)
    monkeypatch.setattr(
        provider_module.importlib,
        "import_module",
        lambda name: module if name == "sentence_transformers" else None,
    )
    monkeypatch.setattr(provider_module, "_package_version", lambda _name: "test")
    return construction_calls


@pytest.mark.asyncio
async def test_provider_loads_once_lazily_and_separates_query_document_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    construction_calls = _install_fake_module(monkeypatch, model)
    provider = SentenceTransformerEmbeddingProvider(
        model_id="test/model",
        revision="immutable",
        dimension=3,
        device="cpu",
        batch_size=2,
        local_files_only=True,
    )

    assert construction_calls == []
    documents = await provider.embed_documents(("doc-a", "doc-b"))
    queries = await provider.embed_queries(("query",))
    await provider.embed_documents(("doc-c",))
    runtime = provider.runtime_info
    await provider.close()
    await provider.close()

    assert len(construction_calls) == 1
    assert construction_calls[0]["revision"] == "immutable"
    assert construction_calls[0]["device"] == "cpu"
    assert construction_calls[0]["trust_remote_code"] is False
    assert construction_calls[0]["local_files_only"] is True
    assert len(documents) == 2
    assert len(queries) == 1
    assert model.document_calls == [
        (("doc-a", "doc-b"), 2),
        (("doc-c",), 2),
    ]
    assert model.query_calls == [(("query",), 2)]
    assert runtime is not None
    assert runtime.device == "cpu"


@pytest.mark.asyncio
async def test_provider_checks_actual_model_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch, _Model(dimension=4))
    provider = SentenceTransformerEmbeddingProvider(dimension=3)

    with pytest.raises(ValueError, match="dimension mismatch"):
        await provider.embed_queries(("query",))


@pytest.mark.asyncio
async def test_provider_rejects_invalid_model_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch, _Model(invalid=math.nan))
    provider = SentenceTransformerEmbeddingProvider(dimension=3)

    with pytest.raises(ValueError, match="must be finite"):
        await provider.embed_documents(("document",))


@pytest.mark.asyncio
async def test_provider_reports_optional_dependency_without_traceback_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> None:
        raise ModuleNotFoundError("sentence_transformers")

    monkeypatch.setattr(provider_module.importlib, "import_module", missing)
    provider = SentenceTransformerEmbeddingProvider()

    with pytest.raises(SemanticDependencyError) as caught:
        await provider.embed_queries(("query",))

    assert str(caught.value) == (
        "Semantic retrieval requires the 'semantic' extra:\n"
        'pip install "crawlforge[semantic]"'
    )


@pytest.mark.asyncio
async def test_provider_measures_truncation_without_retaining_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch, _Model())
    provider = SentenceTransformerEmbeddingProvider(dimension=3)

    statistics = await provider.analyze_document_inputs(("short input", "long"))

    assert statistics.configured_max_sequence_length == 256
    assert statistics.input_count == 2
    assert statistics.truncated_input_count == 1
    assert statistics.maximum_tokenized_length == 300
    assert statistics.average_tokenized_length == 152.0


@pytest.mark.asyncio
async def test_provider_propagates_cancellation_after_owned_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    _install_fake_module(
        monkeypatch,
        _Model(started=started, release=release),
    )
    provider = SentenceTransformerEmbeddingProvider(dimension=3)

    task = asyncio.create_task(provider.embed_documents(("document",)))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    await provider.close()


@pytest.mark.asyncio
async def test_provider_rejects_calls_after_close() -> None:
    provider = SentenceTransformerEmbeddingProvider()
    await provider.close()

    with pytest.raises(RuntimeError, match="closed"):
        await provider.embed_queries(("query",))


@pytest.mark.asyncio
async def test_provider_close_cleanup_survives_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    _install_fake_module(
        monkeypatch,
        _Model(started=started, release=release),
    )
    provider = SentenceTransformerEmbeddingProvider(dimension=3)
    embedding = asyncio.create_task(provider.embed_documents(("document",)))
    await asyncio.to_thread(started.wait, 5)

    closing = asyncio.create_task(provider.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    release.set()
    await embedding
    await provider.close()

    assert provider._model is None
