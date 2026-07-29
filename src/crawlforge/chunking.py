"""Deterministic, heading-aware chunking for cleaned source documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from crawlforge.context_models import (
    DocumentBlock,
    HeuristicTokenEstimator,
    SourceDocument,
    TextChunk,
    TokenEstimator,
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based soft target, hard maximum, and overlap."""

    target_chars: int = 1_200
    max_chars: int = 1_600
    overlap_chars: int = 160

    def __post_init__(self) -> None:
        if self.target_chars <= 0:
            raise ValueError("target_chars must be greater than zero")
        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        if self.target_chars > self.max_chars:
            raise ValueError("target_chars must not exceed max_chars")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars must be smaller than target_chars")


class TextChunker:
    """Split source blocks without crossing heading-section boundaries."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        estimator: TokenEstimator | None = None,
    ) -> None:
        self._config = config or ChunkingConfig()
        self._estimator = estimator or HeuristicTokenEstimator()

    def chunk(self, document: SourceDocument) -> tuple[TextChunk, ...]:
        """Return stable, non-empty chunks under the configured hard maximum."""
        chunks: list[TextChunk] = []
        ordinal = 0

        for heading_path, blocks in self._sections(document.blocks):
            bodies = self._pack_section(heading_path, blocks)
            bodies = self._add_overlap(heading_path, bodies)
            for body in bodies:
                text = self._render_chunk(heading_path, body)
                if not text:
                    continue
                if len(text) > self._config.max_chars:
                    raise RuntimeError("chunk exceeds configured max_chars")
                content_hash = _chunk_content_hash(
                    document.title,
                    heading_path,
                    text,
                )
                chunk_id = _sha256(
                    "\0".join(
                        (
                            document.id,
                            str(ordinal),
                            content_hash,
                        )
                    )
                )
                chunks.append(
                    TextChunk(
                        id=chunk_id,
                        document_id=document.id,
                        ordinal=ordinal,
                        source_url=document.url,
                        document_title=document.title,
                        heading_path=heading_path,
                        text=text,
                        size_chars=len(text),
                        estimated_tokens=self._estimator.count(text),
                        content_hash=content_hash,
                    )
                )
                ordinal += 1

        return tuple(chunks)

    def _sections(
        self,
        blocks: tuple[DocumentBlock, ...],
    ) -> tuple[tuple[tuple[str, ...], tuple[DocumentBlock, ...]], ...]:
        sections: list[tuple[tuple[str, ...], list[DocumentBlock]]] = []
        for block in blocks:
            if block.kind == "heading" or not block.text.strip():
                continue
            if sections and sections[-1][0] == block.heading_path:
                sections[-1][1].append(block)
            else:
                sections.append((block.heading_path, [block]))
        return tuple((path, tuple(items)) for path, items in sections)

    def _pack_section(
        self,
        heading_path: tuple[str, ...],
        blocks: tuple[DocumentBlock, ...],
    ) -> list[str]:
        segments: list[str] = []
        for block in blocks:
            text = (
                block.text.strip("\n") if block.kind == "code" else block.text.strip()
            )
            if not text:
                continue
            if len(text) <= self._config.max_chars:
                segments.append(text)
            elif block.kind == "code":
                segments.extend(_split_code(text, self._config.max_chars))
            else:
                segments.extend(_split_prose(text, self._config.max_chars))

        bodies: list[str] = []
        current: list[str] = []
        for segment in segments:
            candidate = "\n\n".join((*current, segment))
            rendered = self._render_chunk(heading_path, candidate)
            if current and len(rendered) > self._config.target_chars:
                bodies.append("\n\n".join(current))
                current = [segment]
                continue
            if len(rendered) <= self._config.max_chars:
                current.append(segment)
                continue

            if current:
                bodies.append("\n\n".join(current))
            current = [segment]

        if current:
            bodies.append("\n\n".join(current))
        return bodies

    def _add_overlap(
        self,
        heading_path: tuple[str, ...],
        bodies: list[str],
    ) -> list[str]:
        overlap_chars = self._config.overlap_chars
        if overlap_chars == 0 or len(bodies) < 2:
            return bodies

        with_overlap = [bodies[0]]
        for previous, body in zip(bodies[:-1], bodies[1:], strict=True):
            prefix = _overlap_suffix(previous, overlap_chars)
            if prefix:
                candidate = f"{prefix}\n\n{body}"
                if (
                    len(self._render_chunk(heading_path, candidate))
                    <= self._config.max_chars
                ):
                    body = candidate
            with_overlap.append(body)
        return with_overlap

    def _render_chunk(
        self,
        heading_path: tuple[str, ...],
        body: str,
    ) -> str:
        normalized_body = body.strip("\n")
        if not normalized_body:
            return ""
        heading = "\n\n".join(value.strip() for value in heading_path if value.strip())
        if not heading:
            return normalized_body

        rendered = f"{heading}\n\n{normalized_body}"
        if len(rendered) <= self._config.max_chars:
            return rendered
        return normalized_body


def _split_prose(text: str, max_chars: int) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(text.strip())
        if sentence.strip()
    ]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(_split_words(sentence, max_chars))
    return _pack_units(units, max_chars, separator=" ")


def _split_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    units: list[str] = []
    for word in words:
        if len(word) <= max_chars:
            units.append(word)
        else:
            units.extend(
                word[offset : offset + max_chars]
                for offset in range(0, len(word), max_chars)
            )
    return _pack_units(units, max_chars, separator=" ")


def _split_code(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append("\n".join(current))
            current.clear()

    for line in text.splitlines():
        if len(line) > max_chars:
            flush()
            chunks.extend(
                line[offset : offset + max_chars]
                for offset in range(0, len(line), max_chars)
            )
            continue

        candidate = "\n".join((*current, line))
        if current and len(candidate) > max_chars:
            flush()
        current.append(line)
    flush()
    return [chunk for chunk in chunks if chunk]


def _pack_units(
    units: list[str],
    max_chars: int,
    *,
    separator: str,
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not unit:
            continue
        candidate = f"{current}{separator}{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = unit
    if current:
        chunks.append(current)
    return chunks


def _overlap_suffix(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    suffix = text[-max_chars:]
    first_space = suffix.find(" ")
    if first_space != -1 and first_space + 1 < len(suffix):
        suffix = suffix[first_space + 1 :]
    return suffix.strip()


def _chunk_content_hash(
    document_title: str,
    heading_path: tuple[str, ...],
    text: str,
) -> str:
    return _sha256("\0".join((document_title, *heading_path, text)))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
