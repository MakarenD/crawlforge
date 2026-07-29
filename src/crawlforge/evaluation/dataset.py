"""Loading and validation for versioned offline retrieval datasets."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from crawlforge.evaluation.models import (
    QUERY_CATEGORIES,
    EvaluationDataset,
    EvaluationDocument,
    EvaluationQuery,
    EvaluationSection,
    QueryCategory,
    RelevanceJudgment,
)

_SUPPORTED_SCHEMA_VERSION = 1
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class DatasetValidationError(ValueError):
    """Report every deterministic dataset validation problem together."""

    def __init__(self, issues: list[str] | tuple[str, ...]) -> None:
        self.issues = tuple(issues)
        super().__init__("dataset validation failed: " + "; ".join(self.issues))


def load_dataset(root: str | Path) -> EvaluationDataset:
    """Load and validate a local manifest, documents, and JSONL queries."""
    dataset_root = Path(root)
    issues: list[str] = []
    manifest = _read_json_object(dataset_root / "manifest.json", issues)
    query_records = _read_json_lines(dataset_root / "queries.jsonl", issues)

    schema_version = _integer(manifest, "schema_version", "manifest", issues)
    name = _string(manifest, "name", "manifest", issues)
    version = _string(manifest, "version", "manifest", issues)
    description = _string(manifest, "description", "manifest", issues)
    documents = _load_documents(
        dataset_root,
        manifest.get("documents"),
        issues,
    )
    queries = _load_queries(query_records, issues)
    if issues:
        raise DatasetValidationError(issues)

    dataset = EvaluationDataset(
        schema_version=schema_version,
        name=name,
        version=version,
        description=description,
        documents=tuple(documents),
        queries=tuple(queries),
        root=dataset_root,
    )
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: EvaluationDataset) -> None:
    """Validate stable identities, source references, sections, and ground truth."""
    issues: list[str] = []
    if dataset.schema_version != _SUPPORTED_SCHEMA_VERSION:
        issues.append(
            "manifest schema_version must be "
            f"{_SUPPORTED_SCHEMA_VERSION}, got {dataset.schema_version}"
        )
    if not dataset.name.strip():
        issues.append("manifest name must not be empty")
    if not _VERSION_PATTERN.fullmatch(dataset.version):
        issues.append("manifest version must use MAJOR.MINOR.PATCH")
    if not dataset.documents:
        issues.append("corpus must contain at least one document")
    if not dataset.queries:
        issues.append("dataset must contain at least one query")

    documents: dict[str, EvaluationDocument] = {}
    urls: set[str] = set()
    for document in dataset.documents:
        if document.document_id in documents:
            issues.append(f"duplicate document_id: {document.document_id}")
        documents[document.document_id] = document
        if document.url in urls:
            issues.append(f"duplicate document URL: {document.url}")
        urls.add(document.url)
        _validate_document(document, issues)

    query_ids: set[str] = set()
    judgment_ids: set[str] = set()
    for query in dataset.queries:
        if query.query_id in query_ids:
            issues.append(f"duplicate query_id: {query.query_id}")
        query_ids.add(query.query_id)
        if not query.query.strip():
            issues.append(f"{query.query_id}: query must not be empty")
        if query.category not in QUERY_CATEGORIES:
            issues.append(f"{query.query_id}: unsupported category {query.category!r}")

        positive_judgments = 0
        judgment_targets: set[tuple[str, str, object]] = set()
        for judgment in query.relevant_sources:
            if judgment.judgment_id in judgment_ids:
                issues.append(f"duplicate judgment_id: {judgment.judgment_id}")
            judgment_ids.add(judgment.judgment_id)
            positive_judgments += int(judgment.relevance > 0)
            _validate_judgment(query, judgment, documents, issues)
            target = _judgment_target(judgment, documents)
            if target in judgment_targets:
                source = judgment.section_id or judgment.heading_path
                issues.append(
                    f"{query.query_id}: duplicate relevance target "
                    f"{judgment.document_id}/{source}"
                )
            judgment_targets.add(target)

        if query.category == "negative" and positive_judgments:
            issues.append(
                f"{query.query_id}: negative queries must not have "
                "positive ground truth"
            )
        elif query.category != "negative" and positive_judgments == 0:
            issues.append(
                f"{query.query_id}: non-negative query needs positive ground truth"
            )

    if issues:
        raise DatasetValidationError(issues)


def filter_dataset(
    dataset: EvaluationDataset,
    *,
    category: QueryCategory | None = None,
    query_ids: frozenset[str] | None = None,
) -> EvaluationDataset:
    """Return a validated query subset while retaining the complete corpus."""
    known_query_ids = {query.query_id for query in dataset.queries}
    if query_ids:
        unknown = sorted(query_ids - known_query_ids)
        if unknown:
            raise ValueError(f"unknown query IDs: {', '.join(unknown)}")
    selected = tuple(
        query
        for query in dataset.queries
        if (category is None or query.category == category)
        and (not query_ids or query.query_id in query_ids)
    )
    if not selected:
        raise ValueError("evaluation filters selected no queries")
    return replace(dataset, queries=selected)


def _load_documents(
    root: Path,
    raw_documents: object,
    issues: list[str],
) -> list[EvaluationDocument]:
    if not isinstance(raw_documents, list):
        issues.append("manifest.documents must be a list")
        return []

    documents: list[EvaluationDocument] = []
    for index, raw_document in enumerate(raw_documents):
        location = f"manifest.documents[{index}]"
        document = _object(raw_document, location, issues)
        if document is None:
            continue
        document_id = _string(document, "document_id", location, issues)
        relative_path = _string(document, "path", location, issues)
        url = _string(document, "url", location, issues)
        title = _string(document, "title", location, issues)
        sections = _load_sections(document.get("sections"), location, issues)
        content = ""
        document_path = _safe_dataset_path(root, relative_path, location, issues)
        if document_path is not None:
            try:
                content = document_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                issues.append(f"{location}.path cannot be read: {type(error).__name__}")
        documents.append(
            EvaluationDocument(
                document_id=document_id,
                path=relative_path,
                url=url,
                title=title,
                sections=tuple(sections),
                content=content,
            )
        )
    return documents


def _load_sections(
    raw_sections: object,
    location: str,
    issues: list[str],
) -> list[EvaluationSection]:
    if not isinstance(raw_sections, list):
        issues.append(f"{location}.sections must be a list")
        return []
    sections: list[EvaluationSection] = []
    for index, raw_section in enumerate(raw_sections):
        section_location = f"{location}.sections[{index}]"
        section = _object(raw_section, section_location, issues)
        if section is None:
            continue
        sections.append(
            EvaluationSection(
                section_id=_string(
                    section,
                    "section_id",
                    section_location,
                    issues,
                ),
                heading_path=_string_tuple(
                    section.get("heading_path"),
                    f"{section_location}.heading_path",
                    issues,
                ),
            )
        )
    return sections


def _load_queries(
    query_records: list[dict[str, object]],
    issues: list[str],
) -> list[EvaluationQuery]:
    queries: list[EvaluationQuery] = []
    for index, record in enumerate(query_records, start=1):
        location = f"queries.jsonl:{index}"
        raw_category = _string(record, "category", location, issues)
        category: QueryCategory
        if raw_category in QUERY_CATEGORIES:
            category = raw_category
        else:
            issues.append(f"{location}.category is unsupported: {raw_category!r}")
            category = "negative"
        raw_sources = record.get("relevant_sources")
        if not isinstance(raw_sources, list):
            issues.append(f"{location}.relevant_sources must be a list")
            raw_sources = []
        judgments: list[RelevanceJudgment] = []
        for source_index, raw_source in enumerate(raw_sources):
            source_location = f"{location}.relevant_sources[{source_index}]"
            source = _object(raw_source, source_location, issues)
            if source is None:
                continue
            judgments.append(
                RelevanceJudgment(
                    judgment_id=_string(
                        source,
                        "judgment_id",
                        source_location,
                        issues,
                    ),
                    document_id=_string(
                        source,
                        "document_id",
                        source_location,
                        issues,
                    ),
                    relevance=_integer(
                        source,
                        "relevance",
                        source_location,
                        issues,
                    ),
                    canonical_source=_optional_string(
                        source,
                        "canonical_source",
                        source_location,
                        issues,
                    ),
                    section_id=_optional_string(
                        source,
                        "section_id",
                        source_location,
                        issues,
                    ),
                    heading_path=_optional_string_tuple(
                        source.get("heading_path"),
                        f"{source_location}.heading_path",
                        issues,
                    ),
                    evidence=_optional_string(
                        source,
                        "evidence",
                        source_location,
                        issues,
                    ),
                )
            )
        queries.append(
            EvaluationQuery(
                query_id=_string(record, "query_id", location, issues),
                query=_string(record, "query", location, issues),
                category=category,
                relevant_sources=tuple(judgments),
            )
        )
    return queries


def _validate_document(
    document: EvaluationDocument,
    issues: list[str],
) -> None:
    location = f"document {document.document_id or '<empty>'}"
    if not document.document_id.strip():
        issues.append(f"{location}: document_id must not be empty")
    if _is_unsafe_relative_path(document.path):
        issues.append(f"{location}: path must be a safe relative path")
    parsed_url = urlsplit(document.url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        issues.append(f"{location}: URL must be absolute HTTP(S)")
    if not document.content.strip():
        issues.append(f"{location}: content must not be empty")

    soup = BeautifulSoup(document.content, "lxml")
    actual_heading_paths = _html_heading_paths(soup)
    section_ids: set[str] = set()
    heading_paths: set[tuple[str, ...]] = set()
    for section in document.sections:
        if not section.section_id.strip():
            issues.append(f"{location}: section_id must not be empty")
        if section.section_id in section_ids:
            issues.append(f"{location}: duplicate section_id {section.section_id}")
        section_ids.add(section.section_id)
        if soup.find(id=section.section_id) is None:
            issues.append(
                f"{location}/{section.section_id}: section_id is absent from HTML"
            )
        if not section.heading_path:
            issues.append(f"{location}/{section.section_id}: heading_path is empty")
        elif section.heading_path in heading_paths:
            issues.append(
                f"{location}: duplicate heading_path {section.heading_path!r}"
            )
        elif section.heading_path not in actual_heading_paths:
            issues.append(
                f"{location}/{section.section_id}: heading_path is absent from HTML"
            )
        heading_paths.add(section.heading_path)


def _validate_judgment(
    query: EvaluationQuery,
    judgment: RelevanceJudgment,
    documents: dict[str, EvaluationDocument],
    issues: list[str],
) -> None:
    location = f"{query.query_id}/{judgment.judgment_id or '<empty>'}"
    if not judgment.judgment_id.strip():
        issues.append(f"{location}: judgment_id must not be empty")
    if judgment.relevance not in {0, 1, 2, 3}:
        issues.append(f"{location}: relevance must be between 0 and 3")
    document = documents.get(judgment.document_id)
    if document is None:
        issues.append(f"{location}: unknown relevant document {judgment.document_id!r}")
        return
    if (
        judgment.canonical_source is not None
        and judgment.canonical_source != document.url
    ):
        issues.append(f"{location}: canonical_source does not match document URL")

    section = None
    if judgment.section_id is not None:
        section = next(
            (
                candidate
                for candidate in document.sections
                if candidate.section_id == judgment.section_id
            ),
            None,
        )
        if section is None:
            issues.append(f"{location}: unknown section_id {judgment.section_id!r}")
    if judgment.heading_path:
        expected_path = section.heading_path if section is not None else None
        if expected_path is not None and judgment.heading_path != expected_path:
            issues.append(f"{location}: heading_path does not match section_id")
        elif expected_path is None and judgment.heading_path not in {
            item.heading_path for item in document.sections
        }:
            issues.append(f"{location}: heading_path does not exist")
    if judgment.evidence is not None:
        evidence = _normalize_text(judgment.evidence)
        soup = BeautifulSoup(document.content, "lxml")
        section_element = (
            soup.find(id=judgment.section_id)
            if judgment.section_id is not None
            else None
        )
        evidence_scope = section_element or soup
        document_text = _normalize_text(evidence_scope.get_text(" ", strip=True))
        if not evidence:
            issues.append(f"{location}: evidence must not be empty")
        elif evidence not in document_text:
            issues.append(f"{location}: evidence is absent from document")


def _judgment_target(
    judgment: RelevanceJudgment,
    documents: dict[str, EvaluationDocument],
) -> tuple[str, str, object]:
    document = documents.get(judgment.document_id)
    if judgment.section_id is not None:
        return ("section", judgment.document_id, judgment.section_id)
    if judgment.heading_path:
        if document is not None:
            section = next(
                (
                    candidate
                    for candidate in document.sections
                    if candidate.heading_path == judgment.heading_path
                ),
                None,
            )
            if section is not None:
                return ("section", judgment.document_id, section.section_id)
        return ("heading", judgment.document_id, judgment.heading_path)
    return ("document", judgment.document_id, "")


def _html_heading_paths(soup: BeautifulSoup) -> set[tuple[str, ...]]:
    stack: list[tuple[int, str]] = []
    paths: set[tuple[str, ...]] = set()
    for heading in soup.find_all(("h1", "h2", "h3", "h4", "h5", "h6")):
        text = " ".join(heading.get_text(" ", strip=True).split())
        if not text:
            continue
        level = int(heading.name[1])
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        paths.add(tuple(item[1] for item in stack))
    return paths


def _read_json_object(path: Path, issues: list[str]) -> dict[str, object]:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        issues.append(f"{path.name} cannot be read: {type(error).__name__}")
        return {}
    if not isinstance(loaded, dict):
        issues.append(f"{path.name} must contain a JSON object")
        return {}
    return cast(dict[str, object], loaded)


def _read_json_lines(path: Path, issues: list[str]) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        issues.append(f"{path.name} cannot be read: {type(error).__name__}")
        return []
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            issues.append(f"{path.name}:{line_number} must not be blank")
            continue
        try:
            loaded: object = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"{path.name}:{line_number} is not valid JSON")
            continue
        if not isinstance(loaded, dict):
            issues.append(f"{path.name}:{line_number} must contain an object")
            continue
        records.append(cast(dict[str, object], loaded))
    return records


def _safe_dataset_path(
    root: Path,
    relative_path: str,
    location: str,
    issues: list[str],
) -> Path | None:
    if _is_unsafe_relative_path(relative_path):
        issues.append(f"{location}.path must be a safe relative path")
        return None
    candidate = root / relative_path
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        issues.append(f"{location}.path escapes the dataset root")
        return None
    return candidate


def _is_unsafe_relative_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    )


def _object(
    value: object,
    location: str,
    issues: list[str],
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        issues.append(f"{location} must be an object")
        return None
    return cast(dict[str, object], value)


def _string(
    value: dict[str, object],
    key: str,
    location: str,
    issues: list[str],
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        issues.append(f"{location}.{key} must be a non-empty string")
        return ""
    return item


def _optional_string(
    value: dict[str, object],
    key: str,
    location: str,
    issues: list[str],
) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item.strip():
        issues.append(f"{location}.{key} must be a non-empty string when provided")
        return None
    return item


def _integer(
    value: dict[str, object],
    key: str,
    location: str,
    issues: list[str],
) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        issues.append(f"{location}.{key} must be an integer")
        return 0
    return item


def _string_tuple(
    value: object,
    location: str,
    issues: list[str],
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        issues.append(f"{location} must be a non-empty string list")
        return ()
    if not all(isinstance(item, str) and item.strip() for item in value):
        issues.append(f"{location} must contain only non-empty strings")
        return ()
    return tuple(cast(list[str], value))


def _optional_string_tuple(
    value: object,
    location: str,
    issues: list[str],
) -> tuple[str, ...]:
    if value is None:
        return ()
    return _string_tuple(value, location, issues)


def _normalize_text(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return re.sub(r"\s+([.,;:!?])", r"\1", normalized)
