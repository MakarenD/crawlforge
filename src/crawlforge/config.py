"""Validated JSON configuration for advanced crawling."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from crawlforge.storage import CSVStorage, DataStorage, JSONStorage, SQLiteStorage
from crawlforge.urls import canonical_hostname

StorageFormat = Literal["json", "csv", "sqlite"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Configuration for one built-in asynchronous storage backend."""

    format: StorageFormat
    path: Path
    json_lines: bool = True
    indent: int | None = None
    encoding: str = "utf-8"
    batch_size: int = 100

    def create(self) -> DataStorage:
        """Create the configured asynchronous storage backend."""
        if self.format == "json":
            return JSONStorage(
                self.path,
                json_lines=self.json_lines,
                indent=self.indent,
                encoding=self.encoding,
            )
        if self.format == "csv":
            return CSVStorage(self.path, encoding=self.encoding)
        return SQLiteStorage(self.path, batch_size=self.batch_size)


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Console and rotating-file logging settings."""

    level: LogLevel = "INFO"
    file: Path | None = None
    max_bytes: int = 5_000_000
    backup_count: int = 3


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """Default paths for crawl result exports."""

    json: Path | None = None
    html: Path | None = None


@dataclass(frozen=True, slots=True)
class CrawlerConfig:
    """Complete validated configuration for :class:`AdvancedCrawler`."""

    start_urls: tuple[str, ...]
    sitemap_urls: tuple[str, ...] = ()
    max_pages: int = 100
    max_depth: int = 2
    max_concurrent: int = 10
    max_concurrent_per_domain: int | None = None
    rate_limit: float = 1.0
    rate_limit_per_domain: bool = True
    respect_robots: bool = True
    min_delay: float = 0.0
    jitter: float = 0.0
    max_retries: int = 2
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    total_timeout: float | None = None
    same_domain_only: bool = False
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    storage: StorageConfig | None = None
    logging: LoggingConfig = LoggingConfig()
    reports: ReportConfig = ReportConfig()

    def __post_init__(self) -> None:
        if not self.start_urls and not self.sitemap_urls:
            raise ValueError("configuration requires urls or sitemaps")
        for url in (*self.start_urls, *self.sitemap_urls):
            _validate_http_url(url)
        if self.max_pages <= 0:
            raise ValueError("max_pages must be greater than zero")
        if self.max_depth < 0:
            raise ValueError("max_depth must be zero or greater")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")
        if (
            self.max_concurrent_per_domain is not None
            and self.max_concurrent_per_domain <= 0
        ):
            raise ValueError("max_concurrent_per_domain must be greater than zero")
        for name, value in (
            ("rate_limit", self.rate_limit),
            ("connect_timeout", self.connect_timeout),
            ("read_timeout", self.read_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive value")
        if self.total_timeout is not None and (
            not math.isfinite(self.total_timeout) or self.total_timeout <= 0
        ):
            raise ValueError("total_timeout must be a finite positive value")
        for name, value in (("min_delay", self.min_delay), ("jitter", self.jitter)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative value")
        if self.max_retries < 0:
            raise ValueError("max_retries must be zero or greater")
        for name, patterns in (
            ("include_patterns", self.include_patterns),
            ("exclude_patterns", self.exclude_patterns),
        ):
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as error:
                    raise ValueError(f"invalid regex in {name}: {error}") from error
        if self.logging.max_bytes <= 0:
            raise ValueError("logging.max_bytes must be greater than zero")
        if self.logging.backup_count < 0:
            raise ValueError("logging.backup_count must be zero or greater")

    @classmethod
    def from_file(cls, filename: str | Path) -> CrawlerConfig:
        """Load and validate a JSON configuration file."""
        path = Path(filename)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"could not read configuration {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON configuration {path}: line {error.lineno}, "
                f"column {error.colno}"
            ) from error
        if not isinstance(raw, Mapping):
            raise ValueError("configuration root must be a JSON object")
        return cls._from_mapping(raw, base_dir=path.resolve().parent)

    @classmethod
    def _from_mapping(
        cls,
        raw: Mapping[object, object],
        *,
        base_dir: Path,
    ) -> CrawlerConfig:
        _reject_unknown(
            raw,
            {"urls", "sitemaps", "crawler", "filters", "storage", "logging", "reports"},
            "configuration",
        )
        crawler = _mapping(raw.get("crawler"), "crawler")
        filters = _mapping(raw.get("filters"), "filters")
        _reject_unknown(
            crawler,
            {
                "max_pages",
                "max_depth",
                "max_concurrent",
                "max_concurrent_per_domain",
                "rate_limit",
                "rate_limit_per_domain",
                "respect_robots",
                "min_delay",
                "jitter",
                "max_retries",
                "connect_timeout",
                "read_timeout",
                "total_timeout",
            },
            "crawler",
        )
        _reject_unknown(
            filters,
            {"same_domain_only", "include", "exclude"},
            "filters",
        )

        return cls(
            start_urls=_string_tuple(raw.get("urls"), "urls"),
            sitemap_urls=_string_tuple(raw.get("sitemaps"), "sitemaps"),
            max_pages=_integer(crawler.get("max_pages"), "max_pages", 100),
            max_depth=_integer(crawler.get("max_depth"), "max_depth", 2),
            max_concurrent=_integer(
                crawler.get("max_concurrent"),
                "max_concurrent",
                10,
            ),
            max_concurrent_per_domain=_optional_integer(
                crawler.get("max_concurrent_per_domain"),
                "max_concurrent_per_domain",
            ),
            rate_limit=_number(crawler.get("rate_limit"), "rate_limit", 1.0),
            rate_limit_per_domain=_boolean(
                crawler.get("rate_limit_per_domain"),
                "rate_limit_per_domain",
                True,
            ),
            respect_robots=_boolean(
                crawler.get("respect_robots"),
                "respect_robots",
                True,
            ),
            min_delay=_number(crawler.get("min_delay"), "min_delay", 0.0),
            jitter=_number(crawler.get("jitter"), "jitter", 0.0),
            max_retries=_integer(
                crawler.get("max_retries"),
                "max_retries",
                2,
            ),
            connect_timeout=_number(
                crawler.get("connect_timeout"),
                "connect_timeout",
                10.0,
            ),
            read_timeout=_number(
                crawler.get("read_timeout"),
                "read_timeout",
                30.0,
            ),
            total_timeout=_optional_number(
                crawler.get("total_timeout"),
                "total_timeout",
            ),
            same_domain_only=_boolean(
                filters.get("same_domain_only"),
                "same_domain_only",
                False,
            ),
            include_patterns=_string_tuple(filters.get("include"), "filters.include"),
            exclude_patterns=_string_tuple(filters.get("exclude"), "filters.exclude"),
            storage=_storage_config(raw.get("storage"), base_dir=base_dir),
            logging=_logging_config(raw.get("logging"), base_dir=base_dir),
            reports=_report_config(raw.get("reports"), base_dir=base_dir),
        )

    def with_overrides(
        self,
        *,
        start_urls: Sequence[str] | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        rate_limit: float | None = None,
        respect_robots: bool | None = None,
        json_report: Path | None = None,
    ) -> CrawlerConfig:
        """Return a validated copy with explicit command-line overrides."""
        reports = (
            replace(self.reports, json=json_report)
            if json_report is not None
            else self.reports
        )
        return replace(
            self,
            start_urls=(
                tuple(start_urls) if start_urls is not None else self.start_urls
            ),
            max_pages=max_pages if max_pages is not None else self.max_pages,
            max_depth=max_depth if max_depth is not None else self.max_depth,
            rate_limit=rate_limit if rate_limit is not None else self.rate_limit,
            respect_robots=(
                respect_robots if respect_robots is not None else self.respect_robots
            ),
            reports=reports,
        )


def _storage_config(value: object, *, base_dir: Path) -> StorageConfig | None:
    if value is None:
        return None
    raw = _mapping(value, "storage")
    storage_format = _storage_format(raw.get("format"))
    allowed_options = {
        "json": {"format", "path", "json_lines", "indent", "encoding"},
        "csv": {"format", "path", "encoding"},
        "sqlite": {"format", "path", "batch_size"},
    }
    _reject_unknown(raw, allowed_options[storage_format], "storage")
    path = _path(raw.get("path"), "storage.path", base_dir=base_dir)
    json_lines = _boolean(raw.get("json_lines"), "storage.json_lines", True)
    indent = _optional_integer(raw.get("indent"), "storage.indent")
    encoding = _string(raw.get("encoding"), "storage.encoding", "utf-8")
    batch_size = _integer(raw.get("batch_size"), "storage.batch_size", 100)
    if storage_format == "json" and json_lines and indent is not None:
        raise ValueError("storage.indent requires storage.json_lines=false")
    if indent is not None and indent < 0:
        raise ValueError("storage.indent must be zero or greater")
    if batch_size <= 0:
        raise ValueError("storage.batch_size must be greater than zero")
    return StorageConfig(
        format=storage_format,
        path=path,
        json_lines=json_lines,
        indent=indent,
        encoding=encoding,
        batch_size=batch_size,
    )


def _logging_config(value: object, *, base_dir: Path) -> LoggingConfig:
    raw = _mapping(value, "logging")
    _reject_unknown(raw, {"level", "file", "max_bytes", "backup_count"}, "logging")
    level_value = _string(raw.get("level"), "logging.level", "INFO").upper()
    if level_value not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("logging.level must be DEBUG, INFO, WARNING, or ERROR")
    return LoggingConfig(
        level=cast(LogLevel, level_value),
        file=(
            _path(raw["file"], "logging.file", base_dir=base_dir)
            if "file" in raw
            else None
        ),
        max_bytes=_integer(
            raw.get("max_bytes"),
            "logging.max_bytes",
            5_000_000,
        ),
        backup_count=_integer(
            raw.get("backup_count"),
            "logging.backup_count",
            3,
        ),
    )


def _report_config(value: object, *, base_dir: Path) -> ReportConfig:
    raw = _mapping(value, "reports")
    _reject_unknown(raw, {"json", "html"}, "reports")
    return ReportConfig(
        json=(
            _path(raw["json"], "reports.json", base_dir=base_dir)
            if "json" in raw
            else None
        ),
        html=(
            _path(raw["html"], "reports.html", base_dir=base_dir)
            if "html" in raw
            else None
        ),
    )


def _mapping(value: object, name: str) -> Mapping[object, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _storage_format(value: object) -> StorageFormat:
    if value == "json":
        return "json"
    if value == "csv":
        return "csv"
    if value == "sqlite":
        return "sqlite"
    raise ValueError("storage.format must be json, csv, or sqlite")


def _reject_unknown(
    raw: Mapping[object, object],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ValueError(f"unknown {name} option(s): {', '.join(unknown)}")


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be an array of strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return tuple(item.strip() for item in value if isinstance(item, str))


def _string(value: object, name: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name, 0)


def _number(value: object, name: str, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _number(value, name, 0.0)


def _boolean(value: object, name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _path(value: object, name: str, *, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _validate_http_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme.casefold() in {"http", "https"}
            and canonical_hostname(url) is not None
            and not any(character.isspace() for character in url)
            and "\\" not in parsed.netloc
        )
        _port = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"invalid configured URL: {url!r}")
