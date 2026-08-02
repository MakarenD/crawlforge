"""Tests for package metadata and public exports."""

import json
import subprocess
import sys

from crawlforge import (
    AdvancedCrawler,
    ChunkingConfig,
    ContentProcessor,
    ContextEngine,
    ContextResult,
    CrawledPage,
    CrawlerConfig,
    CrawlerQueue,
    CrawlerStats,
    CSVStorage,
    DataStorage,
    ErrorRecord,
    FTS5UnavailableError,
    HeuristicTokenEstimator,
    HTMLParser,
    HybridContextResult,
    HybridRetriever,
    HybridSearchConfig,
    HybridSearchHit,
    IndexInfo,
    IndexingResult,
    IndexSessionSummary,
    JSONStorage,
    NetworkError,
    ParseError,
    PermanentError,
    RankFusionStrategy,
    RateLimiter,
    ReciprocalRankFusion,
    RetryStrategy,
    RobotsParser,
    SearchHit,
    SemaphoreManager,
    SitemapParser,
    SourceDocument,
    SQLiteContextIndex,
    SQLiteStorage,
    TextChunk,
    TextChunker,
    TransientError,
    URLNetworkPolicy,
    URLPolicyError,
    __version__,
)


def test_package_version_is_exposed() -> None:
    """The package exposes a non-empty version string."""
    assert __version__ == "0.1.0"


def test_html_parser_is_exposed() -> None:
    """The package root exposes the public HTML parser."""
    assert HTMLParser.__name__ == "HTMLParser"


def test_crawl_control_types_are_exposed() -> None:
    """The package root exposes queue and concurrency controls."""
    assert CrawlerQueue.__name__ == "CrawlerQueue"
    assert SemaphoreManager.__name__ == "SemaphoreManager"
    assert RateLimiter.__name__ == "RateLimiter"
    assert RobotsParser.__name__ == "RobotsParser"


def test_retry_and_error_types_are_exposed() -> None:
    """The package root exposes retry orchestration and error classification."""
    assert RetryStrategy.__name__ == "RetryStrategy"
    assert ErrorRecord.__name__ == "ErrorRecord"
    assert TransientError.__name__ == "TransientError"
    assert PermanentError.__name__ == "PermanentError"
    assert NetworkError.__name__ == "NetworkError"
    assert ParseError.__name__ == "ParseError"


def test_storage_types_are_exposed() -> None:
    """The package root exposes every asynchronous storage backend."""
    assert DataStorage.__name__ == "DataStorage"
    assert JSONStorage.__name__ == "JSONStorage"
    assert CSVStorage.__name__ == "CSVStorage"
    assert SQLiteStorage.__name__ == "SQLiteStorage"


def test_advanced_crawler_types_are_exposed() -> None:
    """The package root exposes configuration, sitemap, stats, and integration."""
    assert AdvancedCrawler.__name__ == "AdvancedCrawler"
    assert CrawlerConfig.__name__ == "CrawlerConfig"
    assert CrawlerStats.__name__ == "CrawlerStats"
    assert SitemapParser.__name__ == "SitemapParser"


def test_web_context_types_are_exposed() -> None:
    """The package root exposes the stable content and retrieval services."""
    assert CrawledPage.__name__ == "CrawledPage"
    assert SourceDocument.__name__ == "SourceDocument"
    assert TextChunk.__name__ == "TextChunk"
    assert SearchHit.__name__ == "SearchHit"
    assert ContextResult.__name__ == "ContextResult"
    assert IndexingResult.__name__ == "IndexingResult"
    assert IndexInfo.__name__ == "IndexInfo"
    assert IndexSessionSummary.__name__ == "IndexSessionSummary"
    assert HeuristicTokenEstimator.__name__ == "HeuristicTokenEstimator"
    assert ContentProcessor.__name__ == "ContentProcessor"
    assert ChunkingConfig.__name__ == "ChunkingConfig"
    assert TextChunker.__name__ == "TextChunker"
    assert SQLiteContextIndex.__name__ == "SQLiteContextIndex"
    assert FTS5UnavailableError.__name__ == "FTS5UnavailableError"
    assert ContextEngine.__name__ == "ContextEngine"
    assert URLNetworkPolicy.__name__ == "URLNetworkPolicy"
    assert URLPolicyError.__name__ == "URLPolicyError"


def test_hybrid_retrieval_types_are_exposed() -> None:
    """The package root exposes fusion without importing the ML runtime."""
    assert HybridRetriever.__name__ == "HybridRetriever"
    assert ReciprocalRankFusion.__name__ == "ReciprocalRankFusion"
    assert HybridSearchConfig.__name__ == "HybridSearchConfig"
    assert HybridSearchHit.__name__ == "HybridSearchHit"
    assert HybridContextResult.__name__ == "HybridContextResult"
    assert RankFusionStrategy.__name__ == "RankFusionStrategy"


def test_base_package_import_does_not_load_optional_ml_runtime() -> None:
    """Importing crawlforge remains lightweight without semantic inference."""
    script = (
        "import json, sys; import crawlforge; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name == 'torch' or name.startswith('sentence_transformers'))))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == []
