"""CrawlForge package."""

from crawlforge.advanced import AdvancedCrawler
from crawlforge.chunking import ChunkingConfig, TextChunker
from crawlforge.concurrency import SemaphoreManager, SemaphoreStats
from crawlforge.config import CrawlerConfig, LoggingConfig, ReportConfig, StorageConfig
from crawlforge.content import ContentProcessor
from crawlforge.context_engine import ContextEngine, EmptyCrawlError
from crawlforge.context_index import FTS5UnavailableError, SQLiteContextIndex
from crawlforge.context_models import (
    ContextResult,
    DocumentBlock,
    HeuristicTokenEstimator,
    IndexInfo,
    IndexingResult,
    IndexSessionSummary,
    SearchHit,
    SourceDocument,
    SourceReference,
    TextChunk,
    TokenEstimator,
)
from crawlforge.crawler import AsyncCrawler, CrawledPage, CrawlStats, PageHandler
from crawlforge.errors import NetworkError, ParseError, PermanentError, TransientError
from crawlforge.network_policy import URLNetworkPolicy, URLPolicyError
from crawlforge.parser import HTMLParser, ParsedPage
from crawlforge.politeness import RateLimiter, RobotsData, RobotsParser
from crawlforge.queue import CrawlerQueue, QueueStats
from crawlforge.retry import ErrorRecord, RetryStats, RetryStrategy
from crawlforge.sitemap import SitemapParser
from crawlforge.statistics import CrawlerStats, CrawlerStatsSnapshot, DomainCount
from crawlforge.storage import (
    CrawlData,
    CSVStorage,
    DataStorage,
    JSONStorage,
    SQLiteStorage,
)

__version__ = "0.1.0"

__all__ = [
    "AdvancedCrawler",
    "AsyncCrawler",
    "CSVStorage",
    "ChunkingConfig",
    "ContentProcessor",
    "ContextEngine",
    "ContextResult",
    "CrawledPage",
    "CrawlerQueue",
    "CrawlerConfig",
    "CrawlerStats",
    "CrawlerStatsSnapshot",
    "CrawlData",
    "CrawlStats",
    "DataStorage",
    "DocumentBlock",
    "DomainCount",
    "EmptyCrawlError",
    "ErrorRecord",
    "FTS5UnavailableError",
    "HTMLParser",
    "HeuristicTokenEstimator",
    "IndexInfo",
    "IndexSessionSummary",
    "IndexingResult",
    "JSONStorage",
    "LoggingConfig",
    "NetworkError",
    "ParseError",
    "ParsedPage",
    "PermanentError",
    "PageHandler",
    "QueueStats",
    "RateLimiter",
    "RetryStats",
    "RetryStrategy",
    "ReportConfig",
    "RobotsData",
    "RobotsParser",
    "SemaphoreManager",
    "SemaphoreStats",
    "SQLiteStorage",
    "SQLiteContextIndex",
    "SearchHit",
    "SitemapParser",
    "SourceDocument",
    "SourceReference",
    "StorageConfig",
    "TextChunk",
    "TextChunker",
    "TokenEstimator",
    "TransientError",
    "URLNetworkPolicy",
    "URLPolicyError",
    "__version__",
]
