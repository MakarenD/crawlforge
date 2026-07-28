"""CrawlForge package."""

from crawlforge.advanced import AdvancedCrawler
from crawlforge.concurrency import SemaphoreManager, SemaphoreStats
from crawlforge.config import CrawlerConfig, LoggingConfig, ReportConfig, StorageConfig
from crawlforge.crawler import AsyncCrawler, CrawlStats
from crawlforge.errors import NetworkError, ParseError, PermanentError, TransientError
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
    "CrawlerQueue",
    "CrawlerConfig",
    "CrawlerStats",
    "CrawlerStatsSnapshot",
    "CrawlData",
    "CrawlStats",
    "DataStorage",
    "DomainCount",
    "ErrorRecord",
    "HTMLParser",
    "JSONStorage",
    "LoggingConfig",
    "NetworkError",
    "ParseError",
    "ParsedPage",
    "PermanentError",
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
    "SitemapParser",
    "StorageConfig",
    "TransientError",
    "__version__",
]
