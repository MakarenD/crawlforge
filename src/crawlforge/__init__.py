"""CrawlForge package."""

from crawlforge.concurrency import SemaphoreManager, SemaphoreStats
from crawlforge.crawler import AsyncCrawler, CrawlStats
from crawlforge.errors import NetworkError, ParseError, PermanentError, TransientError
from crawlforge.parser import HTMLParser, ParsedPage
from crawlforge.politeness import RateLimiter, RobotsData, RobotsParser
from crawlforge.queue import CrawlerQueue, QueueStats
from crawlforge.retry import ErrorRecord, RetryStats, RetryStrategy
from crawlforge.storage import (
    CrawlData,
    CSVStorage,
    DataStorage,
    JSONStorage,
    SQLiteStorage,
)

__version__ = "0.1.0"

__all__ = [
    "AsyncCrawler",
    "CSVStorage",
    "CrawlerQueue",
    "CrawlData",
    "CrawlStats",
    "DataStorage",
    "ErrorRecord",
    "HTMLParser",
    "JSONStorage",
    "NetworkError",
    "ParseError",
    "ParsedPage",
    "PermanentError",
    "QueueStats",
    "RateLimiter",
    "RetryStats",
    "RetryStrategy",
    "RobotsData",
    "RobotsParser",
    "SemaphoreManager",
    "SemaphoreStats",
    "SQLiteStorage",
    "TransientError",
    "__version__",
]
