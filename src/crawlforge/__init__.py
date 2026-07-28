"""CrawlForge package."""

from crawlforge.concurrency import SemaphoreManager, SemaphoreStats
from crawlforge.crawler import AsyncCrawler, CrawlStats
from crawlforge.parser import HTMLParser, ParsedPage
from crawlforge.politeness import RateLimiter, RobotsData, RobotsParser
from crawlforge.queue import CrawlerQueue, QueueStats

__version__ = "0.1.0"

__all__ = [
    "AsyncCrawler",
    "CrawlerQueue",
    "CrawlStats",
    "HTMLParser",
    "ParsedPage",
    "QueueStats",
    "RateLimiter",
    "RobotsData",
    "RobotsParser",
    "SemaphoreManager",
    "SemaphoreStats",
    "__version__",
]
