"""High-level crawler integration for configuration, sitemaps, and reports."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import TracebackType

from crawlforge.config import CrawlerConfig
from crawlforge.crawler import AsyncCrawler
from crawlforge.logging_config import configure_logging
from crawlforge.parser import ParsedPage
from crawlforge.statistics import render_html_report


class AdvancedCrawler:
    """Integrate crawling, sitemaps, configuration, storage, and reporting."""

    def __init__(
        self,
        config: CrawlerConfig,
        *,
        crawler: AsyncCrawler | None = None,
    ) -> None:
        """Build the integrated crawler from a validated configuration."""
        self.config = config
        self._crawler = crawler or AsyncCrawler(
            max_concurrent=config.max_concurrent,
            max_concurrent_per_domain=config.max_concurrent_per_domain,
            max_depth=config.max_depth,
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
            total_timeout=config.total_timeout,
            requests_per_second=config.rate_limit,
            rate_limit_per_domain=config.rate_limit_per_domain,
            respect_robots=config.respect_robots,
            min_delay=config.min_delay,
            jitter=config.jitter,
            max_retries=config.max_retries,
            storage=config.storage.create() if config.storage is not None else None,
        )
        self._closed = False
        self.results: dict[str, ParsedPage] = {}

    @classmethod
    def from_config(cls, filename: str | Path) -> AdvancedCrawler:
        """Load JSON configuration and configure its logging destinations."""
        config = CrawlerConfig.from_file(filename)
        configure_logging(config.logging)
        return cls(config)

    @property
    def visited_urls(self) -> set[str]:
        """Return every page URL attempted by the underlying crawler."""
        return self._crawler.visited_urls

    @property
    def failed_urls(self) -> dict[str, str]:
        """Return final page failures keyed by requested URL."""
        return self._crawler.failed_urls

    @property
    def sitemap_failures(self) -> dict[str, str]:
        """Return sitemap sources that could not provide crawl seeds."""
        return self._crawler.sitemap_failures

    async def __aenter__(self) -> AdvancedCrawler:
        """Enter the integrated crawler resource context."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close sitemap, HTTP, and storage resources."""
        await self.close()

    async def crawl(self) -> dict[str, ParsedPage]:
        """Discover sitemap seeds, crawl pages, and write configured reports."""
        if self._closed:
            raise RuntimeError("AdvancedCrawler is closed")
        await asyncio.to_thread(self._prepare_output_directories)
        self.results = await self._crawler.crawl(
            list(self.config.start_urls),
            max_pages=self.config.max_pages,
            same_domain_only=self.config.same_domain_only,
            exclude_patterns=self.config.exclude_patterns,
            include_patterns=self.config.include_patterns,
            sitemap_urls=self.config.sitemap_urls,
        )
        if self.config.reports.json is not None:
            await asyncio.to_thread(
                self.export_to_json,
                self.config.reports.json,
            )
        if self.config.reports.html is not None:
            await asyncio.to_thread(
                self.export_to_html_report,
                self.config.reports.html,
            )
        return dict(self.results)

    def get_stats(self) -> dict[str, object]:
        """Return legacy and advanced statistics in one compatible mapping."""
        return {
            **self._crawler.get_stats(),
            **self._crawler.get_advanced_stats(),
            "sitemap_failures": dict(self.sitemap_failures),
        }

    def export_to_json(self, filename: str | Path) -> None:
        """Export statistics, successful pages, and failures as JSON."""
        payload = {
            "stats": self.get_stats(),
            "failed_urls": self.failed_urls,
            "pages": self.results,
        }
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def export_to_html_report(self, filename: str | Path) -> None:
        """Export a standalone report with charts and result tables."""
        pages = [
            {"url": url, "title": page["title"]} for url, page in self.results.items()
        ]
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_html_report(
                self.get_stats(),
                pages=pages,
                failed_urls=self.failed_urls,
            ),
            encoding="utf-8",
        )

    async def close(self) -> None:
        """Close all owned resources safely and idempotently."""
        close_task = asyncio.create_task(self._close_resources())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await close_task
            except Exception as close_error:
                raise cancelled from close_error
            raise

    async def _close_resources(self) -> None:
        if self._closed:
            return
        await self._crawler.close()
        self._closed = True

    def _prepare_output_directories(self) -> None:
        paths = [
            self.config.storage.path if self.config.storage is not None else None,
            self.config.reports.json,
            self.config.reports.html,
        ]
        for path in paths:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
