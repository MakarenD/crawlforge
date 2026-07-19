"""Asynchronous HTTP client for downloading web pages."""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType

import aiohttp

logger = logging.getLogger(__name__)


class AsyncCrawler:
    """Download web pages concurrently through a pooled HTTP session."""

    def __init__(
        self,
        max_concurrent: int = 10,
        *,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
    ) -> None:
        """Configure concurrency and per-request connection/read timeouts."""
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than zero")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than zero")
        if read_timeout <= 0:
            raise ValueError("read_timeout must be greater than zero")

        self._max_concurrent = max_concurrent
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            connect=connect_timeout,
            sock_read=read_timeout,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._session_lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

    async def __aenter__(self) -> AsyncCrawler:
        """Enter the crawler's asynchronous resource context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the pooled HTTP session when leaving the context."""
        await self.close()

    async def fetch_url(self, url: str) -> str:
        """Download one URL, returning an empty string when a request fails."""
        logger.info("Fetching URL: %s", url)

        try:
            async with self._semaphore:
                session = await self._get_session()
                async with session.get(url) as response:
                    response.raise_for_status()
                    content = await response.text()
                    status = response.status
        except aiohttp.ClientResponseError as error:
            logger.warning(
                "HTTP error for %s: %s (%s)",
                url,
                error.status,
                type(error).__name__,
            )
            return ""
        except TimeoutError as error:
            logger.warning("Timeout for %s (%s)", url, type(error).__name__)
            return ""
        except aiohttp.ClientError as error:
            logger.warning("Network error for %s (%s)", url, type(error).__name__)
            return ""

        logger.info("Fetched URL: %s (HTTP %s)", url, status)
        return content

    async def fetch_urls(self, urls: list[str]) -> dict[str, str]:
        """Download URLs concurrently and map each URL to its response body."""
        requested_urls = list(urls)
        pages = await asyncio.gather(
            *(self.fetch_url(url) for url in requested_urls),
        )
        return dict(zip(requested_urls, pages, strict=True))

    async def close(self) -> None:
        """Close the pooled HTTP session; repeated calls are safe."""
        async with self._session_lock:
            self._closed = True
            session = self._session
            if session is None:
                return
            if session.closed:
                self._session = None
                return

            close_task = asyncio.create_task(session.close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                await close_task
                raise
            finally:
                self._discard_session_if_closed(session)

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._closed:
                raise RuntimeError("AsyncCrawler is closed")
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(limit=self._max_concurrent)
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=self._timeout,
                )
            return self._session

    def _discard_session_if_closed(self, session: aiohttp.ClientSession) -> None:
        if session.closed:
            self._session = None
