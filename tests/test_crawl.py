"""Integration tests for queue-driven crawling."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge import AsyncCrawler


@asynccontextmanager
async def serve(app: web.Application) -> AsyncIterator[TestServer]:
    """Run an aiohttp application on an ephemeral local port."""
    server = TestServer(app)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_crawl_respects_depth_and_never_visits_duplicates() -> None:
    """Duplicate links are fetched once and links beyond max_depth are skipped."""
    hits: dict[str, int] = {}

    async def page(request: web.Request) -> web.Response:
        path = request.path
        hits[path] = hits.get(path, 0) + 1
        if path == "/":
            body = '<a href="/level-one#first">One</a>'
            body += '<a href="/level-one#second">Duplicate</a>'
        else:
            body = '<a href="/level-two">Too deep</a>'
        return web.Response(text=body, content_type="text/html")

    app = web.Application()
    app.router.add_get("/{tail:.*}", page)

    async with serve(app) as server, AsyncCrawler(max_depth=1) as crawler:
        root = str(server.make_url("/"))
        level_one = str(server.make_url("/level-one"))
        results = await crawler.crawl(
            [root, f"{root}#duplicate-start"],
            same_domain_only=True,
        )

    assert set(results) == {root, level_one}
    assert crawler.visited_urls == {root, level_one}
    assert hits == {"/robots.txt": 1, "/": 1, "/level-one": 1}
    assert crawler.get_stats()["queued"] == 0


@pytest.mark.asyncio
async def test_crawl_applies_domain_include_and_exclude_filters() -> None:
    """Discovered links must pass every enabled URL filter."""

    async def root_page(request: web.Request) -> web.Response:
        return web.Response(
            text=(
                '<a href="/allowed">Allowed</a>'
                '<a href="/allowed/skip">Excluded</a>'
                '<a href="/blocked">Not included</a>'
                '<a href="https://external.invalid/allowed">External</a>'
            ),
            content_type="text/html",
        )

    async def child(request: web.Request) -> web.Response:
        return web.Response(text=request.path, content_type="text/html")

    app = web.Application()
    app.router.add_get("/", root_page)
    app.router.add_get("/{tail:.*}", child)

    async with serve(app) as server, AsyncCrawler(max_depth=1) as crawler:
        root = str(server.make_url("/"))
        allowed = str(server.make_url("/allowed"))
        results = await crawler.crawl(
            [root],
            same_domain_only=True,
            include_patterns=[r"/allowed"],
            exclude_patterns=[r"/skip(?:$|\?)"],
        )

    assert set(results) == {root, allowed}
    assert crawler.visited_urls == {root, allowed}


@pytest.mark.asyncio
async def test_crawl_uses_sitemap_as_the_only_url_source() -> None:
    """Sitemap entries enter the normal queue without an explicit page seed."""
    page_hits = 0

    async def sitemap(request: web.Request) -> web.Response:
        origin = f"{request.scheme}://{request.host}"
        return web.Response(
            text=(
                "<urlset>"
                f"<url><loc>{origin}/page#one</loc></url>"
                f"<url><loc>{origin}/page#two</loc></url>"
                f"<url><loc>{origin}/excluded</loc></url>"
                "</urlset>"
            ),
            content_type="application/xml",
        )

    async def page(_request: web.Request) -> web.Response:
        nonlocal page_hits
        page_hits += 1
        return web.Response(text="<title>Sitemap page</title>")

    async def excluded(_request: web.Request) -> web.Response:
        raise AssertionError("excluded sitemap location was fetched")

    app = web.Application()
    app.router.add_get("/sitemap.xml", sitemap)
    app.router.add_get("/page", page)
    app.router.add_get("/excluded", excluded)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
        ) as crawler,
    ):
        root = str(server.make_url("/"))
        pages = await crawler.crawl(
            [],
            sitemap_urls=[f"{root}sitemap.xml"],
            exclude_patterns=["/excluded"],
        )

    assert set(pages) == {f"{root}page"}
    assert page_hits == 1
    assert crawler.get_advanced_stats()["status_codes"] == {"200": 1}


@pytest.mark.asyncio
async def test_integrated_sitemap_fetch_rejects_oversized_chunked_body() -> None:
    """The shared polite transport bounds a sitemap before materializing its body."""

    async def oversized(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={"Content-Type": "application/xml"},
        )
        await response.prepare(request)
        await response.write(b"<urlset>" + b"x" * 128 + b"</urlset>")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/sitemap.xml", oversized)

    async with (
        serve(app) as server,
        AsyncCrawler(
            respect_robots=False,
            requests_per_second=1000,
        ) as crawler,
    ):
        crawler._sitemap_max_document_bytes = 32
        sitemap_url = str(server.make_url("/sitemap.xml"))
        with pytest.raises(ValueError, match="no crawlable URLs"):
            await crawler.crawl([], sitemap_urls=[sitemap_url])

    assert "response exceeds 32 bytes" in crawler.sitemap_failures[sitemap_url]


@pytest.mark.asyncio
async def test_crawl_records_failures_and_live_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """HTTP errors are tracked without aborting the crawl or hiding progress."""

    async def root_page(request: web.Request) -> web.Response:
        return web.Response(
            text='<a href="/missing">Missing</a>',
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/", root_page)

    async with serve(app) as server, AsyncCrawler(max_depth=1) as crawler:
        root = str(server.make_url("/"))
        missing = str(server.make_url("/missing"))
        with caplog.at_level(logging.INFO, logger="crawlforge.crawler"):
            results = await crawler.crawl([root], same_domain_only=True)

    assert set(results) == {root}
    assert crawler.failed_urls[missing].startswith("HTTP 404:")
    assert crawler.get_stats()["processed"] == 1
    assert crawler.get_stats()["failed"] == 1
    assert crawler.get_stats()["pages_per_second"] > 0
    advanced = crawler.get_advanced_stats()
    assert advanced["total_pages"] == 2
    assert advanced["successful"] == 1
    assert advanced["failed"] == 1
    assert advanced["status_codes"] == {"200": 1, "404": 1}
    assert advanced["progress_percent"] == 100.0
    assert "Crawl progress: processed=1 queued=0 errors=1 rate=" in caplog.text
    assert "progress=100.0% eta=unknown active=0" in caplog.text


@pytest.mark.asyncio
async def test_crawl_treats_empty_successful_response_as_processed() -> None:
    """An HTTP 200 empty body is successful even though fetch_url returns empty."""

    async def empty(request: web.Request) -> web.Response:
        return web.Response(text="", content_type="text/html")

    app = web.Application()
    app.router.add_get("/", empty)

    async with serve(app) as server, AsyncCrawler() as crawler:
        root = str(server.make_url("/"))
        results = await crawler.crawl([root])

    assert set(results) == {root}
    assert crawler.failed_urls == {}
    assert crawler.get_stats()["processed"] == 1


@pytest.mark.asyncio
async def test_crawl_isolates_decode_and_parser_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Page-local decode and parser errors do not abort successful siblings."""

    async def page(request: web.Request) -> web.Response:
        if request.path == "/decode-error":
            return web.Response(
                body=b"\xff",
                headers={"Content-Type": "text/html; charset=utf-8"},
            )
        return web.Response(text="<p>ok</p>", content_type="text/html")

    app = web.Application()
    app.router.add_get("/{tail:.*}", page)

    async with serve(app) as server, AsyncCrawler() as crawler:
        decode_error = str(server.make_url("/decode-error"))
        parser_error = str(server.make_url("/parser-error"))
        ok = str(server.make_url("/ok"))
        original_parse = crawler._parser.parse_html

        async def controlled_parse(html: str, url: str) -> object:
            if url == parser_error:
                raise RuntimeError("parser failed")
            return await original_parse(html, url)

        monkeypatch.setattr(crawler._parser, "parse_html", controlled_parse)
        results = await crawler.crawl([decode_error, parser_error, ok])

    assert set(results) == {ok}
    assert crawler.failed_urls[decode_error].startswith("UnicodeDecodeError:")
    assert crawler.failed_urls[parser_error] == "RuntimeError: parser failed"
    assert crawler.get_error_stats()["errors_by_type"] == {"ParseError": 2}
    assert crawler.queue.get_stats()["active"] == 0


@pytest.mark.asyncio
async def test_crawl_resolves_links_against_redirect_destination() -> None:
    """Relative links use the final response URL while result keys stay stable."""
    hits: dict[str, int] = {}

    async def old(request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/dir/page")

    async def redirected(request: web.Request) -> web.Response:
        return web.Response(
            text='<a href="next">Next</a>',
            content_type="text/html",
        )

    async def observed(request: web.Request) -> web.Response:
        hits[request.path] = hits.get(request.path, 0) + 1
        return web.Response(text="done", content_type="text/html")

    app = web.Application()
    app.router.add_get("/old", old)
    app.router.add_get("/dir/page", redirected)
    app.router.add_get("/dir/next", observed)
    app.router.add_get("/next", observed)

    async with serve(app) as server, AsyncCrawler(max_depth=1) as crawler:
        old_url = str(server.make_url("/old"))
        correct_next = str(server.make_url("/dir/next"))
        results = await crawler.crawl([old_url], same_domain_only=True)

    assert set(results) == {old_url, correct_next}
    assert results[old_url]["url"] == old_url
    assert hits == {"/dir/next": 1}


@pytest.mark.asyncio
async def test_domain_filter_applies_to_links_after_cross_host_redirect() -> None:
    """Relative links inherit a redirect host before domain filtering."""
    next_requested = False

    async def old(request: web.Request) -> web.StreamResponse:
        port = request.host.rsplit(":", maxsplit=1)[1]
        raise web.HTTPFound(f"http://localhost:{port}/dir/page")

    async def redirected(request: web.Request) -> web.Response:
        return web.Response(
            text='<a href="next">External next</a>',
            content_type="text/html",
        )

    async def next_page(request: web.Request) -> web.Response:
        nonlocal next_requested
        next_requested = True
        return web.Response(text="unexpected")

    app = web.Application()
    app.router.add_get("/old", old)
    app.router.add_get("/dir/page", redirected)
    app.router.add_get("/dir/next", next_page)

    async with serve(app) as server, AsyncCrawler(max_depth=1) as crawler:
        old_url = str(server.make_url("/old"))
        results = await crawler.crawl([old_url], same_domain_only=True)

    assert set(results) == {old_url}
    assert not next_requested


@pytest.mark.parametrize("same_domain_only", [False, True])
@pytest.mark.asyncio
async def test_crawl_skips_invalid_idna_link_without_aborting_siblings(
    same_domain_only: bool,
) -> None:
    """An invalid hostname discovered on a page is ignored in every filter mode."""
    invalid_host = "a" * 64

    async def root_page(request: web.Request) -> web.Response:
        return web.Response(
            text=(
                f'<a href="https://{invalid_host}.example/bad">Bad</a>'
                '<a href="/ok">OK</a>'
            ),
            content_type="text/html",
        )

    async def ok(request: web.Request) -> web.Response:
        return web.Response(text="ok", content_type="text/html")

    app = web.Application()
    app.router.add_get("/", root_page)
    app.router.add_get("/ok", ok)

    async with serve(app) as server, AsyncCrawler(max_depth=1) as crawler:
        root = str(server.make_url("/"))
        ok_url = str(server.make_url("/ok"))
        results = await crawler.crawl(
            [root],
            same_domain_only=same_domain_only,
        )

    assert set(results) == {root, ok_url}
    assert crawler.failed_urls == {}
    assert crawler.queue.get_stats()["active"] == 0


@pytest.mark.asyncio
async def test_same_domain_filter_uses_transport_idna_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unicode roots admit their punycode form without admitting another host."""
    crawler = AsyncCrawler(max_depth=1)
    start = "https://faß.de/"
    punycode = "https://xn--fa-hia.de/next"
    ascii_host = "https://fass.de/next"

    async def fake_fetch(url: str) -> tuple[str, str | None, str]:
        if url == start:
            return (
                f'<a href="{punycode}">Same</a><a href="{ascii_host}">Different</a>',
                None,
                url,
            )
        return "", None, url

    monkeypatch.setattr(crawler, "_fetch_url_with_error", fake_fetch)
    results = await crawler.crawl([start], same_domain_only=True)

    assert set(results) == {start, punycode}
    await crawler.close()


@pytest.mark.asyncio
async def test_scheduler_does_not_starve_an_independent_domain() -> None:
    """Queued work for another domain starts while the first domain is saturated."""
    first_domain_started = asyncio.Event()
    other_domain_started = asyncio.Event()
    release_first_domain = asyncio.Event()

    async def first_domain(request: web.Request) -> web.Response:
        if request.path == "/first":
            first_domain_started.set()
            await release_first_domain.wait()
        return web.Response(text=request.path, content_type="text/html")

    async def other_domain(request: web.Request) -> web.Response:
        other_domain_started.set()
        return web.Response(text="other", content_type="text/html")

    first_app = web.Application()
    first_app.router.add_get("/{tail:.*}", first_domain)
    other_app = web.Application()
    other_app.router.add_get("/", other_domain)

    async with (
        serve(first_app) as first_server,
        serve(other_app) as other_server,
        AsyncCrawler(
            max_concurrent=3,
            max_concurrent_per_domain=1,
        ) as crawler,
    ):
        first_urls = [
            str(first_server.make_url("/first")),
            str(first_server.make_url("/second")),
            str(first_server.make_url("/third")),
        ]
        other_url = str(other_server.make_url("/")).replace(
            "127.0.0.1",
            "localhost",
        )
        task = asyncio.create_task(crawler.crawl([*first_urls, other_url]))
        await asyncio.wait_for(first_domain_started.wait(), timeout=5)
        await asyncio.wait_for(other_domain_started.wait(), timeout=5)
        release_first_domain.set()
        results = await asyncio.wait_for(task, timeout=5)

    assert set(results) == {*first_urls, other_url}


@pytest.mark.asyncio
async def test_scheduler_refills_capacity_before_slow_request_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed fast task immediately makes room for another queued domain."""
    crawler = AsyncCrawler(max_concurrent=2, max_concurrent_per_domain=1)
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()
    third_started = asyncio.Event()
    slow_url = "https://slow.example/"
    fast_url = "https://fast.example/"
    third_url = "https://third.example/"

    async def fake_fetch(url: str) -> tuple[str, str | None, str]:
        if url == slow_url:
            slow_started.set()
            await release_slow.wait()
        elif url == third_url:
            third_started.set()
        return "<p>ok</p>", None, url

    monkeypatch.setattr(crawler, "_fetch_url_with_error", fake_fetch)
    task = asyncio.create_task(crawler.crawl([slow_url, fast_url, third_url]))
    await asyncio.wait_for(slow_started.wait(), timeout=5)
    await asyncio.wait_for(third_started.wait(), timeout=5)

    assert not task.done()

    release_slow.set()
    results = await asyncio.wait_for(task, timeout=5)
    assert set(results) == {slow_url, fast_url, third_url}
    await crawler.close()


@pytest.mark.asyncio
async def test_live_progress_reports_remaining_active_page_tasks(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Progress emitted after a fast page observes its still-running sibling."""
    crawler = AsyncCrawler(max_concurrent=2)
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()
    progress_observed = asyncio.Event()
    slow_url = "https://slow.example/"
    fast_url = "https://fast.example/"

    async def fake_fetch(url: str) -> tuple[str, str | None, str]:
        if url == slow_url:
            slow_started.set()
            await release_slow.wait()
        return "<p>ok</p>", None, url

    original_log_progress = crawler._log_progress

    def observe_progress() -> None:
        original_log_progress()
        if crawler.get_advanced_stats()["active_tasks"] == 1:
            progress_observed.set()

    monkeypatch.setattr(crawler, "_fetch_url_with_error", fake_fetch)
    monkeypatch.setattr(crawler, "_log_progress", observe_progress)
    with caplog.at_level(logging.INFO, logger="crawlforge.crawler"):
        task = asyncio.create_task(crawler.crawl([slow_url, fast_url]))
        await asyncio.wait_for(slow_started.wait(), timeout=5)
        await asyncio.wait_for(progress_observed.wait(), timeout=5)
        assert "active=1" in caplog.text
        release_slow.set()
        await asyncio.wait_for(task, timeout=5)
    await crawler.close()


@pytest.mark.asyncio
async def test_crawl_stops_at_max_pages_with_pending_queue_entries() -> None:
    """The page cap is exact even when a page discovers many links."""

    async def page(request: web.Request) -> web.Response:
        links = "".join(f'<a href="/{index}">{index}</a>' for index in range(5))
        return web.Response(text=links, content_type="text/html")

    app = web.Application()
    app.router.add_get("/{tail:.*}", page)

    async with serve(app) as server, AsyncCrawler(max_depth=2) as crawler:
        results = await crawler.crawl(
            [str(server.make_url("/"))],
            max_pages=3,
            same_domain_only=True,
        )

    assert len(results) == 3
    assert len(crawler.visited_urls) == 3
    assert crawler.get_stats()["queued"] == 3


@pytest.mark.asyncio
async def test_crawl_cancellation_releases_request_capacity() -> None:
    """Cancelling an active crawl releases semaphore capacity for later work."""
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def blocked(request: web.Request) -> web.Response:
        request_started.set()
        await release_request.wait()
        return web.Response(text="late")

    async def ok(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/blocked", blocked)
    app.router.add_get("/ok", ok)

    async with serve(app) as server, AsyncCrawler(max_concurrent=1) as crawler:
        task = asyncio.create_task(
            crawler.crawl([str(server.make_url("/blocked"))]),
        )
        await asyncio.wait_for(request_started.wait(), timeout=5)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert crawler.get_stats()["active"] == 0
        assert crawler.queue.get_stats()["active"] == 0
        assert crawler.queue.get_stats()["failed"] == 1
        release_request.set()
        result = await asyncio.wait_for(
            crawler.fetch_url(str(server.make_url("/ok"))),
            timeout=5,
        )

    assert result == "ok"


@pytest.mark.parametrize(
    ("start_urls", "max_pages", "message"),
    [
        ([], 1, "start_urls"),
        (["not-a-url"], 1, "invalid start URL"),
        (["https://example.com"], 0, "max_pages"),
    ],
)
@pytest.mark.asyncio
async def test_crawl_rejects_invalid_boundaries(
    start_urls: list[str],
    max_pages: int,
    message: str,
) -> None:
    """Invalid roots and page limits fail before allocating HTTP resources."""
    crawler = AsyncCrawler()

    with pytest.raises(ValueError, match=message):
        await crawler.crawl(start_urls, max_pages=max_pages)

    assert crawler._session is None
    await crawler.close()


@pytest.mark.asyncio
async def test_crawl_rejects_invalid_filter_regex() -> None:
    """Malformed filter expressions fail before starting network work."""
    crawler = AsyncCrawler()

    with pytest.raises(ValueError, match="exclude_patterns"):
        await crawler.crawl(
            ["https://example.com"],
            exclude_patterns=["["],
        )

    assert crawler._session is None
    await crawler.close()


@pytest.mark.parametrize("max_depth", [-1, -10])
def test_crawler_rejects_negative_depth(max_depth: int) -> None:
    """A negative depth cannot describe a valid crawl boundary."""
    with pytest.raises(ValueError, match="max_depth"):
        AsyncCrawler(max_depth=max_depth)


@pytest.mark.asyncio
async def test_closed_crawler_rejects_new_crawl() -> None:
    """A closed crawler fails before resetting or queueing crawl state."""
    crawler = AsyncCrawler()
    await crawler.close()

    with pytest.raises(RuntimeError, match="closed"):
        await crawler.crawl(["https://example.com"])
