"""Tests for asynchronous sitemap parsing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from crawlforge.sitemap import SitemapParser


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
async def test_urlset_namespace_returns_unique_urls_in_order() -> None:
    """A namespaced urlset preserves first-seen document order."""

    async def fetcher(_url: str) -> str:
        return """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/one</loc></url>
          <url><loc>https://example.com/two?q=1</loc></url>
          <url><loc>https://example.com/one</loc></url>
        </urlset>
        """

    async with SitemapParser(fetcher) as parser:
        urls = await parser.fetch_sitemap("https://example.com/sitemap.xml")

    assert urls == [
        "https://example.com/one",
        "https://example.com/two?q=1",
    ]


@pytest.mark.asyncio
async def test_recursive_index_deduplicates_urls_and_breaks_cycles() -> None:
    """Nested indexes are traversed depth-first without revisiting a cycle."""
    documents = {
        "https://example.com/root.xml": """
            <sitemapindex>
              <sitemap><loc>https://example.com/first.xml</loc></sitemap>
              <sitemap><loc>https://example.com/nested.xml</loc></sitemap>
            </sitemapindex>
        """,
        "https://example.com/first.xml": """
            <urlset>
              <url><loc>https://example.com/a</loc></url>
              <url><loc>https://example.com/shared</loc></url>
            </urlset>
        """,
        "https://example.com/nested.xml": """
            <sitemapindex>
              <sitemap><loc>https://example.com/root.xml</loc></sitemap>
              <sitemap><loc>https://example.com/second.xml</loc></sitemap>
            </sitemapindex>
        """,
        "https://example.com/second.xml": """
            <urlset>
              <url><loc>https://example.com/shared</loc></url>
              <url><loc>https://example.com/b</loc></url>
            </urlset>
        """,
    }
    fetched: list[str] = []

    async def fetcher(url: str) -> str:
        fetched.append(url)
        return documents[url]

    parser = SitemapParser(fetcher)
    urls = await parser.fetch_sitemap("https://example.com/root.xml")

    assert urls == [
        "https://example.com/a",
        "https://example.com/shared",
        "https://example.com/b",
    ]
    assert fetched == [
        "https://example.com/root.xml",
        "https://example.com/first.xml",
        "https://example.com/nested.xml",
        "https://example.com/second.xml",
    ]


@pytest.mark.asyncio
async def test_default_fetcher_reads_from_local_aiohttp_server() -> None:
    """The owned aiohttp transport fetches a successful sitemap."""

    async def sitemap(_request: web.Request) -> web.Response:
        return web.Response(
            text="<urlset><url><loc>https://example.com/page</loc></url></urlset>",
            content_type="application/xml",
        )

    app = web.Application()
    app.router.add_get("/sitemap.xml", sitemap)

    async with serve(app) as server, SitemapParser() as parser:
        urls = await parser.fetch_sitemap(str(server.make_url("/sitemap.xml")))

    assert urls == ["https://example.com/page"]


@pytest.mark.asyncio
async def test_invalid_xml_is_reported_with_source_url() -> None:
    """Malformed XML produces a clear validation error."""

    async def fetcher(_url: str) -> str:
        return "<urlset>"

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="invalid sitemap XML.*sitemap.xml"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_empty_fetcher_payload_is_rejected_clearly() -> None:
    """An empty injected response is reported before XML parsing."""

    async def fetcher(_url: str) -> bytes:
        return b""

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="empty sitemap document.*sitemap.xml"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_decoded_string_ignores_stale_xml_encoding_declaration() -> None:
    """A decoded string is not reinterpreted using its original byte encoding."""

    async def fetcher(_url: str) -> str:
        return (
            '<?xml version="1.0" encoding="iso-8859-1"?>'
            "<urlset><url><loc>https://example.com/café</loc></url></urlset>"
        )

    parser = SitemapParser(fetcher)

    assert await parser.fetch_sitemap("https://example.com/sitemap.xml") == [
        "https://example.com/café"
    ]


@pytest.mark.asyncio
async def test_unknown_xml_byte_encoding_is_reported_clearly() -> None:
    """An unavailable declared codec is normalized to a source-aware error."""

    async def fetcher(_url: str) -> bytes:
        return b'<?xml version="1.0" encoding="not-a-codec"?><urlset></urlset>'

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="invalid sitemap XML.*sitemap.xml"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document",
    [
        "<?xml potato?><urlset />",
        '<?XML version="1.0"?><urlset />',
        ' <?xml version="1.0"?><urlset />',
    ],
)
async def test_malformed_string_xml_declaration_is_rejected(document: str) -> None:
    """Normalizing a decoded document does not hide invalid declarations."""

    async def fetcher(_url: str) -> str:
        return document

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="invalid sitemap XML.*sitemap.xml"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_invalid_unicode_string_is_reported_with_source_url() -> None:
    """Unencodable Unicode is normalized before the byte-size check."""

    async def fetcher(_url: str) -> str:
        return "<urlset>\ud800</urlset>"

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="invalid sitemap XML.*sitemap.xml") as caught:
        await parser.fetch_sitemap("https://example.com/sitemap.xml")

    assert isinstance(caught.value.__cause__, UnicodeEncodeError)


@pytest.mark.asyncio
async def test_invalid_root_is_rejected() -> None:
    """Only urlset and sitemapindex documents are accepted."""

    async def fetcher(_url: str) -> str:
        return "<feed />"

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="invalid sitemap root"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_document_type_declaration_is_rejected_before_xml_expansion() -> None:
    """Sitemaps cannot use DTD entities to amplify a bounded document."""

    async def fetcher(_url: str) -> str:
        return (
            "<!DOCTYPE urlset [<!ENTITY repeated 'value'>]>"
            "<urlset><url><loc>https://example.com/&repeated;</loc></url></urlset>"
        )

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="document type declarations"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_url",
    [
        "/relative.xml",
        "ftp://example.com/sitemap.xml",
        "https:///missing-host.xml",
        "https://example.com:invalid/sitemap.xml",
    ],
)
async def test_invalid_root_sitemap_url_is_rejected(invalid_url: str) -> None:
    """A root sitemap location must be an absolute HTTP or HTTPS URL."""

    async def fetcher(_url: str) -> str:
        raise AssertionError("invalid URLs must not be fetched")

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="absolute HTTP or HTTPS"):
        await parser.fetch_sitemap(invalid_url)


@pytest.mark.asyncio
async def test_invalid_urlset_location_is_rejected() -> None:
    """Page locations cannot use relative or non-HTTP URLs."""

    async def fetcher(_url: str) -> str:
        return "<urlset><url><loc>/relative</loc></url></urlset>"

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="sitemap entry URL"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_urlset_location_with_whitespace_is_rejected() -> None:
    """Malformed absolute-looking locations do not reach the crawl queue."""

    async def fetcher(_url: str) -> str:
        return "<urlset><url><loc>https://example.com/a path</loc></url></urlset>"

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="sitemap entry URL"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_invalid_nested_sitemap_location_is_rejected() -> None:
    """Nested sitemap locations must also be absolute HTTP URLs."""

    async def fetcher(_url: str) -> str:
        return (
            "<sitemapindex><sitemap><loc>file:///tmp/map.xml</loc></sitemap>"
            "</sitemapindex>"
        )

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="nested sitemap URL"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_missing_location_is_rejected() -> None:
    """Each sitemap entry requires a non-empty loc child."""

    async def fetcher(_url: str) -> str:
        return "<urlset><url /></urlset>"

    parser = SitemapParser(fetcher)
    with pytest.raises(ValueError, match="missing loc"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_max_depth_stops_nested_fetch_before_network_call() -> None:
    """Depth overflow is detected before its sitemap is fetched."""
    documents = {
        "https://example.com/root.xml": (
            "<sitemapindex><sitemap>"
            "<loc>https://example.com/child.xml</loc>"
            "</sitemap></sitemapindex>"
        ),
    }
    fetched: list[str] = []

    async def fetcher(url: str) -> str:
        fetched.append(url)
        return documents[url]

    parser = SitemapParser(fetcher, max_depth=0)
    with pytest.raises(ValueError, match="maximum sitemap depth"):
        await parser.fetch_sitemap("https://example.com/root.xml")

    assert fetched == ["https://example.com/root.xml"]


@pytest.mark.asyncio
async def test_max_sitemaps_counts_unique_documents() -> None:
    """The sitemap-count limit stops an unseen nested document."""

    async def fetcher(url: str) -> str:
        if url.endswith("root.xml"):
            return (
                "<sitemapindex>"
                "<sitemap><loc>https://example.com/one.xml</loc></sitemap>"
                "<sitemap><loc>https://example.com/two.xml</loc></sitemap>"
                "</sitemapindex>"
            )
        return "<urlset />"

    parser = SitemapParser(fetcher, max_sitemaps=2)
    with pytest.raises(ValueError, match="maximum sitemap count"):
        await parser.fetch_sitemap("https://example.com/root.xml")


@pytest.mark.asyncio
async def test_max_urls_counts_unique_page_locations() -> None:
    """The URL limit rejects the first unique entry beyond the boundary."""

    async def fetcher(_url: str) -> str:
        return """
        <urlset>
          <url><loc>https://example.com/one</loc></url>
          <url><loc>https://example.com/one</loc></url>
          <url><loc>https://example.com/two</loc></url>
        </urlset>
        """

    parser = SitemapParser(fetcher, max_urls=1)
    with pytest.raises(ValueError, match="maximum URL count"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_max_document_bytes_applies_to_injected_fetcher() -> None:
    """Injected content is measured in encoded bytes before XML parsing."""

    async def fetcher(_url: str) -> str:
        return "<urlset />"

    parser = SitemapParser(fetcher, max_document_bytes=4)
    with pytest.raises(ValueError, match="max_document_bytes"):
        await parser.fetch_sitemap("https://example.com/sitemap.xml")


@pytest.mark.asyncio
async def test_default_fetcher_rejects_http_errors() -> None:
    """Non-success HTTP responses produce a clear runtime error."""
    app = web.Application()

    async with serve(app) as server, SitemapParser() as parser:
        url = str(server.make_url("/missing.xml"))
        with pytest.raises(RuntimeError, match="HTTP 404"):
            await parser.fetch_sitemap(url)


@pytest.mark.asyncio
async def test_default_fetcher_enforces_document_size_before_parsing() -> None:
    """An oversized HTTP body is rejected using the configured byte limit."""

    async def sitemap(_request: web.Request) -> web.Response:
        return web.Response(body=b"<urlset />", content_type="application/xml")

    app = web.Application()
    app.router.add_get("/sitemap.xml", sitemap)

    async with serve(app) as server, SitemapParser(max_document_bytes=4) as parser:
        with pytest.raises(ValueError, match="max_document_bytes"):
            await parser.fetch_sitemap(str(server.make_url("/sitemap.xml")))


@pytest.mark.asyncio
async def test_cancellation_propagates_from_injected_fetcher() -> None:
    """Cancelling a fetch is never converted into a parser error."""
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def fetcher(_url: str) -> str:
        started.set()
        await blocked.wait()
        return "<urlset />"

    parser = SitemapParser(fetcher)
    task = asyncio.create_task(parser.fetch_sitemap("https://example.com/sitemap.xml"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_close_releases_owned_session_and_prevents_reuse() -> None:
    """Closing releases the owned session and makes the parser terminal."""

    async def sitemap(_request: web.Request) -> web.Response:
        return web.Response(text="<urlset />", content_type="application/xml")

    app = web.Application()
    app.router.add_get("/sitemap.xml", sitemap)

    async with serve(app) as server:
        parser = SitemapParser()
        url = str(server.make_url("/sitemap.xml"))
        await parser.fetch_sitemap(url)
        session = parser._session

        assert session is not None
        assert not session.closed

        await parser.close()
        await parser.close()

    with pytest.raises(RuntimeError, match="closed"):
        await parser.fetch_sitemap(url)

    assert session.closed
