"""Security tests for outbound URL and resolved-address policy."""

from __future__ import annotations

import pytest
from yarl import URL

from crawlforge.crawler import AsyncCrawler
from crawlforge.network_policy import URLNetworkPolicy, URLPolicyError


@pytest.mark.parametrize(
    "url",
    (
        "file:///etc/passwd",
        "ftp://example.com/archive",
        "data:text/plain,hello",
        "http://localhost/",
        "http://service.localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://[fe80::1]/",
        "http://[fec0::1]/",
        "http://[ff02::1]/",
        "http://[::]/",
        "http://240.0.0.1/",
        "http://[::ffff:127.0.0.1]/",
    ),
)
def test_default_policy_blocks_non_http_and_non_public_targets(url: str) -> None:
    policy = URLNetworkPolicy()

    with pytest.raises(URLPolicyError):
        policy.validate_url(url)


def test_default_policy_allows_public_http_and_https_targets() -> None:
    policy = URLNetworkPolicy()

    assert policy.validate_url("https://example.com/docs?q=bounded") == "example.com"
    assert policy.validate_url("http://93.184.216.34/") == "93.184.216.34"


def test_private_network_opt_in_allows_loopback_and_private_addresses() -> None:
    policy = URLNetworkPolicy(allow_private_networks=True)

    assert policy.validate_url("http://localhost:8080/") == "localhost"
    assert policy.validate_url("http://127.0.0.1/") == "127.0.0.1"
    assert policy.validate_url("http://[::1]/") == "::1"
    assert policy.validate_url("http://10.0.0.1/") == "10.0.0.1"
    assert policy.validate_url("http://[fec0::1]/") == "fec0::1"


@pytest.mark.parametrize(
    "url", ("http://224.0.0.1/", "http://[ff02::1]/", "http://[::]/")
)
def test_private_network_opt_in_still_blocks_non_routable_targets(url: str) -> None:
    policy = URLNetworkPolicy(allow_private_networks=True)

    with pytest.raises(URLPolicyError, match="non-routable"):
        policy.validate_url(url)


def test_domain_allowlist_accepts_exact_hosts_and_subdomains_only() -> None:
    policy = URLNetworkPolicy(allowed_domains=("example.com",))

    assert policy.validate_url("https://example.com/") == "example.com"
    assert policy.validate_url("https://docs.example.com/") == "docs.example.com"
    with pytest.raises(URLPolicyError, match="allowlist"):
        policy.validate_url("https://evil-example.com/")
    with pytest.raises(URLPolicyError, match="allowlist"):
        policy.validate_url("https://example.com.evil.test/")


def test_domain_allowlist_normalizes_idna_case_and_trailing_dot() -> None:
    policy = URLNetworkPolicy(allowed_domains=("TÄST.DE.",))

    assert policy.validate_url("https://xn--tst-qla.de/docs") == "xn--tst-qla.de"


def test_ip_allowlist_entry_is_exact_and_has_no_subdomains() -> None:
    policy = URLNetworkPolicy(allowed_domains=("93.184.216.34",))

    assert policy.validate_url("https://93.184.216.34/") == "93.184.216.34"
    with pytest.raises(URLPolicyError, match="allowlist"):
        policy.validate_url("https://docs.93.184.216.34/")


def test_policy_rejects_userinfo_and_malformed_authorities() -> None:
    policy = URLNetworkPolicy()

    with pytest.raises(URLPolicyError, match="user information"):
        policy.validate_url("https://user:secret@example.com/")
    with pytest.raises(URLPolicyError, match="malformed"):
        policy.validate_url("https://example.com:99999/")
    with pytest.raises(URLPolicyError):
        policy.validate_url("https://example.com\\@127.0.0.1/")


def test_resolved_address_validation_fails_closed_for_mixed_answers() -> None:
    policy = URLNetworkPolicy()

    with pytest.raises(URLPolicyError, match="private"):
        policy.validate_resolved_addresses(
            "example.com",
            ("93.184.216.34", "127.0.0.1"),
        )
    with pytest.raises(URLPolicyError, match="did not resolve"):
        policy.validate_resolved_addresses("example.com", ())


@pytest.mark.asyncio
async def test_redirect_from_public_url_to_private_address_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    class RedirectResponse:
        status = 302
        headers = {"Location": "http://127.0.0.1/private"}
        url = URL("https://example.com/start")

        async def __aenter__(self) -> RedirectResponse:
            return self

        async def __aexit__(
            self,
            _exc_type: object,
            _exc_value: object,
            _traceback: object,
        ) -> None:
            return None

    class FakeSession:
        closed = False

        def get(self, url: str, **_kwargs: object) -> RedirectResponse:
            requested.append(url)
            return RedirectResponse()

        async def close(self) -> None:
            self.closed = True

    session = FakeSession()
    crawler = AsyncCrawler(
        respect_robots=False,
        requests_per_second=1000,
        network_policy=URLNetworkPolicy(),
    )

    async def fake_session() -> FakeSession:
        return session

    monkeypatch.setattr(crawler, "_get_session", fake_session)
    try:
        result = await crawler.fetch_url("https://example.com/start")
    finally:
        await crawler.close()

    assert result == ""
    assert requested == ["https://example.com/start"]
