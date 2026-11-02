import asyncio
import socket

import pytest

from web_page_fetch.errors import ErrorCode as E, FetchError
from web_page_fetch.security import validate_url, validate_public_ip, resolve_public_target
from web_page_fetch import security


@pytest.mark.parametrize("value", ["", "file:///a", "ftp://example.com", "https:///", "https://u:p@example.com",
    "https://example.com:80", "http://example.com:443", "https://example.com:", "https://example.com..",
    "https://a..com", "https://example.com/\n", "https://example.com/\\a", "https://[fe80::1%25x]/"])
def test_invalid_url(value):
    with pytest.raises(FetchError) as err:
        validate_url(value)
    assert err.value.code == E.INVALID_URL


@pytest.mark.parametrize("value", ["http://localhost", "https://LOCALHOST.", "http://a.localhost."])
def test_localhost(value):
    with pytest.raises(FetchError) as err:
        validate_url(value)
    assert err.value.code == E.TARGET_NOT_ALLOWED


@pytest.mark.parametrize("value,host", [("https://EXAMPLE.com./p?q=x#f", "example.com"),
    ("https://例子.测试./", "xn--fsqu00a.xn--0zwm56d"), ("http://example.com。/", "example.com")])
def test_canonical_identity(value, host):
    result = validate_url(value)
    assert result.host == result.url.raw_host.decode("ascii") == host
    assert not result.url.fragment


@pytest.mark.parametrize("value", ["127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254",
    "100.100.100.200", "168.63.129.16", "0.0.0.0", "224.0.0.1", "255.255.255.255", "192.0.2.1",
    "::", "::1", "fc00::1", "fe80::1", "ff02::1", "::ffff:8.8.8.8", "::ffff:127.0.0.1",
    "64:ff9b::808:808", "2002:808:808::1", "2001::1", "2001:db8::1", "fe80::1%eth0"])
def test_non_public(value):
    with pytest.raises(FetchError):
        validate_public_ip(value)


@pytest.mark.parametrize("value", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_public(value):
    assert validate_public_ip(value) == value


def test_dns_checks_all_results_and_deduplicates():
    async def run():
        seen = []
        async def resolver(host, port):
            seen.append((host, port))
            return ["8.8.8.8", "1.1.1.1", "8.8.8.8"]
        result = await resolve_public_target(validate_url("https://EXAMPLE.com."), resolver)
        assert result.ips == ("8.8.8.8", "1.1.1.1")
        assert seen == [("example.com", 443)]
        async def mixed(*_):
            return ["8.8.8.8", "10.1.2.3"]
        with pytest.raises(FetchError):
            await resolve_public_target(validate_url("https://example.com"), mixed)
    asyncio.run(run())


@pytest.mark.parametrize("mode,code", [("empty", E.DNS_ERROR), ("error", E.DNS_ERROR), ("timeout", E.TIMEOUT)])
def test_dns_failures(monkeypatch, mode, code):
    monkeypatch.setattr(security, "DNS_TIMEOUT", 0.01)
    async def resolver(*_):
        if mode == "error":
            raise socket.gaierror("sensitive 10.0.0.1 token")
        if mode == "timeout":
            await asyncio.sleep(1)
        return []
    with pytest.raises(FetchError) as err:
        asyncio.run(resolve_public_target(validate_url("https://example.com"), resolver))
    assert err.value.code == code
    assert "sensitive" not in str(err.value)


@pytest.mark.parametrize("value", ["http://127.0.0.1", "http://[::1]", "http://2130706433", "http://127.1"])
def test_private_literal_and_legacy_numeric_never_connect(value):
    async def resolver(*_):
        return ["127.0.0.1"]
    with pytest.raises(FetchError):
        asyncio.run(resolve_public_target(validate_url(value), resolver))


def test_system_resolver_arguments(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        async def fake(host, port, **kwargs):
            assert (host, port) == ("example.com", 443)
            assert kwargs == dict(family=socket.AF_UNSPEC, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))]
        monkeypatch.setattr(loop, "getaddrinfo", fake)
        assert await security.system_resolver("example.com", 443) == ["8.8.8.8"]
    asyncio.run(run())
