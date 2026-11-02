"""Real HTTPX/HTTPcore pipeline, fake only the network backend/stream."""

import asyncio
import socket
import ssl

import httpcore
import pytest
from httpcore._backends.anyio import AnyIOBackend

from web_page_fetch.errors import FetchError
from web_page_fetch.fetcher import WebPageFetcher


class WireStream:
    def __init__(self, response, fail_tls=False):
        self.response = response
        self.fail_tls = fail_tls
        self.written = bytearray()
        self.tls = []
        self.closed = False

    async def read(self, max_bytes, timeout=None):
        data = self.response[:max_bytes]
        self.response = self.response[max_bytes:]
        return data

    async def write(self, buffer, timeout=None):
        self.written.extend(buffer)

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self.tls.append((server_hostname, ssl_context.check_hostname, ssl_context.verify_mode))
        if self.fail_tls:
            # Match AnyIOStream.start_tls: failed handshake closes the stream.
            await self.aclose()
            raise httpcore.ConnectError("certificate canary") from ssl.SSLCertVerificationError("canary")
        return self

    async def aclose(self):
        self.closed = True

    def get_extra_info(self, name):
        return None


def install_network(monkeypatch, responses, fail_tls=False):
    connected, streams, lookups = [], [], []
    async def connect(self, host, port, **kwargs):
        connected.append((host, port))
        stream = WireStream(responses[len(streams)], fail_tls)
        streams.append(stream)
        return stream
    monkeypatch.setattr(AnyIOBackend, "connect_tcp", connect)
    # The controlled resolver below is the ONLY permitted domain resolution.
    # Any fallback to OS resolution in the actual transport fails the test.
    def forbidden_dns(*args, **kwargs):
        lookups.append(args)
        raise AssertionError("unexpected second domain resolution")
    monkeypatch.setattr(socket, "getaddrinfo", forbidden_dns)
    return connected, streams, lookups


OK = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 12\r\n\r\n<p>hello</p>"


@pytest.mark.parametrize("url,identity,ip,port", [
    ("https://EXAMPLE.com./", "example.com", "8.8.8.8", 443),
    ("https://例子.测试./", "xn--fsqu00a.xn--0zwm56d", "8.8.8.8", 443),
    ("http://example.com/", "example.com", "8.8.8.8", 80),
    ("https://example.com/", "example.com", "2606:4700:4700::1111", 443),
    ("https://8.8.8.8/", "8.8.8.8", "8.8.8.8", 443),
    ("https://[2606:4700:4700::1111]/", "2606:4700:4700::1111", "2606:4700:4700::1111", 443),
])
def test_tcp_wire_host_tls_and_no_second_dns(monkeypatch, url, identity, ip, port):
    connections, streams, os_dns = install_network(monkeypatch, [OK])
    dns = []
    async def resolver(host, port):
        dns.append((host, port))
        # If the original domain is resolved a second time it becomes private.
        return [ip] if len(dns) == 1 else ["127.0.0.1"]
    result = asyncio.run(WebPageFetcher(resolver).fetch(url))
    assert result.content == "hello"
    assert connections == [(ip, port)]
    assert len(dns) == (0 if identity == ip else 1)
    assert os_dns == []
    host = f"[{identity}]" if ":" in identity else identity
    assert f"Host: {host}\r\n".encode() in streams[0].written
    if port == 443:
        assert streams[0].tls == [(identity, True, ssl.CERT_REQUIRED)]
    else:
        assert streams[0].tls == []
    assert all(s.closed for s in streams)


def test_certificate_failure_is_not_retried_or_downgraded(monkeypatch):
    connected, streams, _ = install_network(monkeypatch, [OK], fail_tls=True)
    async def resolver(*_):
        return ["8.8.8.8"]
    with pytest.raises(FetchError, match="NETWORK_ERROR"):
        asyncio.run(WebPageFetcher(resolver).fetch("https://example.com/"))
    assert connected == [("8.8.8.8", 443)]
    assert streams[0].tls == [("example.com", True, ssl.CERT_REQUIRED)]
    assert not streams[0].written
    assert streams[0].closed


def test_cross_domain_same_ip_has_separate_tls_and_rechecks_dns(monkeypatch):
    redirect = b"HTTP/1.1 302 Found\r\nLocation: https://SECOND.example./p\r\nContent-Length: 0\r\n\r\n"
    connected, streams, _ = install_network(monkeypatch, [redirect, OK])
    dns = []
    async def resolver(host, port):
        dns.append((host, port))
        return ["8.8.8.8"]
    result = asyncio.run(WebPageFetcher(resolver).fetch("https://FIRST.example./"))
    assert result.url == "https://second.example/p"
    assert dns == [("first.example", 443), ("second.example", 443)]
    assert connected == [("8.8.8.8", 443)] * 2
    for stream, domain in zip(streams, ["first.example", "second.example"]):
        assert stream.tls == [(domain, True, ssl.CERT_REQUIRED)]
        assert f"Host: {domain}\r\n".encode() in stream.written
        assert stream.closed


def test_environment_proxy_ignored(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9999")
    connected, _, _ = install_network(monkeypatch, [OK])
    async def resolver(*_):
        return ["8.8.8.8"]
    assert asyncio.run(WebPageFetcher(resolver).fetch("https://example.com/")).content == "hello"
    assert connected == [("8.8.8.8", 443)]


def test_same_domain_redirect_rebinding_is_rejected(monkeypatch):
    redirect = b"HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 0\r\n\r\n"
    connected, streams, os_dns = install_network(monkeypatch, [redirect])
    dns = []
    async def resolver(host, port):
        dns.append((host, port))
        return ["8.8.8.8"] if len(dns) == 1 else ["10.0.0.1"]
    with pytest.raises(FetchError, match="TARGET_NOT_ALLOWED"):
        asyncio.run(WebPageFetcher(resolver).fetch("https://example.com/"))
    assert len(dns) == 2 and connected == [("8.8.8.8", 443)]
    assert streams[0].closed and os_dns == []
