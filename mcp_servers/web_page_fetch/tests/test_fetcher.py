import asyncio

import httpx
import pytest

from web_page_fetch import fetcher
from web_page_fetch.errors import ErrorCode as E, FetchError
from web_page_fetch.fetcher import WebPageFetcher, MAX_BYTES


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks=(), error=None, delay=0):
        self.chunks = chunks
        self.error = error
        self.delay = delay
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


class Transport(httpx.MockTransport):
    def __init__(self, handler):
        super().__init__(handler)
        self.closed = False

    async def aclose(self):
        self.closed = True


async def public_dns(*_):
    return ["8.8.8.8"]


def service(handler, resolver=public_dns):
    transports = []
    def factory():
        result = Transport(handler)
        transports.append(result)
        return result
    return WebPageFetcher(resolver, factory), transports


def response(stream=None, status=200, headers=None):
    return httpx.Response(status, headers={"Content-Type": "text/html"} | (headers or {}),
                          stream=stream or Stream([b"<title>T</title><article>hello</article>"]))


def test_success_and_closed_and_request_limits():
    body = Stream([b"<title>T</title><article>hello</article>"])
    def handler(request):
        assert request.url.host == "8.8.8.8"
        assert request.headers["host"] == "example.com"
        assert request.headers["accept-encoding"] == "identity"
        assert request.extensions["sni_hostname"] == "example.com"
        assert request.extensions["timeout"] == dict(connect=5.0, read=10.0, write=5.0, pool=5.0)
        return response(body)
    fetch, transports = service(handler)
    result = asyncio.run(fetch.fetch("https://EXAMPLE.com./p?q=x#f"))
    assert result.model_dump() == dict(title="T", content="hello", url="https://example.com/p?q=x",
                                       truncated=False, content_type="text/html")
    assert body.closed and all(t.closed for t in transports)


@pytest.mark.parametrize("content_type", ["application/json", "application/xhtml+xml", "image/png", ""])
def test_non_html_rejected_before_read(content_type):
    body = Stream([b"canary"])
    fetch, _ = service(lambda r: response(body, headers={"Content-Type": content_type}))
    with pytest.raises(FetchError, match="UNSUPPORTED_CONTENT_TYPE"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert body.closed and body.reads == 0


@pytest.mark.parametrize("encoding", ["gzip", "br", "identity, gzip", "", "identity, identity"])
def test_encoding_before_read_and_length(encoding):
    body = Stream([b"compressed"])
    fetch, _ = service(lambda r: response(body, headers={"Content-Encoding": encoding, "Content-Length": str(MAX_BYTES + 1)}))
    with pytest.raises(FetchError, match="UNSUPPORTED_CONTENT_ENCODING"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert body.closed and body.reads == 0


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "1"}, {"Transfer-Encoding": "chunked"}])
def test_raw_body_limit_and_closure(headers, monkeypatch):
    raw_calls = []
    original = httpx.Response.aiter_raw
    async def raw(self, *args, **kwargs):
        raw_calls.append(True)
        async for item in original(self, *args, **kwargs):
            yield item
    async def forbidden(*_a, **_kw):
        raise AssertionError("auto-decoded stream must not be used")
        yield b""
    monkeypatch.setattr(httpx.Response, "aiter_raw", raw)
    monkeypatch.setattr(httpx.Response, "aiter_bytes", forbidden)
    body = Stream([b"a" * 16384] * (MAX_BYTES // 16384 + 1))
    fetch, transports = service(lambda r: response(body, headers=headers))
    with pytest.raises(FetchError, match="RESPONSE_TOO_LARGE"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert raw_calls and body.closed and all(t.closed for t in transports)


def test_declared_limit_rejected_without_read():
    body = Stream([b"a"])
    fetch, _ = service(lambda r: response(body, headers={"Content-Length": str(MAX_BYTES + 1)}))
    with pytest.raises(FetchError, match="RESPONSE_TOO_LARGE"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert body.reads == 0 and body.closed


def test_exact_body_limit_allowed():
    body = Stream([b"a" * MAX_BYTES])
    fetch, _ = service(lambda r: response(body))
    result = asyncio.run(fetch.fetch("https://example.com"))
    assert len(result.content) == 12000 and result.truncated


@pytest.mark.parametrize("error,code", [(httpx.ConnectTimeout("canary"), E.TIMEOUT),
    (httpx.ReadTimeout("canary"), E.TIMEOUT), (httpx.ConnectError("canary"), E.NETWORK_ERROR),
    (httpx.RemoteProtocolError("canary"), E.INVALID_RESPONSE)])
def test_http_exceptions(error, code):
    body = Stream(error=error)
    fetch, transports = service(lambda r: response(body))
    with pytest.raises(FetchError) as err:
        asyncio.run(fetch.fetch("https://example.com"))
    assert err.value.code == code and "canary" not in str(err.value)
    assert body.closed and all(t.closed for t in transports)


@pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
def test_http_status(status):
    body = Stream([b"canary"])
    fetch, _ = service(lambda r: response(body, status))
    with pytest.raises(FetchError, match="HTTP_ERROR"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert body.reads == 0 and body.closed


def test_redirect_to_private_rejected():
    seen = []
    def handler(r):
        seen.append(r)
        return response(status=302, headers={"Location": "https://10.0.0.1/"})
    fetch, transports = service(handler)
    with pytest.raises(FetchError, match="TARGET_NOT_ALLOWED"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert len(seen) == 1 and all(t.closed for t in transports)


@pytest.mark.parametrize("location", ["http://other.example/", "http://example.com/", "//example.com:80/"])
def test_https_downgrade_rejected(location):
    seen = []
    def handler(r):
        seen.append(r)
        return response(status=302, headers={"Location": location})
    fetch, _ = service(handler)
    with pytest.raises(FetchError, match="REDIRECT_ERROR"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert len(seen) == 1


@pytest.mark.parametrize("redirects,success", [(3, True), (4, False)])
def test_redirect_limit_and_dns_rechecked(redirects, success):
    seen, dns = [], []
    async def resolver(host, port):
        dns.append((host, port))
        return ["8.8.8.8"]
    def handler(r):
        seen.append(r)
        if len(seen) <= redirects:
            return response(status=302, headers={"Location": f"/hop{len(seen)}", "Set-Cookie": "private=canary"})
        assert "cookie" not in r.headers
        return response()
    fetch, transports = service(handler, resolver)
    if success:
        assert asyncio.run(fetch.fetch("https://example.com")).content == "hello"
    else:
        with pytest.raises(FetchError, match="REDIRECT_ERROR"):
            asyncio.run(fetch.fetch("https://example.com"))
    assert len(dns) == len(seen) == 4 and all(t.closed for t in transports)


def test_http_upgrade_allowed():
    seen = []
    def handler(r):
        seen.append(r.url.scheme)
        return response(status=302, headers={"Location": "https://example.com/"}) if len(seen) == 1 else response()
    fetch, _ = service(handler)
    assert asyncio.run(fetch.fetch("http://example.com")).content == "hello"
    assert seen == ["http", "https"]


@pytest.mark.parametrize("location", [None, "", "file:///tmp/a", "https://u:p@example.com", "https://example.com:444/"])
def test_invalid_redirect(location):
    headers = {} if location is None else {"Location": location}
    fetch, _ = service(lambda r: response(status=302, headers=headers))
    with pytest.raises(FetchError, match="REDIRECT_ERROR"):
        asyncio.run(fetch.fetch("https://example.com"))


def test_shared_budget_across_redirects(monkeypatch):
    monkeypatch.setattr(fetcher, "TOTAL_TIMEOUT", 0.25)
    seen, deadlines = [], []
    real_timeout_at = asyncio.timeout_at
    def observed_timeout_at(deadline):
        deadlines.append(deadline)
        return real_timeout_at(deadline)
    monkeypatch.setattr(fetcher.asyncio, "timeout_at", observed_timeout_at)
    async def handler(r):
        seen.append(r)
        if len(seen) == 1:
            return response(status=302, headers={"Location": "/next"})
        # Wait until the shared deadline cancels this hop; no sleep-ratio race.
        await asyncio.Event().wait()
        raise AssertionError("deadline failed to cancel the second hop")
    fetch, transports = service(handler)
    with pytest.raises(FetchError, match="TIMEOUT"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert len(seen) == 2 and all(t.closed for t in transports)
    assert len(deadlines) == 1


def test_cancellation_closes_response_and_transport():
    body = Stream([b"a"], delay=10)
    fetch, transports = service(lambda r: response(body))
    async def run():
        task = asyncio.create_task(fetch.fetch("https://example.com"))
        while not body.reads:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert body.closed and all(t.closed for t in transports)


def test_total_timeout_closes_stalled_body(monkeypatch):
    monkeypatch.setattr(fetcher, "TOTAL_TIMEOUT", 0.02)
    body = Stream([b"a"], delay=10)
    fetch, transports = service(lambda r: response(body))
    with pytest.raises(FetchError, match="TIMEOUT"):
        asyncio.run(fetch.fetch("https://example.com"))
    assert body.closed and all(t.closed for t in transports)


@pytest.mark.parametrize("headers,code", [
    ([("Content-Type", "text/html"), ("Content-Type", "text/html")], "UNSUPPORTED_CONTENT_TYPE"),
    ([("Content-Type", "text/html"), ("Content-Encoding", "identity"), ("Content-Encoding", "identity")], "UNSUPPORTED_CONTENT_ENCODING"),
    ([("Content-Type", "text/html"), ("Content-Length", "not-a-number")], "INVALID_RESPONSE"),
])
def test_ambiguous_response_headers_rejected(headers, code):
    body = Stream([b"canary"])
    fetch, _ = service(lambda r: httpx.Response(200, headers=headers, stream=body))
    with pytest.raises(FetchError, match=code):
        asyncio.run(fetch.fetch("https://example.com"))
    assert body.closed and body.reads == 0
