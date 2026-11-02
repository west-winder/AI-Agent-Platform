"""HTTP flow: validate each hop, pin its IP, stream raw bounded HTML."""

import asyncio
from collections.abc import Callable

import httpx

from .contracts import FetchWebPageResult
from .errors import ErrorCode as E, FetchError
from .html_text import decode_html, extract_html_text
from .security import Resolver, ValidatedTarget, resolve_public_target, system_resolver, validate_url

TOTAL_TIMEOUT = 30.0
MAX_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
_REDIRECTS = {301, 302, 303, 307, 308}


class PinnedIPTransport(httpx.AsyncBaseTransport):
    """One validated hop only; never pass its domain to TCP resolution."""

    def __init__(self, target: ValidatedTarget, inner: httpx.AsyncBaseTransport | None = None):
        self.target = target
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport(
            verify=True, trust_env=False, retries=0, http1=True, http2=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url != self.target.url or not self.target.ips:
            raise FetchError(E.TARGET_NOT_ALLOWED)
        host = self.target.host
        headers = request.headers.copy()
        headers["Host"] = f"[{host}]" if ":" in host else host
        extensions = dict(request.extensions)
        # HTTPcore 1.0.9 passes this directly to start_tls: it must be str.
        extensions["sni_hostname"] = host
        pinned = httpx.Request(
            request.method, request.url.copy_with(host=self.target.ips[0]),
            headers=headers, stream=request.stream, extensions=extensions,
        )
        return await self.inner.handle_async_request(pinned)

    async def aclose(self):
        await self.inner.aclose()


class WebPageFetcher:
    def __init__(self, resolver: Resolver = system_resolver,
                 transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None):
        self.resolver = resolver
        self.transport_factory = transport_factory

    async def fetch(self, url: str) -> FetchWebPageResult:
        try:
            deadline = asyncio.get_running_loop().time() + TOTAL_TIMEOUT
            async with asyncio.timeout_at(deadline):
                return await self._fetch(url, deadline)
        except (TimeoutError, httpx.TimeoutException):
            raise FetchError(E.TIMEOUT) from None
        except httpx.RemoteProtocolError:
            raise FetchError(E.INVALID_RESPONSE) from None
        except httpx.RequestError:
            raise FetchError(E.NETWORK_ERROR) from None

    async def _fetch(self, url: str, deadline: float) -> FetchWebPageResult:
        target = validate_url(url)
        for hop in range(MAX_REDIRECTS + 1):
            target = await resolve_public_target(target, self.resolver)
            inner = self.transport_factory() if self.transport_factory else None
            transport = PinnedIPTransport(target, inner)
            async with httpx.AsyncClient(
                transport=transport, trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(10.0, connect=5.0, write=5.0, pool=5.0),
                headers={"Accept": "text/html", "Accept-Encoding": "identity",
                         "User-Agent": "WebPageFetchMCP/1.0"},
            ) as client:
                async with client.stream("GET", target.url) as response:
                    if response.status_code in _REDIRECTS:
                        if hop >= MAX_REDIRECTS or len(response.headers.get_list("location")) != 1:
                            raise FetchError(E.REDIRECT_ERROR)
                        location = response.headers["location"]
                        if not location or any(ord(c) <= 32 or ord(c) == 127 or c == "\\" for c in location):
                            raise FetchError(E.REDIRECT_ERROR)
                        try:
                            next_target = validate_url(str(target.url.join(location)))
                        except (httpx.InvalidURL, FetchError):
                            raise FetchError(E.REDIRECT_ERROR) from None
                        if target.url.scheme == "https" and next_target.url.scheme == "http":
                            raise FetchError(E.REDIRECT_ERROR)
                        target = next_target
                        continue
                    if not 200 <= response.status_code < 300:
                        raise FetchError(E.HTTP_ERROR)
                    encodings = response.headers.get_list("content-encoding")
                    if encodings and (len(encodings) != 1 or encodings[0].strip().lower() != "identity"):
                        raise FetchError(E.UNSUPPORTED_CONTENT_ENCODING)
                    types = response.headers.get_list("content-type")
                    if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "text/html":
                        raise FetchError(E.UNSUPPORTED_CONTENT_TYPE)
                    lengths = response.headers.get_list("content-length")
                    if lengths:
                        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                            raise FetchError(E.INVALID_RESPONSE)
                        if len(lengths[0]) > 10 or int(lengths[0]) > MAX_BYTES:
                            raise FetchError(E.RESPONSE_TOO_LARGE)
                    data = bytearray()
                    async for chunk in response.aiter_raw(chunk_size=16_384):
                        if len(data) + len(chunk) > MAX_BYTES:
                            raise FetchError(E.RESPONSE_TOO_LARGE)
                        data.extend(chunk)
            html = decode_html(bytes(data), types[0])
            text = extract_html_text(html, deadline)
            return FetchWebPageResult(
                title=text.title, url=str(target.url), content=text.content,
                truncated=text.truncated, content_type="text/html",
            )
        raise FetchError(E.REDIRECT_ERROR)
