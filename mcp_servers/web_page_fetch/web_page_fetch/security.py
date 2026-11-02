"""Validate every destination before connecting to its numeric IP."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import ipaddress
import re
import socket
from urllib.parse import urlsplit

import httpx

from .errors import ErrorCode as E, FetchError

DNS_TIMEOUT = 5.0
Resolver = Callable[[str, int], Awaitable[Sequence[str]]]
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_V6_DENY = tuple(map(ipaddress.ip_network, (
    "64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32",
)))


@dataclass(frozen=True)
class ValidatedTarget:
    url: httpx.URL
    host: str
    port: int
    ips: tuple[str, ...] = ()


def validate_url(value: str) -> ValidatedTarget:
    if not isinstance(value, str) or not value or any(
        ord(c) <= 32 or ord(c) == 127 or c == "\\" for c in value
    ):
        raise FetchError(E.INVALID_URL)
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
            raise ValueError
        if parts.username is not None or parts.password is not None:
            raise ValueError
        host = parts.hostname
        if not host or "%" in host:
            raise ValueError
        try:
            host = str(ipaddress.ip_address(host))
        except ValueError:
            host = host.encode("idna").decode("ascii").lower()
            if host.endswith("."):
                host = host[:-1]
            if len(host) > 253 or not all(_LABEL.fullmatch(p) for p in host.split(".")):
                raise ValueError
        scheme = parts.scheme.lower()
        expected = 443 if scheme == "https" else 80
        port = parts.port
        if port is not None and port != expected:
            raise ValueError
        if parts.netloc.endswith(":"):
            raise ValueError
        url = httpx.URL(value).copy_with(host=host, port=expected, fragment=None)
        if url.raw_host.decode("ascii") != host or url.scheme != scheme:
            raise ValueError
    except (ValueError, UnicodeError, httpx.InvalidURL):
        raise FetchError(E.INVALID_URL) from None
    if host == "localhost" or host.endswith(".localhost"):
        raise FetchError(E.TARGET_NOT_ALLOWED)
    return ValidatedTarget(url, host, expected)


def validate_public_ip(value: str) -> str:
    try:
        if "%" in value:
            raise ValueError
        ip = ipaddress.ip_address(value)
    except ValueError:
        raise FetchError(E.TARGET_NOT_ALLOWED) from None
    if (not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
            or str(ip) == "168.63.129.16"):
        raise FetchError(E.TARGET_NOT_ALLOWED)
    if isinstance(ip, ipaddress.IPv6Address):
        if (ip.ipv4_mapped is not None or ip not in ipaddress.ip_network("2000::/3")
                or any(ip in net for net in _V6_DENY)):
            raise FetchError(E.TARGET_NOT_ALLOWED)
    return str(ip)


async def system_resolver(host: str, port: int) -> Sequence[str]:
    results = await asyncio.get_running_loop().getaddrinfo(
        host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return [item[4][0] for item in results]


async def resolve_public_target(
    target: ValidatedTarget, resolver: Resolver = system_resolver,
) -> ValidatedTarget:
    try:
        literal = ipaddress.ip_address(target.host)
    except ValueError:
        literal = None
    if literal is not None:
        ips = (validate_public_ip(str(literal)),)
    else:
        try:
            async with asyncio.timeout(DNS_TIMEOUT):
                addresses = await resolver(target.host, target.port)
        except TimeoutError:
            raise FetchError(E.TIMEOUT) from None
        except OSError:
            raise FetchError(E.DNS_ERROR) from None
        if not addresses:
            raise FetchError(E.DNS_ERROR)
        ips = tuple(dict.fromkeys(validate_public_ip(ip) for ip in addresses))
    return ValidatedTarget(target.url, target.host, target.port, ips)
