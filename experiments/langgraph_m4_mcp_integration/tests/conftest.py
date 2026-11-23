import ipaddress
import socket

import pytest
from langsmith import tracing_context


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    getaddrinfo = socket.getaddrinfo

    def require_loopback(host):
        try:
            if host is None or ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError("Offline MCP tests only allow numeric loopback addresses")

    def guarded_connect(sock, address):
        require_loopback(address[0])
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        require_loopback(address[0])
        return connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        require_loopback(host)
        return getaddrinfo(host, *args, **kwargs)

    # Windows asyncio uses a loopback socket pair internally. MCP uses pipes.
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    with tracing_context(enabled=False):
        yield
