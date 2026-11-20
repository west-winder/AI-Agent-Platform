import ipaddress
import socket

import pytest


def require_loopback(host: str | None) -> None:
    if host is None:
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise AssertionError("Offline tests only allow numeric loopback addresses")


@pytest.fixture(autouse=True)
def local_network_only(monkeypatch):
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock, address):
        require_loopback(address[0])
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        require_loopback(address[0])
        return connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        require_loopback(host)
        return getaddrinfo(host, *args, **kwargs)

    # Windows asyncio 创建事件循环时使用数字回环地址连接自己的 listener。
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
