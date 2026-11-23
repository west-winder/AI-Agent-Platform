import ipaddress
import socket
from unittest.mock import Mock

import pytest
from langsmith import tracing_context

from agent_loop import get_weather


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    getaddrinfo = socket.getaddrinfo

    def require_loopback(host):
        if host is None:
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise AssertionError("Offline tests only allow numeric loopback addresses")

    def guarded_connect(sock, address):
        require_loopback(address[0])
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        require_loopback(address[0])
        return connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        require_loopback(host)
        return getaddrinfo(host, *args, **kwargs)

    # 与 M4-2/M4-3 一致，保留 Windows 本地回环，禁用外部网络和追踪上报。
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    with tracing_context(enabled=False):
        yield


@pytest.fixture
def weather_handler(monkeypatch):
    handler = Mock(wraps=get_weather.func)

    def tracked_weather(city: str) -> str:
        return handler(city=city)

    # ToolNode 安装工具时会读取函数的类型注解。
    monkeypatch.setattr(get_weather, "func", tracked_weather)
    return handler
