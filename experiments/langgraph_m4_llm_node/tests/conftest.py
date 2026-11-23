import ipaddress
import socket
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langsmith import tracing_context


def pytest_addoption(parser):
    parser.addoption(
        "--deepseek-smoke",
        action="store_true",
        help="Run the real DeepSeek smoke test against the configured endpoint.",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("deepseek_smoke") and not item.config.getoption(
        "--deepseek-smoke"
    ):
        pytest.skip("DeepSeek smoke requires --deepseek-smoke")


@pytest.fixture(autouse=True)
def tracing_disabled():
    with tracing_context(enabled=False):
        yield


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
def local_network_only(monkeypatch, request):
    if request.node.get_closest_marker("deepseek_smoke"):
        return

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

    # 与 M4-2 一致，允许 Windows asyncio 初始化时连接数字回环地址。
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture
def model_invoke():
    # Spy 记录 invoke 的实际调用，同时执行原有假模型逻辑。
    with patch.object(
        GenericFakeChatModel, "invoke", autospec=True, side_effect=GenericFakeChatModel.invoke
    ) as spy:
        yield spy
