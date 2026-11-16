import os
import socket

import pytest

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture(autouse=True)
def forbid_external_network(monkeypatch):
    """确保这些测试不可能依赖外部服务，不改变图的调度语义。"""
    def blocked(*args, **kwargs):
        raise AssertionError("M3 实验禁止外部网络调用")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
