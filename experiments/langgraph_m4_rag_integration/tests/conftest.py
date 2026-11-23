import socket
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langsmith import tracing_context


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def block_network(*args, **kwargs):
        raise AssertionError("The synchronous M4-5 tests run without network access")

    monkeypatch.setattr(socket.socket, "connect", block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", block_network)
    monkeypatch.setattr(socket, "getaddrinfo", block_network)
    with tracing_context(enabled=False):
        yield


@pytest.fixture
def model_invoke():
    with patch.object(
        FakeListChatModel, "invoke", autospec=True, side_effect=FakeListChatModel.invoke
    ) as spy:
        yield spy
