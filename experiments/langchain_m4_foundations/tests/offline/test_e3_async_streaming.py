import asyncio
import socket

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk

from e3_async_streaming import (
    FAKE_TEXT,
    QUESTION,
    combine_chunks,
    invoke_async,
    stream_chunks,
    stream_chunks_async,
)


def test_ainvoke_returns_complete_message():
    model = FakeListChatModel(responses=[FAKE_TEXT])
    result = asyncio.run(invoke_async(model, QUESTION))
    assert type(result) is AIMessage
    assert result.content == FAKE_TEXT


def test_stream_yields_ordered_chunks():
    model = FakeListChatModel(responses=[FAKE_TEXT])
    chunks = list(stream_chunks(model, QUESTION))
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert [chunk.content for chunk in chunks] == list(FAKE_TEXT)
    assert combine_chunks(chunks).content == FAKE_TEXT


def test_astream_yields_ordered_chunks_that_can_be_merged():
    async def collect():
        model = FakeListChatModel(responses=[FAKE_TEXT])
        return [chunk async for chunk in stream_chunks_async(model, QUESTION)]

    chunks = asyncio.run(collect())
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert [chunk.content for chunk in chunks] == list(FAKE_TEXT)
    merged = combine_chunks(chunks)
    assert type(merged) is AIMessageChunk
    assert merged.content == FAKE_TEXT


def test_offline_guard_rejects_external_connections():
    with socket.socket() as client:
        with pytest.raises(AssertionError, match="loopback"):
            client.connect(("192.0.2.1", 443))
        with pytest.raises(AssertionError, match="loopback"):
            client.connect_ex(("192.0.2.1", 443))
    with pytest.raises(AssertionError, match="loopback"):
        socket.getaddrinfo("offline.invalid", 443)
