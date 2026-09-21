"""
[KEEP]

LLM Timeout Contract Regression

验证：

1. HTTP-level timeout 配置已经进入 AsyncOpenAI Client。
2. 普通非流式 Chat 受 Application Overall Timeout 保护。
3. Structured Output 受 Application Overall Timeout 保护。
4. Streaming 不机械复用普通 Chat 的 Overall Timeout。

不调用真实 DeepSeek。
不依赖真实网络。
"""

import asyncio
import os
from types import SimpleNamespace

import pytest
from pydantic import BaseModel


# ---------------------------------------------------------
# Test Environment
# ---------------------------------------------------------
#
# llm_service import 时会初始化 AsyncOpenAI。
# KEEP 测试不应该依赖真实 API Key，
# 因此提供一个不会真正使用的测试值。
#

os.environ.setdefault(
    "DEEPSEEK_API_KEY",
    "test-api-key",
)

os.environ.setdefault(
    "DEEPSEEK_BASE_URL",
    "https://example.invalid",
)


from backend.services import llm_service

from backend.exceptions.external_exceptions import (
    ExternalTimeoutError,
)


# =========================================================
# Fake Structured Output
# =========================================================


class FakeStructuredOutput(BaseModel):
    value: str


# =========================================================
# 1. HTTP Timeout Configuration
# =========================================================


def test_http_timeout_configuration_is_applied():

    timeout = llm_service.client.timeout

    assert (
        timeout.connect
        == llm_service.LLM_CONNECT_TIMEOUT_SECONDS
    )

    assert (
        timeout.read
        == llm_service.LLM_READ_TIMEOUT_SECONDS
    )

    assert (
        timeout.write
        == llm_service.LLM_WRITE_TIMEOUT_SECONDS
    )

    assert (
        timeout.pool
        == llm_service.LLM_POOL_TIMEOUT_SECONDS
    )


# =========================================================
# 2. Normal Chat Overall Timeout
# =========================================================


def test_chat_completion_has_overall_timeout(
    monkeypatch,
):

    async def slow_create(**kwargs):

        # 模拟 Provider 很慢。
        await asyncio.sleep(0.20)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="should not reach here"
                    )
                )
            ]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=slow_create
            )
        )
    )

    monkeypatch.setattr(
        llm_service,
        "client",
        fake_client,
    )

    # 生产环境是 120 秒。
    # 测试中缩短成 0.05 秒，
    # 避免真的等待。
    monkeypatch.setattr(
        llm_service,
        "LLM_CHAT_OVERALL_TIMEOUT_SECONDS",
        0.05,
    )

    async def run():

        await llm_service.call_llm(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )

    # Error Handling V1：
    # Overall Timeout 触发后，llm_service 会把 TimeoutError
    # 翻译成项目级 ExternalTimeoutError，
    # 并保留原始 TimeoutError 作为 __cause__。
    #
    # 本测试验证的核心仍然是
    # “普通 Chat 受 Application Overall Timeout 保护”。
    with pytest.raises(ExternalTimeoutError) as exc_info:
        asyncio.run(run())

    assert isinstance(exc_info.value.__cause__, TimeoutError)


# =========================================================
# 3. Structured Output Overall Timeout
# =========================================================


def test_structured_completion_has_overall_timeout(
    monkeypatch,
):

    async def slow_parse(**kwargs):

        await asyncio.sleep(0.20)

        return SimpleNamespace(
            output_parsed=FakeStructuredOutput(
                value="should not reach here"
            )
        )

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            parse=slow_parse
        )
    )

    monkeypatch.setattr(
        llm_service,
        "client",
        fake_client,
    )

    monkeypatch.setattr(
        llm_service,
        "LLM_STRUCTURED_OVERALL_TIMEOUT_SECONDS",
        0.05,
    )

    async def run():

        await llm_service.call_llm_structured(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            output_model=FakeStructuredOutput,
        )

    # 同 test_chat_completion_has_overall_timeout：
    # TimeoutError → ExternalTimeoutError 翻译，
    # 原始 TimeoutError 保留在 __cause__。
    with pytest.raises(ExternalTimeoutError) as exc_info:
        asyncio.run(run())

    assert isinstance(exc_info.value.__cause__, TimeoutError)


# =========================================================
# 4. Streaming Does Not Use Chat Overall Timeout
# =========================================================


def test_streaming_is_not_cut_off_by_chat_overall_timeout(
    monkeypatch,
):

    async def fake_stream():

        # 每次等待都比测试用的 0.05s Overall 更久。
        #
        # 如果以后有人错误地把普通 Chat Overall Timeout
        # 套到整个 Streaming 生命周期，
        # 这个测试就会失败。
        await asyncio.sleep(0.08)

        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="A"
                    )
                )
            ]
        )

        await asyncio.sleep(0.08)

        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="B"
                    )
                )
            ]
        )

    async def fake_create(**kwargs):

        assert kwargs["stream"] is True

        return fake_stream()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=fake_create
            )
        )
    )

    monkeypatch.setattr(
        llm_service,
        "client",
        fake_client,
    )

    monkeypatch.setattr(
        llm_service,
        "LLM_CHAT_OVERALL_TIMEOUT_SECONDS",
        0.05,
    )

    async def collect_chunks():

        chunks = []

        async for chunk in llm_service.stream_llm(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        ):
            chunks.append(chunk)

        return chunks

    chunks = asyncio.run(
        collect_chunks()
    )

    assert chunks == ["A", "B"]