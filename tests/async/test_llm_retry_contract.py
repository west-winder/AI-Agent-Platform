"""
[KEEP]

LLM Retry Contract Regression

长期保护 AI Agent Platform 的 Retry V1 Contract：

1. 项目显式 Retry 配置确实进入 AsyncOpenAI Client。
2. Retryable Server Error 可以由 SDK Retry 后恢复。
3. Non-retryable Authentication Error 不会被无脑 Retry。

不调用真实 DeepSeek。
不依赖真实网络。
"""

import asyncio

import httpx2
import openai
from openai import AsyncOpenAI

from backend.services import llm_service


# =========================================================
# Helpers
# =========================================================


def build_success_response(
    request: httpx2.Request,
) -> httpx2.Response:

    return httpx2.Response(
        status_code=200,
        request=request,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "OK",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    )


def build_server_error_response(
    request: httpx2.Request,
) -> httpx2.Response:

    return httpx2.Response(
        status_code=500,
        request=request,
        json={
            "error": {
                "message": "temporary server error",
                "type": "server_error",
                "param": None,
                "code": "server_error",
            }
        },
    )


def build_auth_error_response(
    request: httpx2.Request,
) -> httpx2.Response:

    return httpx2.Response(
        status_code=401,
        request=request,
        json={
            "error": {
                "message": "invalid api key",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key",
            }
        },
    )


# =========================================================
# 1. Project Retry Configuration
# =========================================================


def test_retry_configuration_is_applied():

    assert (
        llm_service.client.max_retries
        == llm_service.LLM_MAX_RETRIES
    )


# =========================================================
# 2. Retryable Failure Can Recover
# =========================================================


def test_retryable_server_error_can_recover():

    async def run():

        attempt_count = 0

        async def handler(
            request: httpx2.Request,
        ) -> httpx2.Response:

            nonlocal attempt_count

            attempt_count += 1

            # 第一次模拟临时服务器失败
            if attempt_count == 1:
                return build_server_error_response(
                    request
                )

            # Retry 后恢复
            return build_success_response(
                request
            )

        http_client = httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                handler
            ),
            trust_env=False,
        )

        client = AsyncOpenAI(
            api_key="test-api-key",
            base_url="https://example.test/v1",
            http_client=http_client,

            # 这个 Case 只需要证明：
            # SDK 至少允许一次 Retry。
            max_retries=1,
        )

        try:

            response = (
                await client.chat.completions.create(
                    model="fake-model",
                    messages=[
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ],
                )
            )

            assert attempt_count == 2

            assert (
                response
                .choices[0]
                .message
                .content
                == "OK"
            )

        finally:
            await client.close()

    asyncio.run(run())


# =========================================================
# 3. Non-retryable Failure Is Not Retried
# =========================================================


def test_authentication_error_is_not_retried():

    async def run():

        attempt_count = 0

        async def handler(
            request: httpx2.Request,
        ) -> httpx2.Response:

            nonlocal attempt_count

            attempt_count += 1

            return build_auth_error_response(
                request
            )

        http_client = httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                handler
            ),
            trust_env=False,
        )

        client = AsyncOpenAI(
            api_key="bad-api-key",
            base_url="https://example.test/v1",
            http_client=http_client,

            # 即使允许两个 Retry，
            # 401 也不应该触发 Retry。
            max_retries=2,
        )

        try:

            try:

                await client.chat.completions.create(
                    model="fake-model",
                    messages=[
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ],
                )

            except openai.AuthenticationError as exc:

                assert exc.status_code == 401

                # 核心 Contract：
                # 401 只发生一次 HTTP Attempt。
                assert attempt_count == 1

                return

            raise AssertionError(
                "Expected AuthenticationError, "
                "but request succeeded."
            )

        finally:
            await client.close()

    asyncio.run(run())