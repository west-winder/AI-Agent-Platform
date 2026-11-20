import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from langchain_deepseek import ChatDeepSeek
from langsmith import tracing_context


@contextmanager
def fixed_completion(
    message: dict[str, Any], *, finish_reason: str = "stop"
) -> Iterator[tuple[ChatDeepSeek, list[dict[str, Any]]]]:
    """在 HTTP 边界返回固定响应，保留适配器的请求构造和消息解析。"""
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline-response",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", **message},
                        "finish_reason": finish_reason,
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with tracing_context(enabled=False):
            yield (
                ChatDeepSeek(
                    model="offline-model",
                    base_url="https://offline.invalid",
                    api_key="offline-placeholder",
                    http_client=client,
                    max_retries=0,
                ),
                requests,
            )
