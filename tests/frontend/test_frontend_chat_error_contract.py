"""
[KEEP]

Frontend Chat Error Handling V1 Regression

验证 frontend/app.py 对 Error Handling V1 Contract 的适配：

1. Streaming Success
   - type == delta → 正常逐步更新同一条 Assistant Message
2. Streaming Error（SSE error event）
   - 已有 partial answer → 保留 + 追加 [生成中断：...] 提示
   - 无 partial answer → 直接显示简洁错误提示
   - error 之后的后续行不再被拼接
3. Non-Stream Error
   - 504 / 503 → 固定稳定文案
   - 500 + detail == "外部服务请求失败" → 外部服务请求失败
   - 500 + 无可靠 detail → 服务器内部错误
     （HTTP 500 不等于 ExternalRequestError）
   - 其他非 200 → 安全读取 FastAPI detail
   - 连接失败 → 通用提示
4. Non-Stream Success
   - 成功路径不受影响

不启动真实 Backend。
不启动真实 Gradio Server。
通过 mock requests 做 deterministic 验证。

本仓库没有 pytest-asyncio：
chat() 是同步 Generator，直接 list() 收集 yield。
"""

import importlib
import sys
from unittest import mock

import requests


# ==================================================
# Fake HTTP Response
# ==================================================

class FakeHTTPResponse:

    def __init__(self, status_code=200, json_body=None, lines=None):
        self.status_code = status_code
        self._json_body = json_body
        self._lines = lines or []

    def json(self):

        if self._json_body is None:
            raise ValueError("no json body")

        return self._json_body

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def sse(text):
    """把 SSE 行编码成 bytes（iter_lines 的真实返回形态）"""
    return text.encode("utf-8")


# ==================================================
# Frontend Module 加载
#
# frontend/app.py 在 import 阶段就会请求
# GET /agents 与 GET /conversations，
# 必须 mock requests.get 之后才能安全 import。
# ==================================================

def load_frontend_app():

    if "frontend.app" in sys.modules:
        return sys.modules["frontend.app"]

    fake_list_response = FakeHTTPResponse(
        status_code=200,
        json_body=[],
    )

    with mock.patch(
        "requests.get",
        return_value=fake_list_response,
    ):
        return importlib.import_module("frontend.app")


def collect_chat(app, message, conversation_id, stream_enabled, fake_response):
    """
    收集 chat() Generator 的全部 yield，
    返回最后一次 yield 的 gradio_messages。
    """

    with mock.patch(
        "requests.post",
        return_value=fake_response,
    ):
        outputs = list(app.chat(
            message,
            [],
            conversation_id,
            stream_enabled,
        ))

    return outputs[-1][0]


# ==================================================
# 1. Streaming Success
# ==================================================

def test_streaming_success_appends_deltas():

    app = load_frontend_app()

    fake = FakeHTTPResponse(
        status_code=200,
        lines=[
            sse('data: {"type": "delta", "delta": "hello"}'),
            sse(""),
            sse('data: {"type": "delta", "delta": " world"}'),
        ],
    )

    messages = collect_chat(app, "hi", 1, True, fake)

    # 只有一条 assistant message，原地更新
    assistant_messages = [
        m for m in messages
        if m["role"] == "assistant"
    ]
    assert len(assistant_messages) == 1

    assert assistant_messages[-1]["content"] == "hello world"


# ==================================================
# 2. Streaming Error（SSE error event）
# ==================================================

def test_streaming_error_keeps_partial_answer_and_stops():

    app = load_frontend_app()

    fake = FakeHTTPResponse(
        status_code=200,
        lines=[
            sse('data: {"type": "delta", "delta": "partial"}'),
            sse("event: error"),
            sse('data: {"type": "error", "code": "service_unavailable", "message": "外部服务暂时不可用"}'),
            # error 之后到达的行不应该再被拼接
            sse('data: {"type": "delta", "delta": " should not appear"}'),
        ],
    )

    messages = collect_chat(app, "hi", 1, True, fake)

    assistant = messages[-1]

    assert assistant["role"] == "assistant"

    assert assistant["content"] == (
        "partial"
        "\n\n[生成中断：外部服务暂时不可用]"
    )

    # 不显示 raw JSON / traceback / SDK 类名
    assert "service_unavailable" not in assistant["content"]
    assert "ExternalServiceError" not in assistant["content"]
    assert "{" not in assistant["content"]


def test_streaming_error_without_partial_shows_message():

    app = load_frontend_app()

    fake = FakeHTTPResponse(
        status_code=200,
        lines=[
            sse("event: error"),
            sse('data: {"type": "error", "code": "timeout", "message": "外部服务响应超时"}'),
        ],
    )

    messages = collect_chat(app, "hi", 1, True, fake)

    assistant = messages[-1]

    assert assistant["content"] == "外部服务响应超时"


def test_streaming_non_200_shows_stable_message():

    app = load_frontend_app()

    fake = FakeHTTPResponse(
        status_code=404,
        json_body={"detail": "Conversation not found"},
    )

    messages = collect_chat(app, "hi", 1, True, fake)

    assistant = messages[-1]

    assert assistant["content"] == "Conversation not found"


# ==================================================
# 3. Non-Stream Error
# ==================================================

def test_non_stream_error_status_mapping():
    """
    HTTP 504 / 503 与 Backend External Error → HTTP Mapping
    一一对应，使用固定文案。
    """

    app = load_frontend_app()

    cases = [
        (504, "外部服务响应超时"),
        (503, "外部服务暂时不可用"),
    ]

    for status_code, expected in cases:

        fake = FakeHTTPResponse(
            status_code=status_code,
            json_body={"detail": "irrelevant"},
        )

        messages = collect_chat(
            app, "hi", 1, False, fake
        )

        assistant = messages[-1]

        assert assistant["role"] == "assistant"

        assert assistant["content"] == expected, (
            f"status {status_code} 应显示稳定错误提示 {expected}，"
            f"实际 {assistant['content']}"
        )


def test_non_stream_http_500_uses_external_detail_when_reliable():
    """
    HTTP 500 + Backend 明确给出
    detail = "外部服务请求失败"
    → 外部服务请求失败

    即：只有 Router 主动返回的 ExternalRequestError
    才能被认定为 External Error。
    """

    app = load_frontend_app()

    fake = FakeHTTPResponse(
        status_code=500,
        json_body={"detail": "外部服务请求失败"},
    )

    messages = collect_chat(app, "hi", 1, False, fake)

    assert messages[-1]["content"] == (
        "外部服务请求失败"
    )


def test_non_stream_http_500_without_reliable_detail_is_internal_error():
    """
    HTTP 500 不等于 ExternalRequestError：

    Programming Error（AttributeError / TypeError / ...）
    同样会由 FastAPI 返回 500。

    任何「不可靠 detail」都不能被推断成 External Error，
    统一回退「服务器内部错误」。
    """

    app = load_frontend_app()

    cases = [
        None,                                 # body 不是 JSON
        {},                                   # 没有 detail
        {"detail": 123},                      # detail 非 str
        {"detail": ""},                       # detail 为空
        {"detail": "Internal Server Error"},  # FastAPI 默认 500 body
    ]

    for json_body in cases:

        fake = FakeHTTPResponse(
            status_code=500,
            json_body=json_body,
        )

        messages = collect_chat(
            app, "hi", 1, False, fake
        )

        assert messages[-1]["content"] == "服务器内部错误", (
            f"json_body={json_body!r} 应回退「服务器内部错误」，"
            f"实际 {messages[-1]['content']!r}"
        )


def test_non_stream_error_reads_detail():

    app = load_frontend_app()

    fake = FakeHTTPResponse(
        status_code=404,
        json_body={"detail": "Conversation not found"},
    )

    messages = collect_chat(app, "hi", 1, False, fake)

    assert messages[-1]["content"] == (
        "Conversation not found"
    )


def test_non_stream_connection_failure():

    app = load_frontend_app()

    with mock.patch(
        "requests.post",
        side_effect=requests.ConnectionError,
    ):
        outputs = list(app.chat("hi", [], 1, False))

    messages = outputs[-1][0]

    assert messages[-1]["content"] == (
        "请求失败：无法连接服务器"
    )


# ==================================================
# 4. Non-Stream Success（成功路径不回归）
# ==================================================

def test_non_stream_success_unchanged():

    app = load_frontend_app()

    fake = FakeHTTPResponse(
        status_code=200,
        json_body={
            "conversation_id": 1,
            "message_id": 2,
            "answer": "你好，我可以帮你",
        },
    )

    messages = collect_chat(app, "hi", 1, False, fake)

    assert messages[-1]["content"] == "你好，我可以帮你"
