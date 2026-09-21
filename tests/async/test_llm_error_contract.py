"""
[KEEP]

Error Handling V1 Contract Regression

验证 Error Handling V1 的核心边界：

1. External Error Translation
   - SDK Timeout → ExternalTimeoutError
   - Connection / RateLimit / Server Error → ExternalServiceError
   - Auth / Request Error → ExternalRequestError
2. Programming Error Boundary
   - AttributeError 原样 Propagate，不被伪装成 External Error
3. Structured Output Contract
   - parsed=None → ValueError（不变成 ExternalServiceError）
4. Router HTTP Mapping
   - ExternalTimeoutError → 504
   - ExternalServiceError → 503
   - ExternalRequestError → 500
5. Streaming（llm_service 层）
   - Stream 建立阶段失败 → External Error Translation
   - Stream 消费阶段中途失败 → External Error Translation（partial chunk 不丢失）
   - chunk parsing 的 Programming Error → 原样 Propagate
6. Streaming SSE Error Event（Router 层）
   - partial delta 保留 + error payload 不被当成 delta

不调用真实 DeepSeek。
不依赖真实网络。

本仓库没有 pytest-asyncio：
所有 test 函数保持同步 def，
async contract 用 run() = asyncio.run() 驱动。
"""

import asyncio
from types import SimpleNamespace

import httpx2

from openai import (
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    InternalServerError,
)

from fastapi import HTTPException

import backend.services.llm_service as llm_service
import backend.routers.chat as chat_router

from backend.services.llm_service import (
    _translate_external_error,
)

from backend.exceptions.external_exceptions import (
    ExternalServiceError,
    ExternalTimeoutError,
    ExternalRequestError,
)


def run(coro):
    return asyncio.run(coro)


# ==================================================
# HTTP Request / Response 构造
# ==================================================

REQUEST = httpx2.Request(
    "POST",
    "https://example.com/v1/chat/completions",
)


def make_response(status_code: int):
    return httpx2.Response(
        status_code=status_code,
        request=REQUEST,
    )


# ==================================================
# 1. External Error Translation
# ==================================================

def test_translate_timeout_to_external_timeout():
    """
    TimeoutError（Python Overall Timeout）
    +
    APITimeoutError（SDK Timeout）
    → ExternalTimeoutError

    注意 APITimeoutError 是 APIConnectionError 的子类，
    判定顺序必须保证它先落入 Timeout 分支，
    不能被归为 ExternalServiceError。
    """

    for exc in (
        TimeoutError(),
        APITimeoutError(REQUEST),
    ):
        result = _translate_external_error(exc)

        assert type(result) is ExternalTimeoutError, (
            f"{type(exc).__name__} 应翻译为 ExternalTimeoutError，"
            f"实际 {type(result).__name__}"
        )


def test_translate_connection_to_external_service():
    """
    APIConnectionError → ExternalServiceError（精确类型）
    """

    result = _translate_external_error(
        APIConnectionError(request=REQUEST)
    )

    assert type(result) is ExternalServiceError


def test_translate_status_errors():
    """
    APIStatusError Mapping：
    408 → ExternalTimeoutError
    409 / 429 / 500 → ExternalServiceError
    401 / 404 → ExternalRequestError
    """

    result = _translate_external_error(
        AuthenticationError(
            "authentication failed",
            response=make_response(401),
            body=None,
        )
    )
    assert type(result) is ExternalRequestError

    result = _translate_external_error(
        RateLimitError(
            "rate limited",
            response=make_response(429),
            body=None,
        )
    )
    assert type(result) is ExternalServiceError

    result = _translate_external_error(
        InternalServerError(
            "server error",
            response=make_response(500),
            body=None,
        )
    )
    assert type(result) is ExternalServiceError


# ==================================================
# 2. Chat Completion Error Boundary
# ==================================================

class FakeCompletions:

    def __init__(self, exc):
        self.exc = exc

    async def create(self, **kwargs):
        raise self.exc


class FakeChatClient:

    def __init__(self, exc):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(exc)
        )


def run_chat_completion_with(exc):
    original_client = llm_service.client

    llm_service.client = FakeChatClient(exc)

    try:
        return run(llm_service.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            model="fake-model",
        ))
    finally:
        llm_service.client = original_client


def test_chat_completion_translates_sdk_timeout():
    """
    APITimeoutError → ExternalTimeoutError
    且保留原始 SDK 异常作为 __cause__
    """

    original_error = APITimeoutError(REQUEST)

    try:
        run_chat_completion_with(original_error)

    except ExternalTimeoutError as exc:
        assert exc.__cause__ is original_error

    else:
        raise AssertionError(
            "APITimeoutError 应该被翻译为 ExternalTimeoutError"
        )


def test_chat_completion_propagates_programming_error():
    """
    Programming Error（AttributeError）
    不被 External Translation 捕获，
    原样 Propagate（identity 相等）
    """

    original_error = AttributeError(
        "fake programming bug"
    )

    try:
        run_chat_completion_with(original_error)

    except AttributeError as exc:
        assert exc is original_error

    else:
        raise AssertionError(
            "AttributeError 应该原样传播"
        )


# ==================================================
# 3. Structured Output Contract
# ==================================================

class FakeOutputModel(SimpleNamespace):
    pass


class FakeResponses:

    def __init__(self, exc=None, parsed=None):
        self.exc = exc
        self.parsed = parsed

    async def parse(self, **kwargs):

        if self.exc is not None:
            raise self.exc

        return SimpleNamespace(
            output_parsed=self.parsed
        )


class FakeStructuredClient:

    def __init__(self, exc=None, parsed=None):
        self.responses = FakeResponses(
            exc=exc,
            parsed=parsed,
        )


def run_structured_with(*, exc=None, parsed=None):
    original_client = llm_service.client

    llm_service.client = FakeStructuredClient(
        exc=exc,
        parsed=parsed,
    )

    try:
        return run(llm_service.structured_completion(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            output_model=dict,
            model="fake-model",
        ))
    finally:
        llm_service.client = original_client


def test_structured_translates_sdk_timeout():
    """
    APITimeoutError → ExternalTimeoutError
    """

    original_error = APITimeoutError(REQUEST)

    try:
        run_structured_with(exc=original_error)

    except ExternalTimeoutError as exc:
        assert exc.__cause__ is original_error

    else:
        raise AssertionError(
            "APITimeoutError 应该被翻译为 ExternalTimeoutError"
        )


def test_structured_parsed_none_raises_value_error():
    """
    parsed=None 是 Structured Output Contract Failure，
    不是 External Failure，
    必须保持 ValueError。
    """

    try:
        run_structured_with(parsed=None)

    except ValueError as exc:
        assert str(exc) == (
            "LLM structured output 没有返回可解析结果"
        )

    else:
        raise AssertionError(
            "parsed=None 应该抛出 ValueError"
        )


# ==================================================
# 4. Router HTTP Mapping
# ==================================================

def call_chat_endpoint_with(error):
    original_chat = chat_router.chat

    async def fake_chat(**kwargs):
        raise error

    chat_router.chat = fake_chat

    request = SimpleNamespace(
        conversation_id=1,
        content="hello",
    )

    try:
        return run(chat_router.chat_endpoint(
            request=request,
            db=None,
        ))
    finally:
        chat_router.chat = original_chat


def test_router_maps_timeout_to_504():

    try:
        call_chat_endpoint_with(
            ExternalTimeoutError("timeout")
        )

    except HTTPException as exc:
        assert exc.status_code == 504

    else:
        raise AssertionError("应该抛出 HTTPException 504")


def test_router_maps_service_to_503():

    try:
        call_chat_endpoint_with(
            ExternalServiceError("unavailable")
        )

    except HTTPException as exc:
        assert exc.status_code == 503

    else:
        raise AssertionError("应该抛出 HTTPException 503")


def test_router_maps_request_to_500():

    try:
        call_chat_endpoint_with(
            ExternalRequestError("bad request")
        )

    except HTTPException as exc:
        assert exc.status_code == 500

    else:
        raise AssertionError("应该抛出 HTTPException 500")


def test_router_propagates_programming_error():
    """
    Router 的 External Error Handler
    不能吞掉 Programming Error。
    """

    original_error = AttributeError(
        "fake programming bug"
    )

    try:
        call_chat_endpoint_with(original_error)

    except AttributeError as exc:
        assert exc is original_error

    else:
        raise AssertionError(
            "AttributeError 应该原样传播，不被 Router 捕获"
        )


# ==================================================
# 5. Streaming（llm_service 层）Error Boundary
# ==================================================

class FakeStream:

    def __init__(self, events):
        self.events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):

        if not self.events:
            raise StopAsyncIteration

        event = self.events.pop(0)

        # 事件是异常对象时，
        # 模拟 Stream 消费过程中失败。
        if isinstance(event, Exception):
            raise event

        return event


class FakeStreamCompletions:

    def __init__(self, create_error=None, stream_events=None):
        self.create_error = create_error
        self.stream_events = stream_events or []

    async def create(self, **kwargs):

        if self.create_error is not None:
            raise self.create_error

        return FakeStream(self.stream_events)


class FakeStreamClient:

    def __init__(self, create_error=None, stream_events=None):
        self.chat = SimpleNamespace(
            completions=FakeStreamCompletions(
                create_error=create_error,
                stream_events=stream_events,
            )
        )


def make_chunk(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content)
            )
        ]
    )


def collect_stream(*, create_error=None, stream_events=None):
    """
    消费 stream_chat_completion 的 AsyncIterator。

    返回 (chunks, error)：
    - 正常结束 → (chunks, None)
    - 抛出异常 → (已收到的 chunks, 异常对象)
    """

    original_client = llm_service.client

    llm_service.client = FakeStreamClient(
        create_error=create_error,
        stream_events=stream_events,
    )

    chunks = []

    async def consume():
        async for chunk in llm_service.stream_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            model="fake-model",
        ):
            chunks.append(chunk)

    error = None

    try:
        run(consume())
    except BaseException as exc:
        error = exc
    finally:
        llm_service.client = original_client

    return chunks, error


def test_stream_creation_error_translated():
    """
    Stream 建立阶段 SDK Timeout
    → ExternalTimeoutError
    """

    original_error = APITimeoutError(REQUEST)

    chunks, error = collect_stream(create_error=original_error)

    assert chunks == []

    assert isinstance(error, ExternalTimeoutError)

    assert error.__cause__ is original_error


def test_stream_mid_stream_error_translated():
    """
    Stream 已输出一个 chunk 后连接中断
    → ExternalServiceError

    且已产出的 partial chunk 不丢失。
    """

    original_error = APIConnectionError(request=REQUEST)

    chunks, error = collect_stream(
        stream_events=[
            make_chunk("hello"),
            original_error,
        ]
    )

    assert chunks == ["hello"]

    assert isinstance(error, ExternalServiceError)

    assert error.__cause__ is original_error


def test_stream_chunk_programming_error_propagates():
    """
    chunk parsing 的 Programming Error（AttributeError）
    不被 External Translation 捕获。
    """

    bad_chunk = SimpleNamespace()  # 没有 choices

    chunks, error = collect_stream(
        stream_events=[bad_chunk]
    )

    assert chunks == []

    assert isinstance(error, AttributeError)

    assert not isinstance(error, ExternalServiceError)


# ==================================================
# 6. Streaming SSE Error Event（Router 层）
# ==================================================

async def fake_stream_chat_with_mid_stream_error(**kwargs):

    yield "hello"

    raise ExternalServiceError(
        "fake external service failure"
    )


def test_stream_router_emits_sse_error_event():
    """
    正常 delta 输出后，Mid-stream ExternalServiceError：
    - partial delta event 仍然保留
    - ExternalServiceError → SSE error event
    """

    original_stream_chat = chat_router.stream_chat

    chat_router.stream_chat = (
        fake_stream_chat_with_mid_stream_error
    )

    request = SimpleNamespace(
        conversation_id=1,
        content="hello",
    )

    try:
        response = run(chat_router.stream_chat_endpoint(
            request=request,
            db=None,
        ))

        chunks = []

        async def consume():
            async for item in response.body_iterator:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                chunks.append(item)

        run(consume())

        body = "".join(chunks)

        # partial delta 不丢失
        assert '"type": "delta"' in body
        assert '"delta": "hello"' in body

        # error payload 不被当成 delta，
        # 而是转换成 SSE error event
        assert "event: error" in body
        assert '"type": "error"' in body
        assert '"code": "service_unavailable"' in body

    finally:
        chat_router.stream_chat = original_stream_chat
