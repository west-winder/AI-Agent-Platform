"""Mock SDK 与 MockTransport 验证 DeepSeek 边界，无真实网络请求。"""

import asyncio
from copy import deepcopy
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx2
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletion
import pytest

from backend.agent_runtime.contracts import LLMMessage, LLMRequest, LLMToolCall, LLMToolDefinition
from backend.agent_runtime.deepseek_adapter import DeepSeekAdapter
from backend.agent_runtime.errors import LLMProtocolError, LLMResponseTruncatedError
from backend.exceptions.external_exceptions import ExternalRequestError, ExternalServiceError, ExternalTimeoutError

REQUEST = httpx2.Request("POST", "https://example.invalid/chat/completions")


def sdk_call(call_id="call-1", name="echo", arguments='{"text":"武汉"}', type="function"):
    return SimpleNamespace(id=call_id, type=type, function=SimpleNamespace(name=name, arguments=arguments))


def sdk_response(content="完成", finish_reason="stop", calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(role="assistant", content=content, tool_calls=calls),
    )])


def make_adapter(response=None, *, error=None, timeout=1.0):
    create = AsyncMock(return_value=response, side_effect=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    translated = ExternalServiceError("translated")
    translator = Mock(return_value=translated)
    adapter = DeepSeekAdapter(client, model="test-deepseek", overall_timeout_seconds=timeout, translate_error=translator)
    return adapter, create, translator, translated


def request():
    return LLMRequest([LLMMessage("user", "请调用工具")])


def test_request_serializes_all_message_roles_and_preserves_original_raw_json() -> None:
    raw = '{ "text" : "武汉" }'
    call = LLMToolCall("call-1", "echo", raw, {"text": "武汉"})
    schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    unified = LLMRequest(
        [LLMMessage("system", "系统"), LLMMessage("user", "问题"),
         LLMMessage("assistant", "处理中", [call]), LLMMessage("tool", '{"status":"success"}', tool_call_id="call-1")],
        [LLMToolDefinition("echo", "回声", schema)],
    )
    snapshot = deepcopy(unified)
    adapter, create, translator, _ = make_adapter(sdk_response())
    result = asyncio.run(adapter.complete(unified))
    assert result.content == "完成"
    assert result.finish_reason == "stop"
    assert result.tool_calls == []
    create.assert_awaited_once_with(
        model="test-deepseek",
        messages=[
            {"role": "system", "content": "系统"}, {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "处理中", "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "echo", "arguments": raw}}]},
            {"role": "tool", "content": '{"status":"success"}', "tool_call_id": "call-1"},
        ],
        tools=[{"type": "function", "function": {"name": "echo", "description": "回声", "parameters": schema}}],
        tool_choice="auto", extra_body={"thinking": {"type": "disabled"}},
    )
    translator.assert_not_called()
    create.await_args.kwargs["tools"][0]["function"]["parameters"]["properties"].clear()
    assert unified == snapshot


def test_no_tools_omits_tool_choice_and_sdk_http_retry_configuration() -> None:
    adapter, create, _, _ = make_adapter(sdk_response())
    asyncio.run(adapter.complete(request()))
    assert set(create.await_args.kwargs) == {"model", "messages", "extra_body"}


def test_multiple_tool_calls_preserve_order_content_and_parsed_arguments() -> None:
    adapter, _, _, _ = make_adapter(sdk_response("查询中", "tool_calls", [sdk_call("a"), sdk_call("b", arguments="{}")]))
    result = asyncio.run(adapter.complete(request()))
    assert result.content == "查询中"
    assert [call.id for call in result.tool_calls] == ["a", "b"]
    assert result.tool_calls[0].arguments == {"text": "武汉"}
    assert result.tool_calls[0].raw_arguments == '{"text":"武汉"}'
    assert result.tool_calls[0].parse_error is None
    assert result.tool_calls[1].arguments == {}


@pytest.mark.parametrize("raw", ['{"broken":', "[]", "null", "1", "true", '"string"', "", '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}'])
def test_malformed_non_object_or_non_standard_json_is_recoverable(raw: str) -> None:
    adapter, _, _, _ = make_adapter(sdk_response(None, "tool_calls", [sdk_call(arguments=raw)]))
    result = asyncio.run(adapter.complete(request()))
    call = result.tool_calls[0]
    assert call.id == "call-1"
    assert call.arguments is None
    assert call.parse_error
    assert call.raw_arguments == raw
    # 解析失败记录仍能原样回填给 Provider。
    payload = adapter._serialize_message(LLMMessage("assistant", tool_calls=[call]))
    assert payload["tool_calls"][0]["function"]["arguments"] == raw


@pytest.mark.parametrize(
    "response",
    [
        None, SimpleNamespace(), SimpleNamespace(choices=[]), SimpleNamespace(choices="bad"),
        SimpleNamespace(choices=[None]), SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]),
        sdk_response(None), sdk_response(" \n"), sdk_response(123), sdk_response("text", None),
        sdk_response("text", "content_filter"), sdk_response(None, "tool_calls", []),
        sdk_response(None, "stop", [sdk_call()]), sdk_response(None, "tool_calls", "bad"),
        sdk_response(None, "tool_calls", [sdk_call(call_id=None)]),
        sdk_response(None, "tool_calls", [sdk_call(call_id=" ")]),
        sdk_response(None, "tool_calls", [sdk_call(name=None)]),
        sdk_response(None, "tool_calls", [sdk_call(name="")]),
        sdk_response(None, "tool_calls", [sdk_call(arguments=None)]),
        sdk_response(None, "tool_calls", [sdk_call(arguments={})]),
        sdk_response(None, "tool_calls", [sdk_call(type="custom")]),
        sdk_response(None, "tool_calls", [sdk_call(), sdk_call()]),
    ],
)
def test_invalid_sdk_protocol_raises_explicit_error_without_external_translation(response) -> None:
    adapter, create, translator, _ = make_adapter(response)
    with pytest.raises(LLMProtocolError):
        asyncio.run(adapter.complete(request()))
    create.assert_awaited_once()
    translator.assert_not_called()


def test_invalid_assistant_role_and_multiple_choices_rejected() -> None:
    wrong_role = sdk_response()
    wrong_role.choices[0].message.role = "user"
    multiple = sdk_response()
    multiple.choices *= 2
    for response in (wrong_role, multiple):
        adapter, _, _, _ = make_adapter(response)
        with pytest.raises(LLMProtocolError):
            asyncio.run(adapter.complete(request()))


def test_missing_tool_function_or_call_fields_are_protocol_errors() -> None:
    for invalid_call in (SimpleNamespace(type="function"), SimpleNamespace(type="function", function=SimpleNamespace(arguments="{}"))):
        adapter, _, _, _ = make_adapter(sdk_response(None, "tool_calls", [invalid_call]))
        with pytest.raises(LLMProtocolError):
            asyncio.run(adapter.complete(request()))


def test_length_is_rejected_before_parsing_incomplete_tool_calls() -> None:
    adapter, _, translator, _ = make_adapter(sdk_response(None, "length", [sdk_call(arguments="{")]))
    with pytest.raises(LLMResponseTruncatedError):
        asyncio.run(adapter.complete(request()))
    translator.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [LLMMessage("invalid", "bad"), LLMMessage("user", None), LLMMessage("tool", "{}"),
     LLMMessage("tool", "{}", tool_call_id=" "), LLMMessage("user", 123),
     LLMMessage("user", "bad", tool_call_id="a"),
     LLMMessage("user", "bad", [LLMToolCall("a", "echo", "{}", {})])],
)
def test_invalid_request_protocol_never_calls_client(message) -> None:
    adapter, create, _, _ = make_adapter(sdk_response())
    with pytest.raises(LLMProtocolError):
        asyncio.run(adapter.complete(LLMRequest([message])))
    create.assert_not_awaited()


def test_empty_messages_never_calls_client() -> None:
    adapter, create, _, _ = make_adapter(sdk_response())
    with pytest.raises(LLMProtocolError):
        asyncio.run(adapter.complete(LLMRequest([])))
    create.assert_not_awaited()


@pytest.mark.parametrize("error", [TimeoutError(), APITimeoutError(REQUEST), APIConnectionError(request=REQUEST), APIError("SDK error", REQUEST, body=None)])
def test_known_external_errors_use_injected_translation_and_keep_exact_cause(error) -> None:
    adapter, create, translator, translated = make_adapter(error=error)
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(adapter.complete(request()))
    assert raised.value is translated
    assert raised.value.__cause__ is error
    translator.assert_called_once_with(error)
    create.assert_awaited_once()


@pytest.mark.parametrize(
    "status,expected",
    [(408, ExternalTimeoutError), (409, ExternalServiceError), (429, ExternalServiceError),
     (500, ExternalServiceError), (400, ExternalRequestError), (401, ExternalRequestError), (404, ExternalRequestError)],
)
def test_legacy_translation_is_reused_with_original_http_exception(status, expected, legacy_service) -> None:
    error = APIStatusError("upstream error", response=httpx2.Response(status, request=REQUEST), body=None)
    adapter, create, _, _ = make_adapter(error=error)
    adapter = DeepSeekAdapter(adapter._client, model="test", overall_timeout_seconds=1,
                              translate_error=legacy_service._translate_external_error)
    with pytest.raises(expected) as raised:
        asyncio.run(adapter.complete(request()))
    assert type(raised.value) is expected
    assert raised.value.__cause__ is error
    create.assert_awaited_once()


def test_overall_timeout_cancels_await_and_keeps_timeout_cause() -> None:
    cancelled = []

    async def blocked(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    adapter, create, translator, translated = make_adapter()
    create.side_effect = blocked
    adapter = DeepSeekAdapter(adapter._client, model="test", overall_timeout_seconds=0.01, translate_error=translator)
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(adapter.complete(request()))
    assert raised.value is translated
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert translator.call_args.args[0] is raised.value.__cause__
    assert cancelled == [True]
    create.assert_awaited_once()


@pytest.mark.parametrize("error", [AttributeError("bug"), RuntimeError("bug"), asyncio.CancelledError()])
def test_unknown_programming_error_or_cancellation_is_not_translated(error: BaseException) -> None:
    adapter, create, translator, _ = make_adapter(error=error)
    with pytest.raises(type(error)) as raised:
        asyncio.run(adapter.complete(request()))
    assert raised.value is error
    translator.assert_not_called()
    create.assert_awaited_once()


def test_actual_task_cancellation_propagates_without_translation() -> None:
    async def scenario():
        started = asyncio.Event()

        async def blocked(**kwargs):
            started.set()
            await asyncio.Event().wait()

        adapter, create, translator, _ = make_adapter()
        create.side_effect = blocked
        task = asyncio.create_task(adapter.complete(request()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        translator.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_overall_timeout_rejected(timeout) -> None:
    with pytest.raises(ValueError):
        make_adapter(timeout=timeout)


@pytest.mark.parametrize("model", ["", " ", None])
def test_invalid_model_rejected(model) -> None:
    with pytest.raises(ValueError):
        DeepSeekAdapter(SimpleNamespace(), model=model, overall_timeout_seconds=1, translate_error=Mock())


def test_real_sdk_mock_transport_and_sdk_response_model() -> None:
    received = []

    def respond(req: httpx2.Request) -> httpx2.Response:
        received.append(json.loads(req.content))
        return httpx2.Response(200, json={
            "id": "completion-1", "object": "chat.completion", "created": 1, "model": "test",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "echo", "arguments": '{"text":"SDK"}'}}],
            }}],
        })

    async def scenario():
        async with AsyncOpenAI(api_key="test-not-a-secret", base_url="https://example.invalid/v1", max_retries=0,
                               http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond))) as client:
            adapter = DeepSeekAdapter(client, model="test", overall_timeout_seconds=1, translate_error=Mock())
            result = await adapter.complete(request())
            assert result.tool_calls[0].arguments == {"text": "SDK"}

    asyncio.run(scenario())
    assert len(received) == 1
    assert received[0]["thinking"] == {"type": "disabled"}
    # 同时用安装的 SDK 类型验证非工具回答字段形状。
    sdk = ChatCompletion.model_validate({
        "id": "completion-2", "object": "chat.completion", "created": 1, "model": "test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "完成"}}],
    })
    assert DeepSeekAdapter._convert_response(sdk).content == "完成"


def test_adapter_logs_no_prompts_arguments_or_upstream_exception_text(caplog) -> None:
    adapter, _, _, _ = make_adapter(error=APIConnectionError(message="private-upstream", request=REQUEST))
    with caplog.at_level(logging.INFO, logger="backend.agent_runtime.deepseek_adapter"):
        with pytest.raises(ExternalServiceError):
            asyncio.run(adapter.complete(LLMRequest([LLMMessage("user", "private-prompt")])))
    assert "private-" not in str([record.__dict__ for record in caplog.records])
    assert caplog.records[-1].error_type == "APIConnectionError"
