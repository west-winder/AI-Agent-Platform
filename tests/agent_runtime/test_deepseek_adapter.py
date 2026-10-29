"""Mock SDK 与 MockTransport 验证 DeepSeek 边界，无真实网络请求。"""

import asyncio
from copy import deepcopy
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpcore2
import httpx2
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletion
import pytest

from backend.agent_runtime.contracts import LLMMessage, LLMRequest, LLMToolCall, LLMToolDefinition
from backend.agent_runtime.deepseek_adapter import DeepSeekAdapter, classify_llm_error
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


@pytest.mark.parametrize("status,delay", [(408, 0), (500, 0), (503, 0), (599, 0), (409, None), (400, None), (401, None), (422, None), (429, None)])
def test_agent_classification_uses_preserved_sdk_status_without_changing_translation(status, delay, legacy_service) -> None:
    sdk = APIStatusError("private", response=httpx2.Response(status, request=REQUEST), body=None)
    error = legacy_service._translate_external_error(sdk)
    adapter, create, translator, _ = make_adapter(error=sdk)
    translator.return_value = error
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(adapter.complete(request()))
    assert raised.value is error
    assert error.__cause__ is sdk
    assert classify_llm_error(error).delay_seconds == delay
    assert classify_llm_error(error).status_code == status
    create.assert_awaited_once()


@pytest.mark.parametrize("headers,expected", [
    ({"Retry-After": "1.5"}, 1.5), ({"retry-after": "0"}, 0),
    ({"retry-after-ms": "250"}, 0.25), ({"retry-after": "120"}, 120),
    ({"retry-after": "Thu, 01 Jan 1970 00:16:50 GMT"}, 10),
    ({}, None), ({"retry-after": "bad"}, None), ({"retry-after": "inf"}, None),
    ({"retry-after": "nan"}, None), ({"retry-after": "-1"}, None),
    ({"retry-after": "121"}, None), ({"retry-after-ms": "bad"}, None),
    ({"retry-after": "Thu, 01 Jan 1970 00:16:39 GMT"}, None),
])
def test_rate_limit_requires_real_valid_bounded_wait_hint(monkeypatch, headers, expected) -> None:
    monkeypatch.setattr("backend.agent_runtime.deepseek_adapter.time.time", lambda: 1000)
    sdk = APIStatusError("private", response=httpx2.Response(429, headers=headers, request=REQUEST), body=None)
    error = ExternalServiceError("unavailable")
    error.__cause__ = sdk
    assert classify_llm_error(error).delay_seconds == expected
    assert sdk.response.headers == httpx2.Headers(headers)


@pytest.mark.parametrize("source,expected", [
    (httpx2.ConnectError("network", request=REQUEST), 0),
    (httpx2.ReadError("network", request=REQUEST), None),
    (httpx2.WriteError("network", request=REQUEST), None),
    (AttributeError("bug"), None), (RuntimeError("bug"), None),
    (httpx2.RemoteProtocolError("protocol", request=REQUEST), None), (None, None),
])
def test_connection_classification_requires_confirmed_network_source(source, expected) -> None:
    sdk = APIConnectionError(request=REQUEST)
    sdk.__cause__ = source
    error = ExternalServiceError("unavailable")
    error.__cause__ = sdk
    assert classify_llm_error(error).delay_seconds == expected


def test_network_wrapper_does_not_hide_programming_or_tls_configuration_error() -> None:
    import errno
    import socket
    import ssl

    for root in (AttributeError("bug"), ssl.SSLCertVerificationError("certificate"),
                 PermissionError(errno.EACCES, "permission"), OSError("unknown"),
                 socket.gaierror(socket.EAI_NONAME, "invalid host")):
        network = httpx2.ConnectError("network", request=REQUEST)
        network.__cause__ = root
        sdk = APIConnectionError(request=REQUEST)
        sdk.__cause__ = network
        error = ExternalServiceError("unavailable")
        error.__cause__ = sdk
        assert classify_llm_error(error).delay_seconds is None


def test_confirmed_temporary_os_and_dns_failures_remain_retryable() -> None:
    import errno
    import socket

    for root in (OSError(errno.ETIMEDOUT, "timeout"), socket.gaierror(socket.EAI_AGAIN, "temporary")):
        network = httpx2.ConnectError("network", request=REQUEST)
        network.__cause__ = root
        sdk = APIConnectionError(request=REQUEST)
        sdk.__cause__ = network
        error = ExternalServiceError("unavailable")
        error.__cause__ = sdk
        assert classify_llm_error(error).delay_seconds == 0


def test_legacy_exception_construction_and_missing_cause_remain_compatible() -> None:
    for cls in (ExternalServiceError, ExternalTimeoutError, ExternalRequestError):
        error = cls("old message")
        assert error.args == ("old message",)
        assert str(error) == "old message"
        assert error.__cause__ is None
        assert classify_llm_error(error).delay_seconds == (0 if cls is ExternalTimeoutError else None)


def core_sdk_connection_error(core_type, root_kind):
    import errno
    import ssl
    from httpx2._transports.default import map_httpcore_exceptions

    roots = {
        "refused": ConnectionRefusedError(10061, "temporary refusal"),
        "timeout": TimeoutError("temporary timeout"),
        "reset": ConnectionResetError(10054, "temporary reset"),
        "pipe": BrokenPipeError(32, "broken pipe"),
        "aborted": ConnectionAbortedError("temporary abort"),
        "approved_errno": ConnectionError(errno.ECONNRESET, "temporary reset"),
        "connection_unknown": ConnectionError("unknown"),
        "connection_permission": ConnectionError(13, "permission"),
        "tls": ssl.SSLCertVerificationError("certificate"),
        "handshake": ssl.SSLError("handshake"),
        "permission": PermissionError(13, "permission"),
        "bug": AttributeError("bug"),
        "runtime": RuntimeError("bug"),
        "unknown": OSError("unknown"),
        "auth": APIStatusError("unauthorized", response=httpx2.Response(401, request=REQUEST), body=None),
        "type_only": httpx2.NetworkError("no transient evidence", request=REQUEST),
    }
    root = roots.get(root_kind)
    if root_kind in ("aggregate_refused", "aggregate_bug"):
        root = OSError("All connection attempts failed")
        root.__cause__ = roots["refused" if root_kind == "aggregate_refused" else "bug"]
    core = core_type("transport failure")
    core.__cause__ = core if root_kind == "cycle" else root
    try:
        with map_httpcore_exceptions():
            raise core
    except httpx2.RequestError as mapped:
        # 也验证宽泛 APIConnectionError 包装连接超时时的保守分类。
        sdk = APIConnectionError(request=REQUEST)
        sdk.__cause__ = mapped
        return sdk


@pytest.mark.parametrize("core_type,root", [(httpcore2.ConnectError, "refused"),
                                         (httpcore2.ConnectTimeout, "timeout"),
                                         (httpcore2.ConnectError, "aggregate_refused")])
@pytest.mark.parametrize("recover", [True, False])
def test_core_connection_chain_classification_and_runtime_limit(core_type, root, recover, legacy_service) -> None:
    from backend.agent_runtime.bootstrap import build_agent_runtime
    from backend.tools.infrastructure.contracts import ToolContext
    from backend.tools.infrastructure.registry import ToolRegistry

    first = core_sdk_connection_error(core_type, root)
    final = core_sdk_connection_error(core_type, root)
    adapter, create, translator, _ = make_adapter()
    translator.side_effect = legacy_service._translate_external_error
    create.side_effect = [first, sdk_response() if recover else final]
    runtime = build_agent_runtime(llm=adapter, registry=ToolRegistry())
    if recover:
        result = asyncio.run(runtime.run(request().messages, ToolContext()))
        assert result.final_answer == "完成"
        assert result.iterations == 1
    else:
        with pytest.raises(ExternalServiceError) as raised:
            asyncio.run(runtime.run(request().messages, ToolContext()))
        assert raised.value.__cause__ is final
    assert create.await_count == 2
    assert isinstance(first.__cause__.__cause__, core_type)
    # 分类器消费 Adapter 保留的完整原因链，而非只有裸 SDK 异常。
    external = legacy_service._translate_external_error(first)
    external.__cause__ = first
    assert classify_llm_error(external).delay_seconds == 0


@pytest.mark.parametrize("core_type", [httpcore2.ConnectError, httpcore2.ConnectTimeout])
@pytest.mark.parametrize("root", ["tls", "permission", "bug", "unknown", "missing", "cycle", "aggregate_bug"])
def test_core_connection_chain_rejects_unsafe_causes_in_runtime(core_type, root, legacy_service) -> None:
    from backend.agent_runtime.bootstrap import build_agent_runtime
    from backend.tools.infrastructure.contracts import ToolContext
    from backend.tools.infrastructure.registry import ToolRegistry

    error = core_sdk_connection_error(core_type, root)
    adapter, create, translator, _ = make_adapter(error=error)
    translator.side_effect = legacy_service._translate_external_error
    runtime = build_agent_runtime(llm=adapter, registry=ToolRegistry())
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(runtime.run(request().messages, ToolContext()))
    assert raised.value.__cause__ is error
    assert classify_llm_error(raised.value).delay_seconds is None
    create.assert_awaited_once()


@pytest.mark.parametrize("core_type", [httpcore2.ReadError, httpcore2.WriteError])
@pytest.mark.parametrize("root", ["reset", "refused", "aborted", "pipe", "timeout", "approved_errno"])
@pytest.mark.parametrize("recover", [True, False])
def test_read_write_chain_recovers_once_or_propagates_final_failure(core_type, root, recover, legacy_service) -> None:
    from backend.agent_runtime.bootstrap import build_agent_runtime
    from backend.tools.infrastructure.contracts import ToolContext
    from backend.tools.infrastructure.registry import ToolRegistry

    first = core_sdk_connection_error(core_type, root)
    final = core_sdk_connection_error(core_type, root)
    assert isinstance(first.__cause__.__cause__, core_type)
    adapter, create, translator, _ = make_adapter()
    translator.side_effect = legacy_service._translate_external_error
    create.side_effect = [first, sdk_response() if recover else final]
    runtime = build_agent_runtime(llm=adapter, registry=ToolRegistry())
    if recover:
        result = asyncio.run(runtime.run(request().messages, ToolContext()))
        assert result.final_answer == "完成"
        assert result.iterations == 1
    else:
        with pytest.raises(ExternalServiceError) as raised:
            asyncio.run(runtime.run(request().messages, ToolContext()))
        assert raised.value.__cause__ is final
    assert create.await_count == 2


@pytest.mark.parametrize("core_type", [httpcore2.ReadError, httpcore2.WriteError])
@pytest.mark.parametrize("root", ["tls", "handshake", "permission", "auth", "bug", "runtime",
                                 "unknown", "missing", "cycle", "aggregate_bug", "type_only",
                                 "connection_unknown", "connection_permission"])
def test_read_write_chain_rejects_unconfirmed_causes_without_runtime_retry(core_type, root, legacy_service) -> None:
    from backend.agent_runtime.bootstrap import build_agent_runtime
    from backend.tools.infrastructure.contracts import ToolContext
    from backend.tools.infrastructure.registry import ToolRegistry

    error = core_sdk_connection_error(core_type, root)
    adapter, create, translator, _ = make_adapter(error=error)
    translator.side_effect = legacy_service._translate_external_error
    runtime = build_agent_runtime(llm=adapter, registry=ToolRegistry())
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(runtime.run(request().messages, ToolContext()))
    assert raised.value.__cause__ is error
    assert classify_llm_error(raised.value).delay_seconds is None
    create.assert_awaited_once()


@pytest.mark.parametrize("error_type", [httpx2.ReadError, httpx2.WriteError])
def test_read_write_bare_error_does_not_authorize_runtime_retry(error_type, legacy_service) -> None:
    from backend.agent_runtime.bootstrap import build_agent_runtime
    from backend.tools.infrastructure.contracts import ToolContext
    from backend.tools.infrastructure.registry import ToolRegistry

    error = APIConnectionError(request=REQUEST)
    error.__cause__ = error_type("no transient evidence", request=REQUEST)
    adapter, create, translator, _ = make_adapter(error=error)
    translator.side_effect = legacy_service._translate_external_error
    runtime = build_agent_runtime(llm=adapter, registry=ToolRegistry())
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(runtime.run(request().messages, ToolContext()))
    assert raised.value.__cause__ is error
    create.assert_awaited_once()


@pytest.mark.parametrize("core_type", [httpcore2.ConnectError, httpcore2.ConnectTimeout])
def test_generic_connection_error_keeps_existing_connect_policy(core_type) -> None:
    sdk = core_sdk_connection_error(core_type, "connection_unknown")
    error = ExternalServiceError("unavailable")
    error.__cause__ = sdk
    # 本次只收紧 Read/Write，保留 Connect 入口原有的类型判定。
    assert classify_llm_error(error).delay_seconds == 0
