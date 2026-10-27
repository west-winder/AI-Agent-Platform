"""使用 FakeLLM 和真实 Registry/Dispatcher 验证完整循环。"""

import asyncio
from copy import deepcopy
import json
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from backend.agent_runtime.contracts import (
    AgentRunStatus,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
)
from backend.agent_runtime.errors import LLMProtocolError, LLMResponseTruncatedError
from backend.agent_runtime.runtime import AgentRuntime
from backend.exceptions.external_exceptions import ExternalTimeoutError
from backend.tools.infrastructure.contracts import RegisteredTool, ToolContext
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.errors import ToolExecutionError
from backend.tools.infrastructure.registry import ToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)


class FakeLLM:
    def __init__(self, *responses: LLMResponse | BaseException) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        response = self.responses[len(self.requests) - 1]
        if isinstance(response, BaseException):
            raise response
        return response


def call(call_id: str = "call-1", text: str = "你好", name: str = "echo") -> LLMToolCall:
    args = {"text": text}
    return LLMToolCall(call_id, name, json.dumps(args, ensure_ascii=False), args)


def tool_response(*calls: LLMToolCall) -> LLMResponse:
    return LLMResponse(None, "tool_calls", list(calls))


@pytest.fixture
def components():
    events: list[tuple[str, ToolContext]] = []

    async def echo(args: EchoArgs, context: ToolContext) -> dict[str, Any]:
        events.append((args.text, context))
        return {"text": args.text}

    handler = AsyncMock(spec=echo, wraps=echo)
    registry = ToolRegistry()
    registry.register(RegisteredTool("echo", "回声测试", EchoArgs, handler))
    dispatcher = ToolDispatcher(registry)
    dispatcher.dispatch = AsyncMock(wraps=dispatcher.dispatch)
    return registry, dispatcher, handler, events


def make_runtime(llm: FakeLLM, components, max_iterations: int = 5) -> AgentRuntime:
    registry, dispatcher, _, _ = components
    return AgentRuntime(llm, registry, dispatcher, max_iterations)


def test_direct_final_answer_and_tools_generated_from_registry(components) -> None:
    llm = FakeLLM(LLMResponse("直接回答", "stop"))
    messages = [LLMMessage("system", "系统提示"), LLMMessage("user", "问题")]
    result = asyncio.run(make_runtime(llm, components).run(messages, ToolContext()))
    assert result.status is AgentRunStatus.COMPLETED
    assert result.final_answer == "直接回答"
    assert result.iterations == 1
    assert result.messages == [*messages, LLMMessage("assistant", "直接回答")]
    request = llm.requests[0]
    assert request.messages == messages
    assert request.tools[0].name == "echo"
    assert request.tools[0].parameters == EchoArgs.model_json_schema()
    assert not hasattr(request.tools[0], "handler")
    components[1].dispatch.assert_not_awaited()


def test_one_tool_then_answer_preserves_raw_call_and_context(components) -> None:
    tool_call = LLMToolCall("call-1", "echo", '{ "text" : "武汉" }', {"text": "武汉"})
    llm = FakeLLM(LLMResponse("查询中", "tool_calls", [tool_call]), LLMResponse("完成", "stop"))
    context = ToolContext(user_id=42, conversation_id=7)
    messages = [LLMMessage("user", "请调用工具")]
    result = asyncio.run(make_runtime(llm, components).run(messages, context))
    assert result.status is AgentRunStatus.COMPLETED
    assert result.iterations == 2
    assistant, tool = llm.requests[1].messages[1:]
    assert assistant.content == "查询中"
    assert assistant.tool_calls == [tool_call]
    assert assistant.tool_calls[0].raw_arguments == '{ "text" : "武汉" }'
    assert tool.role == "tool"
    assert tool.tool_call_id == assistant.tool_calls[0].id
    assert json.loads(tool.content) == {"status": "success", "data": {"text": "武汉"}, "error": None}
    assert components[3] == [("武汉", context)]
    components[1].dispatch.assert_awaited_once_with(tool_name="echo", arguments={"text": "武汉"}, context=context)
    assert len(llm.requests[0].messages) == 1
    assert len(result.messages) == 4


def test_multiple_tool_rounds_and_sequential_calls_in_one_round(components) -> None:
    llm = FakeLLM(
        tool_response(call("a", "一"), call("b", "二")),
        tool_response(call("c", "三")),
        LLMResponse("全部完成", "stop"),
    )
    result = asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "执行")], ToolContext()))
    assert result.iterations == 3
    assert [text for text, _ in components[3]] == ["一", "二", "三"]
    assert [message.tool_call_id for message in result.messages if message.role == "tool"] == ["a", "b", "c"]
    assert [len(request.messages) for request in llm.requests] == [1, 4, 6]


@pytest.mark.parametrize("raw", ['{"broken":', "[]", "null", '"text"'])
def test_parse_error_skips_dispatch_and_allows_model_correction(components, raw: str) -> None:
    bad_call = LLMToolCall("bad", "echo", raw, parse_error="invalid object")
    llm = FakeLLM(tool_response(bad_call), tool_response(call("fixed")), LLMResponse("恢复成功", "stop"))
    result = asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "执行")], ToolContext()))
    failure = json.loads(llm.requests[1].messages[-1].content)
    assert failure == {
        "status": "error", "data": None,
        "error": {"code": "INVALID_ARGUMENTS", "message": "工具参数不是合法的 JSON 对象", "retryable": False},
    }
    assert llm.requests[1].messages[-1].tool_call_id == "bad"
    assert llm.requests[1].messages[-2].tool_calls[0].raw_arguments == raw
    assert result.final_answer == "恢复成功"
    components[1].dispatch.assert_awaited_once()
    assert components[1].dispatch.await_args.kwargs["arguments"] == {"text": "你好"}


@pytest.mark.parametrize("failure_kind", ["schema", "unknown", "execution"])
def test_tool_failure_in_batch_is_returned_and_following_tool_still_runs(components, failure_kind: str) -> None:
    if failure_kind == "schema":
        failed = LLMToolCall("bad", "echo", "{}", {})
        code = "INVALID_ARGUMENTS"
    elif failure_kind == "unknown":
        failed = call("bad", name="missing")
        code = "TOOL_NOT_FOUND"
    else:
        async def fail(args: EchoArgs, context: ToolContext) -> dict[str, Any]:
            raise ToolExecutionError("TEMP_UNAVAILABLE", "暂时不可用", retryable=True)

        components[0].register(RegisteredTool("fail", "已知失败", EchoArgs, fail))
        failed = call("bad", name="fail")
        code = "TEMP_UNAVAILABLE"
    llm = FakeLLM(tool_response(call("before", "先"), failed, call("after", "后")), LLMResponse("处理完成", "stop"))
    result = asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "执行")], ToolContext()))
    tool_messages = [message for message in result.messages if message.role == "tool"]
    assert [message.tool_call_id for message in tool_messages] == ["before", "bad", "after"]
    payloads = [json.loads(message.content) for message in tool_messages]
    assert [payload["status"] for payload in payloads] == ["success", "error", "success"]
    assert payloads[1]["error"]["code"] == code
    assert payloads[1]["error"]["retryable"] is (failure_kind == "execution")
    assert [text for text, _ in components[3]] == ["先", "后"]
    assert result.status is AgentRunStatus.COMPLETED
    assert components[1].dispatch.await_count == 3


@pytest.mark.parametrize("error", [NameError("bug"), RuntimeError("bug"), asyncio.CancelledError()])
def test_unexpected_handler_error_and_cancellation_propagate_without_executing_rest(components, error: BaseException) -> None:
    components[2].side_effect = error
    llm = FakeLLM(tool_response(call("first"), call("second")))
    with pytest.raises(type(error)) as raised:
        asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "执行")], ToolContext()))
    assert raised.value is error
    assert len(llm.requests) == 1
    components[1].dispatch.assert_awaited_once()


@pytest.mark.parametrize("error", [ExternalTimeoutError("timeout"), AttributeError("bug"), asyncio.CancelledError()])
def test_llm_errors_propagate_without_retry(components, error: BaseException) -> None:
    llm = FakeLLM(error)
    with pytest.raises(type(error)) as raised:
        asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "问题")], ToolContext()))
    assert raised.value is error
    assert len(llm.requests) == 1
    components[1].dispatch.assert_not_awaited()


@pytest.mark.parametrize(
    "invalid_call",
    [LLMToolCall("", "echo", "{}", {}), LLMToolCall(None, "echo", "{}", {}),
     LLMToolCall("bad", "", "{}", {}), LLMToolCall("bad", None, "{}", {})],
)
def test_entire_batch_protocol_is_checked_before_any_side_effect(components, invalid_call: LLMToolCall) -> None:
    llm = FakeLLM(tool_response(call("valid"), invalid_call))
    with pytest.raises(LLMProtocolError):
        asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "执行")], ToolContext()))
    components[1].dispatch.assert_not_awaited()


def test_duplicate_ids_rejected_before_dispatch(components) -> None:
    llm = FakeLLM(tool_response(call("same"), call("same")))
    with pytest.raises(LLMProtocolError, match="Duplicate"):
        asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "执行")], ToolContext()))
    components[1].dispatch.assert_not_awaited()


@pytest.mark.parametrize(
    "response,error_type",
    [(LLMResponse("半句", "length"), LLMResponseTruncatedError),
     (LLMResponse(None, "length", [call()]), LLMResponseTruncatedError),
     (LLMResponse(None, "stop"), LLMProtocolError),
     (LLMResponse(" \n", "stop"), LLMProtocolError),
     (LLMResponse(None, "tool_calls"), LLMProtocolError),
     (LLMResponse("回答", "content_filter"), LLMProtocolError),
     (LLMResponse(None, "stop", [call()]), LLMProtocolError)],
)
def test_invalid_or_truncated_response_never_succeeds_or_executes_tools(components, response, error_type) -> None:
    llm = FakeLLM(response)
    with pytest.raises(error_type):
        asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "问题")], ToolContext()))
    components[1].dispatch.assert_not_awaited()


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_limit_counts_requests_skips_last_batch_and_has_no_dangling_calls(components, limit: int) -> None:
    llm = FakeLLM(*(tool_response(call(f"a-{i}"), call(f"b-{i}")) for i in range(limit + 1)))
    result = asyncio.run(make_runtime(llm, components, limit).run([LLMMessage("user", "执行")], ToolContext()))
    assert result.status is AgentRunStatus.MAX_ITERATIONS_REACHED
    assert result.final_answer is None
    assert result.iterations == limit
    assert len(llm.requests) == limit
    assert components[1].dispatch.await_count == 2 * (limit - 1)
    assistant_ids = [call.id for message in result.messages for call in message.tool_calls]
    tool_ids = [message.tool_call_id for message in result.messages if message.role == "tool"]
    assert assistant_ids == tool_ids
    assert f"a-{limit - 1}" not in assistant_ids


@pytest.mark.parametrize("limit", [1, 3])
def test_final_answer_on_last_allowed_request_completes(components, limit: int) -> None:
    llm = FakeLLM(*(tool_response(call(str(i))) for i in range(limit - 1)), LLMResponse("最后一轮回答", "stop"))
    result = asyncio.run(make_runtime(llm, components, limit).run([LLMMessage("user", "执行")], ToolContext()))
    assert result.status is AgentRunStatus.COMPLETED
    assert result.iterations == len(llm.requests) == limit
    assert result.final_answer == "最后一轮回答"


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, None])
def test_invalid_iteration_limit_rejected(components, limit) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        make_runtime(FakeLLM(), components, limit)


def test_deep_message_copy_isolates_caller_history_and_request_mutations(components) -> None:
    prior = LLMToolCall("past", "echo", '{"nested":{"items":[1]}}', {"nested": {"items": [1]}})
    messages = [LLMMessage("assistant", tool_calls=[prior]), LLMMessage("tool", "{}", tool_call_id="past"), LLMMessage("user", "继续")]
    original = deepcopy(messages)

    class MutatingLLM(FakeLLM):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            request.messages[0].tool_calls[0].arguments["nested"]["items"].append(2)
            request.messages.clear()
            return LLMResponse("完成", "stop")

    result = asyncio.run(make_runtime(MutatingLLM(), components).run(messages, ToolContext()))
    assert messages == original
    assert result.messages[:-1] == original
    result.messages[0].tool_calls[0].arguments["nested"]["items"].append(3)
    assert messages == original


def test_runtime_can_be_reused_and_runs_have_independent_history(components) -> None:
    llm = FakeLLM(LLMResponse("第一次", "stop"), LLMResponse("第二次", "stop"))
    runtime = make_runtime(llm, components)
    first = asyncio.run(runtime.run([LLMMessage("user", "一")], ToolContext()))
    second = asyncio.run(runtime.run([LLMMessage("user", "二")], ToolContext()))
    assert first.iterations == second.iterations == 1
    assert llm.requests[1].messages == [LLMMessage("user", "二")]


def test_empty_input_rejected_before_llm_call(components) -> None:
    llm = FakeLLM()
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(make_runtime(llm, components).run([], ToolContext()))
    assert llm.requests == []


def test_structured_logs_exclude_message_and_arguments(components, caplog) -> None:
    llm = FakeLLM(tool_response(call(text="private-value")), LLMResponse("private-answer", "stop"))
    with caplog.at_level(logging.INFO, logger="backend.agent_runtime.runtime"):
        asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "private-question")], ToolContext()))
    assert "private-" not in str([record.__dict__ for record in caplog.records])
    record = next(record for record in caplog.records if record.message == "agent_runtime.tool_result")
    assert record.iteration == 1
    assert record.tool_name == "echo"
    assert record.tool_status == "success"
