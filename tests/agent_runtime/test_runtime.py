"""使用 FakeLLM 和真实 Registry/Dispatcher 验证完整循环。"""

import asyncio
from copy import deepcopy
import json
import logging
from typing import Any
from unittest.mock import AsyncMock

import httpx2
from openai import APIConnectionError, APIStatusError
import pytest
from pydantic import BaseModel, Field

from backend.agent_runtime.contracts import (
    AgentRunStatus,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
)
from backend.agent_runtime.errors import LLMProtocolError, LLMResponseTruncatedError
from backend.agent_runtime.runtime import AgentRuntime
from backend.agent_runtime.context_builder import ContextBudget, ContextBuilder, estimate_input_tokens
from backend.agent_runtime.errors import ContextBudgetExceededError, ContextInputError
from backend.exceptions.external_exceptions import ExternalServiceError, ExternalTimeoutError
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


def http_failure(status: int, headers=None) -> ExternalServiceError:
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    error = ExternalServiceError("private-api-key-and-upstream-text")
    error.__cause__ = APIStatusError(
        "private-upstream", response=httpx2.Response(status, headers=headers, request=request), body=None
    )
    return error


def network_failure() -> ExternalServiceError:
    sdk = APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    sdk.__cause__ = httpx2.ConnectError("network")
    error = ExternalServiceError("unavailable")
    error.__cause__ = sdk
    return error


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


@pytest.mark.parametrize("error", [AttributeError("bug"), asyncio.CancelledError()])
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


def test_context_trimming_preserves_history_and_fixed_start_across_tool_rounds(components) -> None:
    """每轮重新构造 Context，完整 History 和请求起点不受裁剪影响。"""
    class RecordingBuilder(ContextBuilder):
        def __init__(self) -> None:
            """记录独立快照而非 Runtime 的可变 History 引用。"""
            super().__init__(ContextBudget(5000, 0, 0))
            self.snapshots = []

        def build(self, history, tools, current_turn_start):
            """记录每轮输入再执行实际裁剪。"""
            self.snapshots.append((deepcopy(history), current_turn_start))
            return super().build(history, tools, current_turn_start)

    builder = RecordingBuilder()
    llm = FakeLLM(
        tool_response(call("a", "一"), call("b", "二")),
        tool_response(call("c", "三")), LLMResponse("完成", "stop"),
        LLMResponse("新 Run", "stop"),
    )
    registry, dispatcher, _, _ = components
    runtime = AgentRuntime(llm, registry, dispatcher, context_builder=builder)
    messages = [LLMMessage("system", "系统"), LLMMessage("user", "旧" * 3000),
                LLMMessage("assistant", "旧回答"), LLMMessage("user", "当前问题")]
    original = deepcopy(messages)
    first = asyncio.run(runtime.run(messages, ToolContext()))
    assert [start for _, start in builder.snapshots] == [3, 3, 3]
    assert [len(history) for history, _ in builder.snapshots] == [4, 7, 9]
    assert [len(request.messages) for request in llm.requests] == [2, 5, 7]
    assert llm.requests[2].messages == [messages[0], *first.messages[3:-1]]
    assert first.messages[:4] == messages == original
    assert len(first.messages) == 10
    second = asyncio.run(runtime.run([LLMMessage("user", "下一次")], ToolContext()))
    assert second.messages == [LLMMessage("user", "下一次"), LLMMessage("assistant", "新 Run")]
    assert builder.snapshots[-1][1] == 0
    assert llm.requests[-1].messages == [LLMMessage("user", "下一次")]


def test_context_budget_failure_before_first_llm_call(components) -> None:
    """必要内容超额直接抛错，不调用模型、工具或泄露消息。"""
    llm = FakeLLM()
    registry, dispatcher, _, _ = components
    runtime = AgentRuntime(llm, registry, dispatcher,
                           context_builder=ContextBuilder(ContextBudget(1, 0, 0)))
    with pytest.raises(ContextBudgetExceededError) as raised:
        asyncio.run(runtime.run([LLMMessage("user", "private-question")], ToolContext()))
    assert "private-question" not in str(raised.value)
    assert llm.requests == []
    dispatcher.dispatch.assert_not_awaited()


def test_tool_result_overflow_stops_before_next_llm_call(components) -> None:
    """本轮工具正常执行后，下一轮超额不再调用模型且不重试。"""
    registry, dispatcher, _, _ = components
    messages = [LLMMessage("user", "问题")]
    tools = [LLMToolDefinition("echo", "回声测试", EchoArgs.model_json_schema())]
    budget = ContextBudget(estimate_input_tokens(messages, tools), 0, 0)
    llm = FakeLLM(tool_response(call()))
    runtime = AgentRuntime(llm, registry, dispatcher, context_builder=ContextBuilder(budget))
    with pytest.raises(ContextBudgetExceededError):
        asyncio.run(runtime.run(messages, ToolContext()))
    assert len(llm.requests) == 1
    dispatcher.dispatch.assert_awaited_once()
    assert messages == [LLMMessage("user", "问题")]


@pytest.mark.parametrize("messages", [
    [LLMMessage("system", "系统")], [LLMMessage("user", " ")],
    [LLMMessage("user", "问题"), LLMMessage("assistant", "回答")],
    [LLMMessage("user", "问题"), LLMMessage("assistant", tool_calls=[call()])],
])
def test_invalid_current_request_rejected_before_model_and_tools(components, messages) -> None:
    """不接受缺失请求起点、已完成请求或悬空工具调用的输入。"""
    llm = FakeLLM()
    with pytest.raises(ContextInputError):
        asyncio.run(make_runtime(llm, components).run(messages, ToolContext()))
    assert llm.requests == []
    components[1].dispatch.assert_not_awaited()


def test_history_that_initially_fits_is_trimmed_after_tool_result(components) -> None:
    """工具结果增长后才触发裁剪，后续请求仍保留当前调用及结果。"""
    registry, dispatcher, _, _ = components
    messages = [LLMMessage("user", "x" * 1000), LLMMessage("assistant", "旧答"),
                LLMMessage("user", "新问题")]
    tools = [LLMToolDefinition("echo", "回声测试", EchoArgs.model_json_schema())]
    budget = ContextBudget(estimate_input_tokens(messages, tools), 0, 0)
    llm = FakeLLM(tool_response(call()), LLMResponse("完成", "stop"))
    runtime = AgentRuntime(llm, registry, dispatcher, context_builder=ContextBuilder(budget))
    result = asyncio.run(runtime.run(messages, ToolContext()))
    assert llm.requests[0].messages == messages
    assert llm.requests[1].messages == result.messages[2:-1]
    assert [message.role for message in llm.requests[1].messages] == ["user", "assistant", "tool"]
    assert result.messages[:3] == messages


@pytest.mark.parametrize("error", [ExternalTimeoutError("timeout"), http_failure(503), http_failure(429, {"retry-after": "0"}), network_failure()])
def test_retry_recovers_without_consuming_iteration(components, error) -> None:
    llm = FakeLLM(error, LLMResponse("恢复", "stop"))
    result = asyncio.run(make_runtime(llm, components, 1).run([LLMMessage("user", "问题")], ToolContext()))
    assert result.iterations == 1
    assert result.status is AgentRunStatus.COMPLETED
    assert len(llm.requests) == 2
    assert llm.requests[0] == llm.requests[1]
    assert llm.requests[0] is not llm.requests[1]
    components[1].dispatch.assert_not_awaited()


@pytest.mark.parametrize("final", [http_failure(502), AttributeError("private-bug"), LLMProtocolError("private-protocol")])
def test_two_failures_propagate_final_error_with_safe_correlated_logs(components, caplog, final) -> None:
    first = http_failure(503)
    llm = FakeLLM(first, final, LLMResponse("不得调用", "stop"))
    with caplog.at_level(logging.WARNING, logger="backend.agent_runtime.runtime"):
        with pytest.raises(type(final)) as raised:
            asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "private-prompt")], ToolContext()))
    assert raised.value is final
    final_status = 502 if isinstance(final, ExternalServiceError) else None
    if final_status:
        assert raised.value.__cause__.status_code == final_status
    assert len(llm.requests) == 2
    records = [r for r in caplog.records if r.msg == "agent_runtime.llm_failure"]
    assert [r.attempt for r in records] == [1, 2]
    assert [r.status_code for r in records] == [503, final_status]
    assert [r.will_retry for r in records] == [True, False]
    assert records[0].llm_call_id == records[1].llm_call_id
    assert "private-" not in str([r.__dict__ for r in records])


@pytest.mark.parametrize("error", [
    http_failure(409), http_failure(400), http_failure(401), http_failure(403), http_failure(422),
    http_failure(429), http_failure(429, {"retry-after": "bad"}),
    http_failure(429, {"retry-after": "nan"}), http_failure(429, {"retry-after": "121"}),
    http_failure(429, {"retry-after": "-1"}), LLMProtocolError("invalid"),
    LLMResponseTruncatedError("truncated"), ExternalServiceError("unknown"), TimeoutError("unknown"),
])
def test_non_retryable_failure_is_preserved(components, error) -> None:
    llm = FakeLLM(error, LLMResponse("不得调用", "stop"))
    with pytest.raises(type(error)) as raised:
        asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "问题")], ToolContext()))
    assert raised.value is error
    assert len(llm.requests) == 1


def test_sdk_wrapped_programming_error_is_not_retried(components) -> None:
    sdk = APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    sdk.__cause__ = AttributeError("private-bug")
    error = ExternalServiceError("unavailable")
    error.__cause__ = sdk
    llm = FakeLLM(error)
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(make_runtime(llm, components).run([LLMMessage("user", "问题")], ToolContext()))
    assert raised.value is error
    assert raised.value.__cause__.__cause__ is sdk.__cause__
    assert len(llm.requests) == 1


def test_failed_attempt_cannot_mutate_retry_snapshot(components) -> None:
    class MutatingLLM(FakeLLM):
        async def complete(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                request.messages.clear()
                request.tools[0].parameters.clear()
                raise ExternalTimeoutError("timeout")
            assert request.messages[0].content == "问题"
            assert request.tools[0].parameters["properties"]["text"]
            return LLMResponse("完成", "stop")

    messages = [LLMMessage("user", "问题")]
    llm = MutatingLLM()
    result = asyncio.run(make_runtime(llm, components).run(messages, ToolContext()))
    assert result.messages[0] == messages[0]
    assert len(llm.requests) == 2


def test_retry_after_tool_does_not_replay_tool_and_budget_resets_per_iteration(components, monkeypatch) -> None:
    async def scenario():
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        with monkeypatch.context() as patch:
            patch.setattr(loop, "time", lambda: clock[0])

            class ClockLLM(FakeLLM):
                async def complete(self, request):
                    clock[0] += 100
                    return await super().complete(request)

            # 第二轮单独拥有 180 秒；第二轮首次失败后再使用 50 秒。
            class LLM(ClockLLM):
                async def complete(self, request):
                    if len(self.requests) == 2:
                        clock[0] -= 50
                    return await super().complete(request)

            llm = LLM(tool_response(call()), ExternalTimeoutError("timeout"), LLMResponse("完成", "stop"))
            result = await make_runtime(llm, components).run([LLMMessage("user", "问题")], ToolContext())
            assert result.iterations == 2
            assert result.status is AgentRunStatus.COMPLETED
            assert llm.requests[1].messages == llm.requests[2].messages
            components[1].dispatch.assert_awaited_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("wait", [0, 30])
def test_second_attempt_is_cancelled_at_original_180_second_deadline(components, monkeypatch, wait) -> None:
    async def scenario():
        loop = asyncio.get_running_loop()
        started = loop.time()
        clock = [started]
        cancellations = []
        sleeps = []
        first = ExternalTimeoutError("first") if not wait else http_failure(429, {"retry-after": str(wait)})
        with monkeypatch.context() as patch:
            patch.setattr(loop, "time", lambda: clock[0])

            async def sleep(delay):
                sleeps.append(delay)
                clock[0] += delay

            patch.setattr("backend.agent_runtime.runtime.asyncio.sleep", sleep)

            class LLM(FakeLLM):
                async def complete(self, request):
                    self.requests.append(request)
                    if len(self.requests) == 1:
                        clock[0] += 120
                        raise first
                    assert clock[0] - started == 120 + wait
                    # 恰在最初截止时间挂起；不能获得新的 120 秒。
                    clock[0] = started + 180
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancellations.append(True)

            llm = LLM()
            with pytest.raises(ExternalTimeoutError) as raised:
                await make_runtime(llm, components).run([LLMMessage("user", "问题")], ToolContext())
            assert isinstance(raised.value.__cause__, TimeoutError)
            assert clock[0] - started == 180
            assert cancellations == [True]
            assert len(llm.requests) == 2
            assert sleeps == ([wait] if wait else [])

    asyncio.run(scenario())


@pytest.mark.parametrize("elapsed,headers", [(180, None), (120, {"retry-after": "60"}), (150, {"retry-after": "40"})])
def test_insufficient_budget_preserves_first_exception(components, monkeypatch, elapsed, headers) -> None:
    async def scenario():
        loop = asyncio.get_running_loop()
        clock = [loop.time()]
        first = http_failure(429, headers) if headers else ExternalTimeoutError("first")
        with monkeypatch.context() as patch:
            patch.setattr(loop, "time", lambda: clock[0])

            class LLM(FakeLLM):
                async def complete(self, request):
                    self.requests.append(request)
                    clock[0] += elapsed
                    raise first

            llm = LLM()
            with pytest.raises(ExternalServiceError) as raised:
                await make_runtime(llm, components).run([LLMMessage("user", "问题")], ToolContext())
            assert raised.value is first
            assert len(llm.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["first", "second", "wait"])
def test_active_cancellation_preserved_in_each_recovery_phase(components, monkeypatch, phase) -> None:
    async def scenario():
        entered = asyncio.Event()
        calls = []

        async def blocked_sleep(delay):
            entered.set()
            await asyncio.Event().wait()

        if phase == "wait":
            monkeypatch.setattr("backend.agent_runtime.runtime.asyncio.sleep", blocked_sleep)

        class LLM(FakeLLM):
            async def complete(self, request):
                calls.append(request)
                if len(calls) == 1 and phase != "first":
                    if phase == "wait":
                        raise http_failure(429, {"retry-after": "1"})
                    raise ExternalTimeoutError("first")
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(make_runtime(LLM(), components).run([LLMMessage("user", "问题")], ToolContext()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == (2 if phase == "second" else 1)

    asyncio.run(scenario())


def test_adapter_timeout_then_runtime_budget_preserves_nested_timeout_semantics(components, monkeypatch) -> None:
    from backend.agent_runtime.deepseek_adapter import DeepSeekAdapter

    async def scenario():
        calls = []
        loop = asyncio.get_running_loop()
        start = loop.time()
        clock = [start]

        async def blocked(**kwargs):
            calls.append(kwargs)
            clock[0] = start + (120 if len(calls) == 1 else 180)
            await asyncio.Event().wait()

        from types import SimpleNamespace
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=blocked)))
        adapter = DeepSeekAdapter(client, model="test", overall_timeout_seconds=120,
                                  translate_error=lambda exc: ExternalTimeoutError("adapter timeout"))
        with monkeypatch.context() as patch:
            patch.setattr(loop, "time", lambda: clock[0])
            with pytest.raises(ExternalTimeoutError) as raised:
                await make_runtime(adapter, components).run([LLMMessage("user", "问题")], ToolContext())
        assert len(calls) == 2
        assert isinstance(raised.value.__cause__, TimeoutError)
        assert clock[0] - start == 180

    asyncio.run(scenario())


def test_retry_still_skips_tool_batch_on_last_iteration(components) -> None:
    llm = FakeLLM(ExternalTimeoutError("first"), tool_response(call()))
    result = asyncio.run(make_runtime(llm, components, 1).run([LLMMessage("user", "问题")], ToolContext()))
    assert result.status is AgentRunStatus.MAX_ITERATIONS_REACHED
    assert result.iterations == 1
    assert result.messages == [LLMMessage("user", "问题")]
    components[1].dispatch.assert_not_awaited()


@pytest.mark.parametrize("phase", ["first", "wait", "success"])
def test_budget_covers_first_call_wait_and_success_before_deadline(components, monkeypatch, phase) -> None:
    async def scenario():
        loop = asyncio.get_running_loop()
        start = loop.time()
        clock = [start]
        first = http_failure(429, {"retry-after": "30"})
        waits = []
        with monkeypatch.context() as patch:
            patch.setattr(loop, "time", lambda: clock[0])

            async def sleep(delay):
                waits.append(delay)
                if phase == "wait":
                    clock[0] = start + 180
                    await asyncio.Event().wait()
                clock[0] += delay

            patch.setattr("backend.agent_runtime.runtime.asyncio.sleep", sleep)

            class LLM(FakeLLM):
                async def complete(self, request):
                    self.requests.append(request)
                    if phase == "first":
                        clock[0] = start + 180
                        await asyncio.Event().wait()
                    if len(self.requests) == 1:
                        clock[0] += 120
                        raise first
                    assert clock[0] - start == 150
                    clock[0] += 29
                    return LLMResponse("完成", "stop")

            llm = LLM()
            runtime = make_runtime(llm, components)
            if phase == "success":
                result = await runtime.run([LLMMessage("user", "问题")], ToolContext())
                assert result.iterations == 1
                assert clock[0] - start == 179
                assert waits == [30]
                assert len(llm.requests) == 2
            else:
                with pytest.raises(ExternalServiceError) as raised:
                    await runtime.run([LLMMessage("user", "问题")], ToolContext())
                if phase == "wait":
                    assert raised.value is first
                else:
                    assert isinstance(raised.value, ExternalTimeoutError)
                    assert isinstance(raised.value.__cause__, TimeoutError)
                assert len(llm.requests) == 1
                assert clock[0] - start == 180

    asyncio.run(scenario())
