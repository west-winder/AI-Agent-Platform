"""统一契约的协议约束；不涉及真实模型或网络。"""

from dataclasses import fields, is_dataclass

import pytest

from backend.agent_runtime.contracts import (
    AgentRunResult,
    AgentRunStatus,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    validate_llm_response,
)
from backend.agent_runtime.errors import LLMProtocolError, LLMResponseTruncatedError


@pytest.mark.parametrize(
    "contract",
    [LLMMessage, LLMToolDefinition, LLMToolCall, LLMRequest, LLMResponse, AgentRunResult],
)
def test_contracts_are_simple_dataclasses(contract: type) -> None:
    assert is_dataclass(contract)


def test_tool_definitions_do_not_carry_handlers_or_provider_configuration() -> None:
    assert {field.name for field in fields(LLMToolDefinition)} == {"name", "description", "parameters"}
    assert {field.name for field in fields(LLMRequest)} == {"messages", "tools"}


def test_message_and_request_mutable_defaults_are_independent() -> None:
    first = LLMMessage(role="assistant")
    second = LLMMessage(role="assistant")
    first.tool_calls.append(LLMToolCall("a", "echo", "{}", {}))
    assert second.tool_calls == []
    first_request = LLMRequest([first])
    second_request = LLMRequest([second])
    first_request.tools.append(LLMToolDefinition("echo", "Echo", {}))
    assert second_request.tools == []
    first_response = LLMResponse("ok", "stop")
    second_response = LLMResponse("ok", "stop")
    first_response.tool_calls.append(first.tool_calls[0])
    assert second_response.tool_calls == []


def test_raw_arguments_and_parse_failure_remain_available() -> None:
    call = LLMToolCall("call-1", "echo", '{ "broken": ', parse_error="Invalid JSON")
    assert call.raw_arguments == '{ "broken": '
    assert call.arguments is None
    validate_llm_response(LLMResponse(None, "tool_calls", [call]))


def test_status_values_are_explicit_and_serializable_strings() -> None:
    assert AgentRunStatus.COMPLETED == "COMPLETED"
    assert AgentRunStatus.MAX_ITERATIONS_REACHED == "MAX_ITERATIONS_REACHED"


@pytest.mark.parametrize(
    "response",
    [
        LLMResponse(None, "stop"),
        LLMResponse("  \n", "stop"),
        LLMResponse(123, "stop"),
        LLMResponse("partial", "content_filter"),
        LLMResponse("ok", None),
        LLMResponse(None, "tool_calls"),
        LLMResponse(None, "stop", [LLMToolCall("a", "echo", "{}", {})]),
        LLMResponse(None, "tool_calls", [LLMToolCall("", "echo", "{}", {})]),
        LLMResponse(None, "tool_calls", [LLMToolCall("a", " ", "{}", {})]),
        LLMResponse(None, "tool_calls", [LLMToolCall("a", "echo", None, {})]),
        LLMResponse(None, "tool_calls", [LLMToolCall("a", "echo", "{}")]),
        LLMResponse(None, "tool_calls", [LLMToolCall("a", "echo", "{}", {}, "bad")]),
        LLMResponse(None, "tool_calls", [LLMToolCall("a", "echo", "{}", parse_error="")]),
        LLMResponse(None, "tool_calls", [LLMToolCall("a", "echo", "{}", parse_error=123)]),
        LLMResponse(None, "tool_calls", [None]),
        LLMResponse(None, "tool_calls", None),
        None,
    ],
)
def test_invalid_response_contract_is_rejected(response: LLMResponse) -> None:
    with pytest.raises(LLMProtocolError):
        validate_llm_response(response)


def test_length_has_specific_truncation_error_even_with_tool_calls() -> None:
    with pytest.raises(LLMResponseTruncatedError):
        validate_llm_response(LLMResponse("partial", "length", [LLMToolCall("a", "echo", "{")]))
