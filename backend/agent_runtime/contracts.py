"""Provider 无关的数据契约；不保存 Handler，也不解析模型 JSON。"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from backend.agent_runtime.errors import LLMProtocolError, LLMResponseTruncatedError

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    id: str
    name: str
    # 必须保留原文；不能用重新序列化的 arguments 替换 Assistant 历史。
    raw_arguments: str
    arguments: dict[str, Any] | None = None
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: MessageRole
    content: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class LLMToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: list[LLMMessage]
    tools: list[LLMToolDefinition] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str | None
    finish_reason: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)


class AgentRunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    status: AgentRunStatus
    final_answer: str | None
    iterations: int
    messages: list[LLMMessage]


def validate_tool_calls(tool_calls: list[LLMToolCall]) -> None:
    """校验整批关联信息，确保无效的后续调用不会导致前面的工具先执行。"""
    if not isinstance(tool_calls, list):
        raise LLMProtocolError("tool_calls must be a list")
    seen_ids: set[str] = set()
    for call in tool_calls:
        if not isinstance(call, LLMToolCall):
            raise LLMProtocolError("Invalid tool call record")
        if not isinstance(call.id, str) or not call.id.strip():
            raise LLMProtocolError("Tool call ID is required")
        if not isinstance(call.name, str) or not call.name.strip():
            raise LLMProtocolError("Tool name is required")
        if call.id in seen_ids:
            raise LLMProtocolError("Duplicate tool call ID in one batch")
        seen_ids.add(call.id)
        if not isinstance(call.raw_arguments, str):
            raise LLMProtocolError("Raw tool arguments must be a string")
        if call.parse_error is None:
            if not isinstance(call.arguments, dict):
                raise LLMProtocolError("Parsed tool arguments must be an object")
        elif (
            not isinstance(call.parse_error, str)
            or not call.parse_error.strip()
            or call.arguments is not None
        ):
            raise LLMProtocolError("Inconsistent tool argument parse result")


def validate_llm_response(response: LLMResponse) -> None:
    """Adapter 和 Runtime 共用协议规则；FakeLLM 也不能绕过执行前检查。"""
    if not isinstance(response, LLMResponse):
        raise LLMProtocolError("Expected an LLMResponse")
    if response.finish_reason == "length":
        raise LLMResponseTruncatedError("LLM response was truncated")
    if response.finish_reason not in ("stop", "tool_calls"):
        raise LLMProtocolError("Unsupported or missing finish_reason")
    if response.content is not None and not isinstance(response.content, str):
        raise LLMProtocolError("LLM content must be text or None")
    validate_tool_calls(response.tool_calls)
    if response.tool_calls:
        if response.finish_reason != "tool_calls":
            raise LLMProtocolError("Tool calls require finish_reason='tool_calls'")
    elif response.finish_reason != "stop":
        raise LLMProtocolError("finish_reason='tool_calls' requires tool calls")
    elif response.content is None or not response.content.strip():
        raise LLMProtocolError("LLM response has no valid final answer")
