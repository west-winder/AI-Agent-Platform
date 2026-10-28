"""无状态的 ContextBuilder V1：校验协议，按完整旧 User Turn 裁剪。"""

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from backend.agent_runtime.contracts import (
    LLMMessage,
    LLMToolDefinition,
    validate_tool_calls,
)
from backend.agent_runtime.errors import ContextBudgetExceededError, ContextInputError


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """本地估算单位的总预算；扣除输出预留和安全余量后用于输入。

    默认值是应用策略，不代表任何模型的真实上下文上限。
    输出预留只参与预算计算，不设置模型的 max_tokens。
    """

    input_budget: int = 32768
    output_reserve: int = 4096
    safety_margin: int = 1024

    def __post_init__(self) -> None:
        """拒绝非整数、负数和没有可用输入空间的配置。"""
        for value in (self.input_budget, self.output_reserve, self.safety_margin):
            if type(value) is not int or value < 0:
                raise ValueError("Context budget values must be non-negative integers")
        if self.available_input <= 0:
            raise ValueError("Context budget must leave positive input space")

    @property
    def available_input(self) -> int:
        """返回实际可用于 messages、tools 和协议开销的估算额度。"""
        return self.input_budget - self.output_reserve - self.safety_margin


def _json_bytes(value: Any) -> int:
    """每个 UTF-8 字节计一个估算单位，含 JSON 标点及转义开销。"""
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def estimate_input_tokens(
    messages: list[LLMMessage], tools: list[LLMToolDefinition]
) -> int:
    """保守近似输入成本；包含工具 schema 和协议开销，非精确 token 数。

    只计入发送给模型的字段，原始工具参数保留原文。额外计入每条消息
    16 单位、每个工具 16 单位和请求 32 单位，覆盖未展开的协议开销。
    近似估算不能严格保证模型 API 接受请求。
    """
    cost = 32
    for message in messages:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments},
                }
                for call in message.tool_calls
            ]
        cost += _json_bytes(payload) + 16
    for tool in tools:
        cost += _json_bytes({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }) + 16
    return cost


def _validate_history(history: list[LLMMessage]) -> None:
    """在裁剪前检查全部历史；工具结果必须紧随调用批次且一一对应。"""
    if not isinstance(history, list) or not history:
        raise ContextInputError("History must be a non-empty list")
    pending: set[str] = set()
    for message in history:
        if not isinstance(message, LLMMessage):
            raise ContextInputError("History requires LLMMessage records")
        if message.role not in ("system", "user", "assistant", "tool"):
            raise ContextInputError("Unsupported history message role")
        if message.content is not None and not isinstance(message.content, str):
            raise ContextInputError("History message content must be text or None")
        if not isinstance(message.tool_calls, list):
            raise ContextInputError("History tool_calls must be a list")
        if message.role != "assistant" and message.tool_calls:
            raise ContextInputError("Only assistant messages may contain tool calls")
        if message.role != "tool" and message.tool_call_id is not None:
            raise ContextInputError("Only tool messages may contain tool_call_id")
        if message.role != "assistant" and message.content is None:
            raise ContextInputError("Non-assistant messages require text content")
        if message.role == "user" and not message.content.strip():
            raise ContextInputError("User messages must contain non-empty text")
        if pending and message.role != "tool":
            raise ContextInputError("Tool call batch requires all results before continuing")
        if message.role == "tool":
            call_id = message.tool_call_id
            if not isinstance(call_id, str) or call_id not in pending:
                raise ContextInputError("Orphan, duplicate or mismatched tool result")
            pending.remove(call_id)
        elif message.tool_calls:
            validate_tool_calls(message.tool_calls)
            pending = {call.id for call in message.tool_calls}
        elif message.role == "assistant" and (
            message.content is None or not message.content.strip()
        ):
            raise ContextInputError("Assistant messages require content or tool calls")
    if pending:
        raise ContextInputError("History ends with an incomplete tool call batch")


def find_current_turn_start(history: list[LLMMessage]) -> int:
    """确定标准新请求起点；V1 不接受已开始推理的请求或断点恢复。"""
    _validate_history(history)
    if history[-1].role != "user":
        raise ContextInputError("Run input must end with the current user message")
    return len(history) - 1


class ContextBuilder:
    """只持有不可变预算配置；不持有跨 Run 的历史或裁剪状态。"""

    def __init__(self, budget: ContextBudget | None = None) -> None:
        """使用显式预算或应用默认预算创建构造器。"""
        self.budget = budget if budget is not None else ContextBudget()

    def build(
        self,
        history: list[LLMMessage],
        tools: list[LLMToolDefinition],
        current_turn_start: int,
    ) -> list[LLMMessage]:
        """保留系统消息和当前任务；从最早完整旧 Turn 开始删除。

        返回深拷贝，隔离嵌套工具参数。异常历史即使可裁剪也拒绝。
        无 User 归属的前缀及未完成的旧 Turn 保守保留。
        """
        _validate_history(history)
        if (
            type(current_turn_start) is not int
            or not 0 <= current_turn_start < len(history)
            or history[current_turn_start].role != "user"
            or any(message.role == "user" for message in history[current_turn_start + 1:])
        ):
            raise ContextInputError("current_turn_start must identify the last user message")

        context = list(history)
        if estimate_input_tokens(context, tools) <= self.budget.available_input:
            return deepcopy(context)

        starts = [i for i, message in enumerate(history) if message.role == "user"]
        removed: set[int] = set()
        for start, end in zip(starts, starts[1:]):
            # 只移除已结束的旧对话；当前 Run 后缀始终保持完整。
            turn = [i for i in range(start, end) if history[i].role != "system"]
            last = history[turn[-1]]
            if last.role != "assistant" or last.tool_calls:
                continue
            removed.update(turn)
            context = [message for i, message in enumerate(history) if i not in removed]
            if estimate_input_tokens(context, tools) <= self.budget.available_input:
                return deepcopy(context)

        raise ContextBudgetExceededError("Required context exceeds configured input budget")
