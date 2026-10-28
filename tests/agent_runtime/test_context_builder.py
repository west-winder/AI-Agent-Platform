"""ContextBuilder 的预算、完整 Turn、协议校验和隔离测试。"""

from copy import deepcopy

import pytest

from backend.agent_runtime.context_builder import (
    ContextBudget,
    ContextBuilder,
    estimate_input_tokens,
    find_current_turn_start,
)
from backend.agent_runtime.contracts import LLMMessage, LLMToolCall, LLMToolDefinition
from backend.agent_runtime.errors import (
    ContextBudgetExceededError,
    ContextInputError,
    LLMProtocolError,
)


def builder_for(messages, tools=None, *, extra=0) -> ContextBuilder:
    """按确定性样本设定恰好容纳该输入的预算。"""
    cost = estimate_input_tokens(messages, tools or []) + extra
    return ContextBuilder(ContextBudget(cost, output_reserve=0, safety_margin=0))


def tool_batch() -> list[LLMMessage]:
    """创建含两个并列调用的完整批次，结果允许交换顺序。"""
    return [
        LLMMessage("assistant", tool_calls=[
            LLMToolCall("a", "echo", '{ "text": "甲" }', {"text": "甲"}),
            LLMToolCall("b", "echo", '{"nested":[1]}', {"nested": [1]}),
        ]),
        LLMMessage("tool", '{"ok":true}', tool_call_id="b"),
        LLMMessage("tool", '{"ok":true}', tool_call_id="a"),
    ]


def test_full_history_at_exact_budget_is_deep_snapshot() -> None:
    """预算相等时不裁剪；返回消息及嵌套参数均隔离。"""
    history = [LLMMessage("system", "系统"), LLMMessage("user", "问题"), *tool_batch()]
    original = deepcopy(history)
    context = builder_for(history).build(history, [], 1)
    assert context == original
    assert context is not history
    context[2].tool_calls[1].arguments["nested"].append(2)
    context.clear()
    assert history == original


def test_trim_oldest_complete_turn_preserving_recent_turn_and_all_systems() -> None:
    """先删最早完整 Turn，系统消息即使位于旧 Turn 内也保留原序。"""
    history = [
        LLMMessage("system", "规则一"),
        LLMMessage("user", "旧问题" * 100),
        LLMMessage("system", "规则二"),
        *tool_batch(),
        LLMMessage("assistant", "旧回答"),
        LLMMessage("user", "较新问题"),
        LLMMessage("assistant", "较新回答"),
        LLMMessage("user", "当前问题"),
    ]
    original = deepcopy(history)
    expected = [history[0], history[2], *history[-3:]]
    assert builder_for(expected).build(history, [], len(history) - 1) == expected
    assert history == original


def test_current_multi_tool_batch_cannot_be_split_or_truncated() -> None:
    """必要批次全部保留；少一个单位也必须抛错。"""
    history = [LLMMessage("user", "当前问题"), *tool_batch()]
    assert builder_for(history).build(history, [], 0) == history
    with pytest.raises(ContextBudgetExceededError):
        builder_for(history, extra=-1).build(history, [], 0)


def test_remove_all_old_turns_and_preserve_current_chain() -> None:
    """预算要求时删除多个旧 Turn，完整保留当前多轮工具链。"""
    current = [LLMMessage("user", "当前"), *tool_batch(), *tool_batch()]
    old = [LLMMessage("user", "历史"), LLMMessage("assistant", "回答")]
    history = [*old, *old, *current]
    assert builder_for(current).build(history, [], 4) == current


def test_tool_schema_counts_towards_budget_and_failure_does_not_leak_content() -> None:
    """工具 schema 也占额度，异常不回显消息或 schema。"""
    history = [LLMMessage("user", "private-question")]
    tools = [LLMToolDefinition("echo", "private-schema" * 100, {"type": "object"})]
    builder = builder_for(history)
    with pytest.raises(ContextBudgetExceededError) as raised:
        builder.build(history, tools, 0)
    assert "private" not in str(raised.value)
    assert builder_for(history, tools).build(history, tools, 0) == history


def test_output_reserve_and_safety_margin_reduce_available_input() -> None:
    """预留在构造请求前生效，零预留也可配置。"""
    history = [LLMMessage("user", "问题")]
    cost = estimate_input_tokens(history, [])
    exact = ContextBudget(cost + 30, output_reserve=20, safety_margin=10)
    assert exact.available_input == cost
    assert ContextBuilder(exact).build(history, [], 0) == history
    with pytest.raises(ContextBudgetExceededError):
        ContextBuilder(ContextBudget(cost + 29, 20, 10)).build(history, [], 0)


@pytest.mark.parametrize("kwargs", [
    {"input_budget": 0}, {"input_budget": -1}, {"input_budget": True},
    {"input_budget": 1.5}, {"output_reserve": -1}, {"safety_margin": True},
    {"input_budget": 10, "output_reserve": 10, "safety_margin": 0},
])
def test_invalid_budget_rejected(kwargs) -> None:
    """无效配置立即失败。"""
    with pytest.raises(ValueError):
        ContextBudget(**kwargs)


def test_utf8_raw_arguments_and_protocol_overhead_are_counted() -> None:
    """中文按 UTF-8 字节计费，原始 JSON 空白保留并计费。"""
    ascii_cost = estimate_input_tokens([LLMMessage("user", "a")], [])
    chinese_cost = estimate_input_tokens([LLMMessage("user", "中")], [])
    assert chinese_cost == ascii_cost + 2
    batch = tool_batch()
    original_cost = estimate_input_tokens(batch, [])
    call = batch[0].tool_calls[0]
    batch[0].tool_calls[0] = LLMToolCall(
        call.id, call.name, call.raw_arguments + " " * 50, call.arguments
    )
    assert estimate_input_tokens(batch, []) == original_cost + 50
    assert ascii_cost > len("a".encode("utf-8"))


@pytest.mark.parametrize("history", [
    [], [LLMMessage("system", "系统")], [LLMMessage("user", "  ")],
    [LLMMessage("user", None)], [LLMMessage("user", "问题"), LLMMessage("assistant", "答案")],
    [LLMMessage("user", "问题"), *tool_batch()],
])
def test_run_start_rejects_missing_blank_or_already_started_request(history) -> None:
    """V1 只接受以有效新 User Message 结束的请求。"""
    with pytest.raises(ContextInputError):
        find_current_turn_start(history)


def test_last_user_start_and_complete_unowned_prefix_supported() -> None:
    """起点定位最后 User，兼容原有完整工具前缀并保护它。"""
    history = [*tool_batch(), LLMMessage("user", "继续")]
    assert find_current_turn_start(history) == 3
    assert builder_for(history).build(history, [], 3) == history
    with pytest.raises(ContextBudgetExceededError):
        builder_for(history[-1:]).build(history, [], 3)


@pytest.mark.parametrize("start", [-1, 0, 1, True, None, 0.0, 3])
def test_builder_rejects_wrong_or_stale_start(start) -> None:
    """起点必须是最后一条 User 的整数索引。"""
    history = [LLMMessage("user", "旧"), LLMMessage("assistant", "答"), LLMMessage("user", "新")]
    with pytest.raises(ContextInputError):
        ContextBuilder().build(history, [], start)


@pytest.mark.parametrize("bad_prefix", [
    [LLMMessage("tool", "{}", tool_call_id="orphan")],
    tool_batch()[:1],
    tool_batch()[:2],
    [*tool_batch(), LLMMessage("tool", "{}", tool_call_id="a")],
    [tool_batch()[0], LLMMessage("tool", "{}", tool_call_id="wrong")],
    [tool_batch()[0], LLMMessage("system", "插入"), *tool_batch()[1:]],
    [LLMMessage("user", "非法", tool_batch()[0].tool_calls)],
    [LLMMessage("assistant", "非法", tool_call_id="a")],
    [LLMMessage("assistant")],
    [LLMMessage("invalid", "非法")],
    [LLMMessage("user", 12)],
    [LLMMessage("assistant", tool_calls=[tool_batch()[0].tool_calls[0]] * 2)],
    ["not-a-message"],
])
def test_malformed_history_rejected_even_if_old_turn_would_be_trimmed(bad_prefix) -> None:
    """不通过裁剪掩盖孤立、重复、缺失或错位的工具协议错误。"""
    history = [LLMMessage("user", "旧请求"), *bad_prefix,
               LLMMessage("assistant", "旧答"), LLMMessage("user", "新请求")]
    with pytest.raises(LLMProtocolError):
        ContextBuilder().build(history, [], len(history) - 1)


def test_incomplete_old_turn_is_protected() -> None:
    """只有结果但没有最终回答的旧 Turn 不作为已完成对话删除。"""
    history = [LLMMessage("user", "旧问题"), *tool_batch(), LLMMessage("user", "新问题")]
    with pytest.raises(ContextBudgetExceededError):
        builder_for(history[-1:]).build(history, [], len(history) - 1)


def test_system_content_alone_can_exhaust_required_budget() -> None:
    """不能为了适配预算删除旧 Turn 内的系统规则。"""
    history = [LLMMessage("user", "旧"), LLMMessage("system", "规则" * 100),
               LLMMessage("assistant", "旧答"), LLMMessage("user", "新")]
    with pytest.raises(ContextBudgetExceededError):
        builder_for(history[-1:]).build(history, [], 3)
