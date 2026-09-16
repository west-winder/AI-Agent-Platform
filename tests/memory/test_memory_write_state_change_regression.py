"""
Memory Write - State Change Semantic Contract Tests

类型：

[KEEP]

背景：

Memory Lifecycle C++ Negation Bug。

真实现象：

    Existing Memory:
        ID 9  用户最近在学C++   status=current

    用户输入:
        我没学C++了
        我最近没学C++了

    实际结果:
        ID 9 仍为 current，没有新增 Memory。

根因（已由 Runtime Trace 确认，不在本测试范围内重复验证）：

    Extractor / Validator 的 Memory 定义
    停留在 Lifecycle 之前的语义：

        Memory = 长期稳定、不容易变化的信息

    于是：

        "用户不再学习C++"

    被 Extractor 当成"没有值得长期保存的信息"丢弃，
    或被 Validator 当成"可能变化的临时状态"判 valid=False。

    而 Lifecycle 实际需要的是：

        Memory = 对用户当前状态有意义、
                 未来可能被转成 historical 的状态信息。

本测试保护的内容：

    Extractor / Validator 的 System Prompt
    必须显式覆盖以下 Semantic Contract：

        1. 有持续意义的 Current State
        2. State Change / 状态变化
        3. State Termination / 状态终止（停止 / 不再）
        4. State Transition / 状态转移（转向）
        5. State Restart / 状态重启（重新开始）
        6. "可能变化"不是自动拒绝理由
        7. 否定句表达新状态，不是"没有信息"
        8. 一次性事件（One-off Event）仍然必须排除

    同时保护：

        9.  Validator 不得承担
            duplicate / conflict / related / new 的判断

        10. Validator 只能看到 Candidate 自身，
            不得获得 Existing Memories

设计约束：

    1. deterministic
    2. 不调用真实 DeepSeek
    3. 不联网
    4. 不写数据库
    5. 不加载 Embedding 模型

    全部通过 monkeypatch 替换 call_llm 实现。

重要：

    本测试只做 Semantic Contract 级别的
    关键字包含校验，

    不 assert 整段 Prompt 完全相等。

    目的是：

        允许后续继续润色 Prompt 措辞，
        但不允许把关键语义删掉。

运行：

    python tests/memory/test_memory_write_state_change_regression.py
"""

import asyncio
import sys
from pathlib import Path

from unittest.mock import patch


# ============================================================
# Project Root Bootstrap
#
# 与项目现有测试保持一致：
# 允许从任意目录运行本文件。
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from backend.memory.memory_write.memory_extractor import (  # noqa: E402
    MemoryExtractor,
)

from backend.memory.memory_write.memory_validator import (  # noqa: E402
    MemoryValidator,
)

from backend.schemas.memory_candidate import (  # noqa: E402
    MemoryCandidate,
)


# ============================================================
# call_llm Patch 目标
#
# 两个模块都在模块级直接 import call_llm，
# 因此分别 patch 各自模块的引用。
# ============================================================

EXTRACTOR_CALL_LLM = (
    "backend.memory.memory_write."
    "memory_extractor.call_llm"
)

VALIDATOR_CALL_LLM = (
    "backend.memory.memory_write."
    "memory_validator.call_llm"
)


# ============================================================
# Helpers
# ============================================================

def run(coro):
    """
    在同步测试中真实执行并等待
    async production contract。

    背景：

    MemoryExtractor.extract /
    MemoryValidator.validate

    都已经变成 async Contract。

    本模块的测试函数保持同步 def test_xxx()：

        1. 直接 python 运行时会真的执行
        2. pytest 会原生收集执行，
           不会被当成 async test 静默跳过
        3. coroutine 一定被 await，
           不会出现 coroutine was never awaited
        4. 不会产生假 PASS

    这里只负责把 coroutine 驱动到底，
    不参与任何业务断言。
    """

    return asyncio.run(coro)


def run_extractor(
    user_message: str,
    llm_response: str,
):
    """
    用 Fake call_llm 运行真实 MemoryExtractor。

    返回：
        (captured, candidates)
    """

    captured = {}

    async def fake_call_llm(messages, *args, **kwargs):

        captured["messages"] = messages

        return llm_response

    with patch(
        EXTRACTOR_CALL_LLM,
        side_effect=fake_call_llm,
    ):

        candidates = run(
            MemoryExtractor().extract(
                user_message
            )
        )

    return captured, candidates


def run_validator(
    candidate,
    llm_response: str,
):
    """
    用 Fake call_llm 运行真实 MemoryValidator。

    返回：
        (captured, validation_result)
    """

    captured = {}

    async def fake_call_llm(messages, *args, **kwargs):

        captured["messages"] = messages

        return llm_response

    with patch(
        VALIDATOR_CALL_LLM,
        side_effect=fake_call_llm,
    ):

        result = run(
            MemoryValidator().validate(
                candidate
            )
        )

    return captured, result


def system_prompt_of(captured):
    """
    从捕获的 messages 中取出 system prompt。
    """

    messages = captured.get(
        "messages"
    )

    assert isinstance(
        messages,
        list
    ), "没有捕获到 messages"

    for message in messages:

        if message.get(
            "role"
        ) == "system":

            return message.get(
                "content",
                ""
            )

    raise AssertionError(
        "没有捕获到 system prompt"
    )


def user_prompt_of(captured):

    messages = captured.get(
        "messages"
    )

    assert isinstance(
        messages,
        list
    ), "没有捕获到 messages"

    for message in messages:

        if message.get(
            "role"
        ) == "user":

            return message.get(
                "content",
                ""
            )

    raise AssertionError(
        "没有捕获到 user prompt"
    )


def assert_prompt_contains(
    prompt: str,
    tokens,
    label: str,
):
    """
    Semantic Contract 级校验：

    只校验关键语义是否存在，
    不校验整段 Prompt 是否完全相等。
    """

    missing = [
        token
        for token in tokens
        if token not in prompt
    ]

    assert not missing, (
        f"{label} Prompt 缺少关键语义："
        f"{missing}"
    )

    print(
        f"[PASS] {label} Prompt 覆盖 "
        f"{len(tokens)} 项关键语义"
    )


# ============================================================
# 1. Extractor Prompt Semantic Contract
# ============================================================

def test_extractor_prompt_covers_state_semantics():
    """
    Contract：

    Extractor Prompt 必须显式覆盖：

        Current State
        状态变化 / State Change
        状态终止（停止 / 不再）
        状态转移（转向）
        状态重启（重新开始）
        持续状态
        可能变化 ≠ 不值得保存
    """

    captured, _ = run_extractor(
        user_message="我最近没学C++了",
        llm_response='{"memories": []}',
    )

    prompt = system_prompt_of(
        captured
    )

    assert_prompt_contains(
        prompt,
        [
            "Current State",
            "State Change",
            "状态变化",
            "状态终止",
            "停止",
            "不再",
            "转向",
            "状态重启",
            "重新开始",
            "持续状态",
            "可能变化",
            "不值得保存",
        ],
        "Extractor",
    )


def test_extractor_prompt_still_excludes_one_off_event():
    """
    Contract：

    修正不能矫枉过正。

    Extractor Prompt 仍必须排除：

        一次性事件（One-off Event）

    并且必须给出
    "一次性事件 vs 持续状态"的区分标准。
    """

    captured, _ = run_extractor(
        user_message="我今天下午学了两个小时 C++",
        llm_response='{"memories": []}',
    )

    prompt = system_prompt_of(
        captured
    )

    assert_prompt_contains(
        prompt,
        [
            "一次性事件",
            "One-off Event",
            "瞬时动作",
            "持续状态",
        ],
        "Extractor 排除项",
    )


# ============================================================
# 2. Validator Prompt Semantic Contract
# ============================================================

def test_validator_prompt_may_change_is_not_auto_reject():
    """
    Contract：

    "可能变化"不得再被当作
    自动拒绝理由。

    这是本次 Bug 的核心修正点之一。
    """

    candidate = MemoryCandidate(
        content="用户不再学习C++了",
        memory_type="fact",
    )

    captured, _ = run_validator(
        candidate,
        '{"valid": true, "reason": "ok"}',
    )

    prompt = system_prompt_of(
        captured
    )

    assert_prompt_contains(
        prompt,
        [
            "可能变化",
            "不是自动拒绝理由",
            "否定句",
        ],
        "Validator 可能变化",
    )

    # --------------------------------------------------
    # 额外校验：
    #
    # "可能变化"附近必须是澄清语义，
    # 不能出现在"应该拒绝"清单里。
    # --------------------------------------------------

    assert (
        "不是自动拒绝理由" in prompt
    ), (
        "Validator Prompt 必须明确写出 "
        "「可能变化」不是自动拒绝理由"
    )


def test_validator_prompt_covers_state_semantics():
    """
    Contract：

    Validator Prompt 必须显式覆盖：

        Current State
        状态变化
        状态终止
        状态转移
        状态重启
        持续状态
        否定句 = 新状态（不再 / 停止）
    """

    candidate = MemoryCandidate(
        content="用户从Python转向Java",
        memory_type="fact",
    )

    captured, _ = run_validator(
        candidate,
        '{"valid": true, "reason": "ok"}',
    )

    prompt = system_prompt_of(
        captured
    )

    assert_prompt_contains(
        prompt,
        [
            "Current State",
            "状态变化",
            "状态终止",
            "状态转移",
            "状态重启",
            "持续状态",
            "不再",
            "停止",
            "转向",
        ],
        "Validator 状态语义",
    )


def test_validator_prompt_still_rejects_one_off_event():
    """
    Contract：

    修正不能矫枉过正。

    一次性事件仍必须可拒绝。
    """

    candidate = MemoryCandidate(
        content="用户今天下午学了两个小时C++",
        memory_type="fact",
    )

    captured, _ = run_validator(
        candidate,
        '{"valid": false, "reason": "一次性事件"}',
    )

    prompt = system_prompt_of(
        captured
    )

    assert_prompt_contains(
        prompt,
        [
            "一次性事件",
            "One-off Event",
            "瞬时动作",
            "持续状态",
        ],
        "Validator 排除项",
    )


def test_validator_prompt_declares_no_relationship_duty():
    """
    Contract：

    Validator 不得承担
    Relationship Judge 的职责。

    Prompt 必须显式声明
    不判断 duplicate / conflict / related / new。
    """

    candidate = MemoryCandidate(
        content="用户不再学习C++了",
        memory_type="fact",
    )

    captured, _ = run_validator(
        candidate,
        '{"valid": true, "reason": "ok"}',
    )

    prompt = system_prompt_of(
        captured
    )

    assert_prompt_contains(
        prompt,
        [
            "duplicate",
            "conflict",
            "related",
            "new",
        ],
        "Validator 职责边界",
    )

    assert (
        "不得据此判 valid=false" in prompt
        or "不得据此判 valid=false" in prompt
    ), (
        "Validator Prompt 必须声明："
        "即使与已有 Memory 矛盾，"
        "也不得据此判 valid=false"
    )


# ============================================================
# 3. Visibility Contract
# ============================================================

def test_extractor_only_receives_user_message():
    """
    Contract：

    Extractor 只应收到：

        system prompt
        +
        user message

    不得注入任何 Existing Memory。
    """

    user_message = "我最近没学C++了"

    captured, _ = run_extractor(
        user_message=user_message,
        llm_response='{"memories": []}',
    )

    messages = captured["messages"]

    assert len(messages) == 2, (
        "Extractor 应当只收到 system + user "
        f"两条消息，实际 {len(messages)} 条"
    )

    assert (
        user_prompt_of(captured) == user_message
    ), "Extractor user prompt 应当是原始 user message"

    print(
        "[PASS] Extractor 只收到 "
        "system + user 两条消息"
    )


def test_validator_only_receives_candidate():
    """
    Contract：

    Validator 只应看到 Candidate 自身
    （content + memory_type）。

    不得获得 Existing Memories。
    """

    candidate = MemoryCandidate(
        content="用户不再学习C++了",
        memory_type="fact",
    )

    captured, _ = run_validator(
        candidate,
        '{"valid": true, "reason": "ok"}',
    )

    messages = captured["messages"]

    assert len(messages) == 2, (
        "Validator 应当只收到 system + user "
        f"两条消息，实际 {len(messages)} 条"
    )

    user_prompt = user_prompt_of(
        captured
    )

    assert (
        candidate.content in user_prompt
    ), "Validator user prompt 应包含 Candidate content"

    assert (
        candidate.memory_type in user_prompt
    ), "Validator user prompt 应包含 Candidate memory_type"

    # --------------------------------------------------
    # 不得出现 Existing Memory 注入痕迹
    # --------------------------------------------------

    forbidden = [
        "Existing Memory",
        "Existing Memories",
        "existing_memories",
        "similarity",
    ]

    leaked = [
        token
        for token in forbidden
        if token in user_prompt
    ]

    assert not leaked, (
        "Validator user prompt 不应包含 "
        f"Existing Memory 相关信息：{leaked}"
    )

    print(
        "[PASS] Validator 只看到 Candidate 自身，"
        "未获得 Existing Memories"
    )


# ============================================================
# 4. JSON Parsing Contract（Fake LLM 固定返回）
# ============================================================

def test_extractor_parses_fixed_valid_json():
    """
    Contract：

    Fake LLM 返回固定合法 JSON 时，
    MemoryExtractor 必须正确解析为
    MemoryCandidate。

    这里刻意使用"状态终止"语义，
    因为这正是本次 Bug 的场景。
    """

    response = (
        '{"memories": ['
        '{'
        '"content": "用户不再学习C++", '
        '"memory_type": "fact"'
        '}'
        ']}'
    )

    _, candidates = run_extractor(
        user_message="我没学C++了",
        llm_response=response,
    )

    assert len(candidates) == 1, (
        f"应当解析出 1 个 Candidate，"
        f"实际 {len(candidates)}"
    )

    candidate = candidates[0]

    assert isinstance(
        candidate,
        MemoryCandidate
    ), "返回值应当是 MemoryCandidate"

    assert (
        candidate.content == "用户不再学习C++"
    ), f"content 不符：{candidate.content}"

    assert (
        candidate.memory_type == "fact"
    ), f"memory_type 不符：{candidate.memory_type}"

    print(
        "[PASS] Extractor 正确解析 "
        "State Termination Candidate"
    )


def test_extractor_fails_closed_on_invalid_json():
    """
    Contract：

    LLM 返回非法 JSON 时，
    Extractor 必须返回空列表（Fail Closed），
    不得抛异常。
    """

    _, candidates = run_extractor(
        user_message="我没学C++了",
        llm_response="这不是JSON",
    )

    assert candidates == [], (
        "非法 JSON 应当返回空列表"
    )

    print(
        "[PASS] Extractor 对非法 JSON "
        "Fail Closed 返回 []"
    )


def test_extractor_skips_invalid_memory_type():
    """
    Contract：

    memory_type 不在白名单时，
    该 Candidate 必须被丢弃。
    """

    response = (
        '{"memories": ['
        '{'
        '"content": "用户不再学习C++", '
        '"memory_type": "unknown_type"'
        '}'
        ']}'
    )

    _, candidates = run_extractor(
        user_message="我没学C++了",
        llm_response=response,
    )

    assert candidates == [], (
        "非法 memory_type 的 Candidate "
        "必须被丢弃"
    )

    print(
        "[PASS] Extractor 丢弃非法 memory_type"
    )


def test_validator_parses_valid_true():
    """
    Contract：

    Fake LLM 返回 valid=true 时，
    MemoryValidator 必须解析为
    valid=True 的 MemoryValidationResult。

    重点：

        "状态终止"类 Candidate
        在 Validator 层不得被硬编码拒绝。
    """

    candidate = MemoryCandidate(
        content="用户不再学习C++了",
        memory_type="fact",
    )

    _, result = run_validator(
        candidate,
        '{"valid": true, '
        '"reason": "用户学习状态发生变化"}',
    )

    assert result.valid is True, (
        "State Termination Candidate "
        "应当可以通过 Validator"
    )

    assert (
        result.reason
        == "用户学习状态发生变化"
    ), f"reason 不符：{result.reason}"

    print(
        "[PASS] Validator 允许 "
        "State Termination Candidate 通过"
    )


def test_validator_parses_valid_false():
    """
    Contract：

    Fake LLM 返回 valid=false 时，
    必须正确解析，
    并且保留 reason。

    这保证"一次性事件仍可拒绝"的通道完好。
    """

    candidate = MemoryCandidate(
        content="用户今天下午学了两个小时C++",
        memory_type="fact",
    )

    _, result = run_validator(
        candidate,
        '{"valid": false, '
        '"reason": "一次性事件"}',
    )

    assert result.valid is False

    assert (
        result.reason == "一次性事件"
    ), f"reason 不符：{result.reason}"

    print(
        "[PASS] Validator 仍可拒绝 "
        "一次性事件 Candidate"
    )


def test_validator_rule_layer_rejects_too_short():
    """
    Contract：

    Rule Validation 层
    （content 长度 < 5）
    必须在不调用 LLM 的情况下
    直接返回 valid=False。
    """

    candidate = MemoryCandidate(
        content="C++",
        memory_type="fact",
    )

    captured, result = run_validator(
        candidate,
        '{"valid": true, "reason": "不应被调用"}',
    )

    assert result.valid is False

    assert (
        "messages" not in captured
    ), "Rule 层失败时不应调用 LLM"

    print(
        "[PASS] Validator Rule 层 "
        "短路且不调用 LLM"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "MEMORY WRITE STATE CHANGE "
        "REGRESSION TESTS [KEEP]"
    )
    print("=" * 70)
    print()

    # --------------------------------------------------
    # Extractor Prompt
    # --------------------------------------------------

    test_extractor_prompt_covers_state_semantics()
    test_extractor_prompt_still_excludes_one_off_event()

    # --------------------------------------------------
    # Validator Prompt
    # --------------------------------------------------

    test_validator_prompt_may_change_is_not_auto_reject()
    test_validator_prompt_covers_state_semantics()
    test_validator_prompt_still_rejects_one_off_event()
    test_validator_prompt_declares_no_relationship_duty()

    # --------------------------------------------------
    # Visibility
    # --------------------------------------------------

    test_extractor_only_receives_user_message()
    test_validator_only_receives_candidate()

    # --------------------------------------------------
    # JSON Parsing
    # --------------------------------------------------

    test_extractor_parses_fixed_valid_json()
    test_extractor_fails_closed_on_invalid_json()
    test_extractor_skips_invalid_memory_type()

    test_validator_parses_valid_true()
    test_validator_parses_valid_false()
    test_validator_rule_layer_rejects_too_short()

    print()
    print("=" * 70)
    print(
        "ALL MEMORY WRITE STATE CHANGE "
        "REGRESSION TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()
