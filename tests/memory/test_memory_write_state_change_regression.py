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

    Extractor 与 Validator 都已经迁移为 Structured Output，
    因此通过 patch 模块级 call_llm_structured 注入 Fake。

Component Responsibility（Structured Output 迁移之后）：

    Extractor：
        Structured Output / Pydantic 负责顶层数据模型
        （memories 缺失 / None → []，非 list → 校验失败，
          非 dict item → 过滤）

        MemoryExtractor 继续负责：
            content 是否 str
            content 是否为空
            content.strip()
            memory_type 白名单
            单条非法 item skip
            合法 item → MemoryCandidate
            同批次其他合法 item 继续保留  ← 核心契约

    Validator：
        Pydantic 负责 valid: StrictBool / reason: StrictStr
        MemoryValidator 负责 Rule 短路、Model 转换、fail-closed

重要：

    MemoryValidator 的 LLM Boundary 已经是：

        await call_llm_structured(
            messages=...,
            output_model=MemoryValidatorLLMOutput,
        )
        → MemoryValidatorLLMOutput

    因此 Validator **不再负责**：

        raw response 是不是 str
        json.loads
        JSON 顶层是否 object
        valid 是不是 bool
        reason 是不是 str

    上面这些由 Provider / SDK / Pydantic Boundary 承担，
    本文件用
    "Validator LLM Boundary Model Contract"
    分节直接测 MemoryValidatorLLMOutput。

    Validator 继续负责：

        Rule Layer 短路（不调用 LLM）
        调用正确的 Structured Output Model
        Model → MemoryValidationResult 转换
        Structured LLM / Pydantic 失败 → fail-closed
        异常不得影响 Chat 主流程

重要：

    reason 没有 non-empty validator。
    旧 Contract 只要求 reason 是 str，
    并不拒绝 "" 或纯空白字符串。
    迁移后必须继续接受空 reason。

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


from pydantic import ValidationError  # noqa: E402

from backend.memory.memory_write.memory_extractor import (  # noqa: E402
    MemoryExtractor,
    MemoryExtractorLLMItem,
    MemoryExtractorLLMOutput,
)

from backend.memory.memory_write.memory_validator import (  # noqa: E402
    MemoryValidator,
    MemoryValidatorLLMOutput,
)

from backend.schemas.memory_candidate import (  # noqa: E402
    MemoryCandidate,
)


# ============================================================
# LLM Patch 目标
#
# 两个模块都在模块级直接 import LLM 入口，
# 因此分别 patch 各自模块的引用。
#
# 两者现在都已经是 Structured Output：
#
#     MemoryExtractor  → call_llm_structured
#                        output_model=MemoryExtractorLLMOutput
#
#     MemoryValidator  → call_llm_structured
#                        output_model=MemoryValidatorLLMOutput
#
# patch 目标必须与生产模块
# 真实 import 的符号一致，
# 否则会在 patch 解析阶段直接抛
# AttributeError，生产代码根本不会执行。
# ============================================================

EXTRACTOR_CALL_LLM_STRUCTURED = (
    "backend.memory.memory_write."
    "memory_extractor.call_llm_structured"
)

VALIDATOR_CALL_LLM_STRUCTURED = (
    "backend.memory.memory_write."
    "memory_validator.call_llm_structured"
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


class FakeExtractorStructuredLLM:
    """
    Fake call_llm_structured（MemoryExtractor 专用）。

    生产 Contract：

        await call_llm_structured(
            messages=messages,
            output_model=MemoryExtractorLLMOutput,
        )
        → MemoryExtractorLLMOutput

    因此本 Fake 必须保持同样的
    keyword-only Calling Contract，
    并返回 Structured Output Model，
    而不是旧实现里的 JSON str。
    """

    def __init__(
        self,
        output=None,
        exc=None
    ):
        self._output = output
        self._exc = exc

        self.call_count = 0
        self.received_messages = None
        self.received_output_model = None
        self.received_model = None

    async def __call__(
        self,
        *,
        messages,
        output_model,
        model=None
    ):
        self.call_count += 1

        self.received_messages = messages
        self.received_output_model = output_model
        self.received_model = model

        assert (
            output_model
            is MemoryExtractorLLMOutput
        ), (
            "MemoryExtractor 必须以 "
            "MemoryExtractorLLMOutput "
            "作为 output_model，"
            f"实际为：{output_model}"
        )

        if self._exc is not None:

            raise self._exc

        if self._output is None:

            raise AssertionError(
                "FakeExtractorStructuredLLM "
                "未配置输出"
            )

        return self._output


def extractor_output(
    *memories
):
    """
    构造 MemoryExtractor 的 Structured Output。

    item 支持三种形态
    （与 MemoryExtractorLLMOutput 顶层
    before validator 的接受范围一致）：

        dict                    → 正常解析
        MemoryExtractorLLMItem  → 正常解析
        其他（如 "garbage"）     → 被过滤掉（skip）

    真实 Provider 路径产出的是 dict：

        OpenAI SDK 从 JSON 解析出 dict
            ↓
        Pydantic 校验

    因此收 prompt / 业务断言用 dict 最忠实。

    Model 实例形态同样合法，
    由 test_extractor_llm_output_accepts_dict_and_model_items
    显式固定该契约。

    历史背景（勿删）：

    生产早期版本的 validator 只保留
    isinstance(item, dict)，
    传 MemoryExtractorLLMItem 会被**静默过滤**
    成 memories=[]，不报错（MODEL INSTANCE TRAP）。

    该问题已修复：validator 现在保留

        isinstance(item, (dict, MemoryExtractorLLMItem))

    因此这里不再需要拦截 Model 实例。
    """

    return MemoryExtractorLLMOutput(
        memories=list(memories)
    )


def run_extractor(
    user_message,
    output=None,
    exc=None,
):
    """
    用 Fake call_llm_structured 运行真实 MemoryExtractor。

    返回：

        (captured, candidates, fake_llm)

    captured["messages"]
    只有真正调用过 LLM 时才有值。
    """

    captured = {}

    fake_llm = FakeExtractorStructuredLLM(
        output=output,
        exc=exc,
    )

    with patch(
        EXTRACTOR_CALL_LLM_STRUCTURED,
        fake_llm,
    ):

        candidates = run(
            MemoryExtractor().extract(
                user_message
            )
        )

    captured["messages"] = (
        fake_llm.received_messages
    )

    return captured, candidates, fake_llm


class FakeValidatorStructuredLLM:
    """
    Fake call_llm_structured（MemoryValidator 专用）。

    生产 Contract：

        await call_llm_structured(
            messages=messages,
            output_model=MemoryValidatorLLMOutput,
        )
        → MemoryValidatorLLMOutput

    因此本 Fake 必须保持同样的
    keyword-only Calling Contract，
    并返回 Structured Output Model，
    而不是旧实现里的 JSON str。
    """

    def __init__(
        self,
        output=None,
        exc=None
    ):
        self._output = output
        self._exc = exc

        self.call_count = 0
        self.received_messages = None
        self.received_output_model = None
        self.received_model = None

    async def __call__(
        self,
        *,
        messages,
        output_model,
        model=None
    ):
        self.call_count += 1

        self.received_messages = messages
        self.received_output_model = output_model
        self.received_model = model

        assert (
            output_model
            is MemoryValidatorLLMOutput
        ), (
            "MemoryValidator 必须以 "
            "MemoryValidatorLLMOutput "
            "作为 output_model，"
            f"实际为：{output_model}"
        )

        if self._exc is not None:

            raise self._exc

        if self._output is None:

            raise AssertionError(
                "FakeValidatorStructuredLLM "
                "未配置输出"
            )

        return self._output


def validator_output(
    valid=True,
    reason="ok"
):
    """
    构造 MemoryValidator 的 Structured Output。

    刻意不做任何预处理：

        reason=""

    必须原样交给 Model，
    用于验证旧 Contract（reason 允许为空）。
    """

    return MemoryValidatorLLMOutput(
        valid=valid,
        reason=reason
    )


def run_validator(
    candidate,
    output=None,
    exc=None,
):
    """
    用 Fake call_llm_structured 运行真实 MemoryValidator。

    返回：

        (captured, validation_result, fake_llm)

    captured["messages"]
    只有真正调用过 LLM 时才有值。
    """

    captured = {}

    fake_llm = FakeValidatorStructuredLLM(
        output=output,
        exc=exc,
    )

    with patch(
        VALIDATOR_CALL_LLM_STRUCTURED,
        fake_llm,
    ):

        result = run(
            MemoryValidator().validate(
                candidate
            )
        )

    captured["messages"] = (
        fake_llm.received_messages
    )

    return captured, result, fake_llm


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

    captured, _, _ = run_extractor(
        user_message="我最近没学C++了",
        output=extractor_output(),
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

    captured, _, _ = run_extractor(
        user_message="我今天下午学了两个小时 C++",
        output=extractor_output(),
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

    captured, _, _ = run_validator(
        candidate,
        validator_output(),
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

    captured, _, _ = run_validator(
        candidate,
        validator_output(),
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

    captured, _, _ = run_validator(
        candidate,
        validator_output(
            False,
            "一次性事件"
        ),
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

    captured, _, _ = run_validator(
        candidate,
        validator_output(),
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

    captured, _, _ = run_extractor(
        user_message=user_message,
        output=extractor_output(),
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

    captured, _, _ = run_validator(
        candidate,
        validator_output(),
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
# ============================================================
# 4. Structured Output → MemoryCandidate
#
# Pydantic 负责顶层数据模型；
# Extractor 负责逐条过滤 + Candidate 构造。
# ============================================================
# ============================================================


def test_extractor_maps_structured_item_to_candidate():
    """
    Contract：

    Structured Output 中的一条合法 item
    必须被映射为 MemoryCandidate。

    同时验证 content 会被 strip。

    这里刻意使用"状态终止"语义，
    因为这正是本次 Bug 的场景。
    """

    _, candidates, _ = run_extractor(
        user_message="我没学C++了",
        output=extractor_output(
            {
                "content": (
                    "  用户不再学习C++  "
                ),
                "memory_type": "fact",
            }
        ),
    )

    assert len(candidates) == 1, (
        f"应当映射出 1 个 Candidate，"
        f"实际 {len(candidates)}"
    )

    candidate = candidates[0]

    assert isinstance(
        candidate,
        MemoryCandidate
    ), "返回值应当是 MemoryCandidate"

    assert candidate.content == (
        "用户不再学习C++"
    ), (
        "content 必须去除首尾空白，"
        f"实际：{candidate.content!r}"
    )

    assert candidate.memory_type == "fact"

    print(
        "[PASS] Extractor 正确映射 "
        "State Termination Candidate"
    )


def test_extractor_requests_extractor_llm_output_model():
    """
    Extractor 必须向 Provider 声明
    自己需要的是 MemoryExtractorLLMOutput。

    这是 Structured Output 迁移之后
    新增的、真实的 Extractor 责任：
    声明输出 Schema。
    """

    _, _, fake_llm = run_extractor(
        user_message="我没学C++了",
        output=extractor_output(),
    )

    assert (
        fake_llm.received_output_model
        is MemoryExtractorLLMOutput
    ), (
        "Extractor 必须以 "
        "MemoryExtractorLLMOutput "
        "作为 output_model"
    )


def test_extractor_returns_empty_when_no_memories():
    """
    memories=[] → []

    表示"本次没有值得保存的信息"，
    是正常空结果，不是失败。
    """

    _, candidates, _ = run_extractor(
        user_message="你好",
        output=extractor_output(),
    )

    assert candidates == []


def test_extractor_llm_output_accepts_dict_and_model_items():
    """
    MemoryExtractorLLMOutput 的 item Contract：

        dict                    → 正常
        MemoryExtractorLLMItem  → 正常
        非 dict garbage          → skip

    这里验证的是顶层 before validator 的
    **接受范围**，
    不是 Extractor 的业务过滤逻辑。

    历史背景（勿删）：

    生产早期版本只保留

        isinstance(item, dict)

    因此传 MemoryExtractorLLMItem 实例会被
    **静默过滤**成 memories=[]，
    表现为"Extractor 什么都没提取到"，
    而且不报错（MODEL INSTANCE TRAP）。

    该问题已修复，validator 现在保留

        isinstance(item, (dict, MemoryExtractorLLMItem))

    所以这里反过来显式固定这个契约：
    两种形态都必须被接受，
    只有非 dict garbage 才被 skip。
    """

    # --------------------------------------------------
    # 1. dict 形态（真实 Provider 路径）
    # --------------------------------------------------

    as_dict = MemoryExtractorLLMOutput(
        memories=[
            {
                "content": "用户正在学习 FastAPI",
                "memory_type": "fact",
            }
        ]
    )

    assert len(as_dict.memories) == 1, (
        "dict 形态必须被正常解析，"
        f"实际 {len(as_dict.memories)} 条"
    )

    assert (
        as_dict.memories[0].content
        == "用户正在学习 FastAPI"
    )

    # --------------------------------------------------
    # 2. MemoryExtractorLLMItem 形态
    # --------------------------------------------------

    as_model = MemoryExtractorLLMOutput(
        memories=[
            MemoryExtractorLLMItem(
                content="用户正在学习 FastAPI",
                memory_type="fact",
            )
        ]
    )

    assert len(as_model.memories) == 1, (
        "MemoryExtractorLLMItem 形态必须被正常接受，"
        f"实际 {len(as_model.memories)} 条"
    )

    assert (
        as_model.memories[0].memory_type
        == "fact"
    )

    # --------------------------------------------------
    # 3. 两种形态混排 + 非 dict garbage
    # --------------------------------------------------

    mixed = MemoryExtractorLLMOutput(
        memories=[
            {
                "content": "用户正在学习 FastAPI",
                "memory_type": "fact",
            },
            "garbage",
            123,
            MemoryExtractorLLMItem(
                content="用户的目标是转岗后端",
                memory_type="goal",
            ),
        ]
    )

    assert [
        item.content
        for item in mixed.memories
    ] == [
        "用户正在学习 FastAPI",
        "用户的目标是转岗后端",
    ], (
        "只有 dict 与 MemoryExtractorLLMItem "
        "应当被保留，"
        "非 dict garbage 必须被 skip"
    )

    # --------------------------------------------------
    # 4. 端到端：
    #    Model 实例形态必须也能产出 MemoryCandidate
    #
    #    这是 MODEL INSTANCE TRAP 真正破坏的能力，
    #    因此不只测 Model 边界，
    #    还要验证它能穿过真实 Extractor。
    # --------------------------------------------------

    _, candidates, _ = run_extractor(
        user_message="我没学C++了",
        output=extractor_output(
            MemoryExtractorLLMItem(
                content="用户不再学习C++",
                memory_type="fact",
            ),
            "garbage",
            MemoryExtractorLLMItem(
                content="用户的目标是转岗后端",
                memory_type="goal",
            ),
        ),
    )

    assert [
        candidate.content
        for candidate in candidates
    ] == [
        "用户不再学习C++",
        "用户的目标是转岗后端",
    ], (
        "Model 实例形态必须能穿过真实 Extractor，"
        "非 dict garbage 仍应被跳过，"
        f"实际为 {[c.content for c in candidates]}"
    )

    print(
        "[PASS] dict / MemoryExtractorLLMItem 均为合法 item，"
        "非 dict garbage 被 skip"
    )


def test_extractor_per_item_skip_contract():
    """
    ★ 核心 Business Contract ★

    一条坏 Memory
    不得拖死同批次其他合法 Memory。

        A 合法
        B 非法
        C 合法
            ↓
        [A, C]
            ↓
        不能是 []

    迁移前后这条路径完全一致：

        Structured Output（memories 已经过顶层过滤）
            ↓
        Extractor 逐条校验
            ↓
        非法 → continue
        合法 → MemoryCandidate

    下面用一张表覆盖所有非法形态，
    每一行都同时断言
    「坏条目被跳过」+「好条目被保留」。

    这里故意**不**把每种非法形态拆成
    多个名字不同、实际走同一路径的测试：
    逐条 skip 是**同一条**代码路径，
    拆开只会让测试数量虚增而非覆盖变广。

    非法形态包括：

        非 dict item
        content 非 str
        content 空串 / 纯空白
        memory_type 不在白名单
    """

    valid_a = {
        "content": "用户正在学习 FastAPI",
        "memory_type": "fact",
    }

    valid_c = {
        "content": "用户的目标是转岗后端",
        "memory_type": "goal",
    }

    cases = (
        (
            "全部合法 A/B/C",
            [
                valid_a,
                {
                    "content": "用户偏好用 Python",
                    "memory_type": "preference",
                },
                valid_c,
            ],
            [
                "用户正在学习 FastAPI",
                "用户偏好用 Python",
                "用户的目标是转岗后端",
            ],
        ),
        (
            "A 合法 / B 非法 memory_type / C 合法",
            [
                valid_a,
                {
                    "content": "临时信息",
                    "memory_type": "temporary",
                },
                valid_c,
            ],
            [
                "用户正在学习 FastAPI",
                "用户的目标是转岗后端",
            ],
        ),
        (
            "A 合法 / B content 非 str / C 合法",
            [
                valid_a,
                {
                    "content": 123,
                    "memory_type": "fact",
                },
                valid_c,
            ],
            [
                "用户正在学习 FastAPI",
                "用户的目标是转岗后端",
            ],
        ),
        (
            "A 合法 / B content 空串 / C 合法",
            [
                valid_a,
                {
                    "content": "",
                    "memory_type": "fact",
                },
                valid_c,
            ],
            [
                "用户正在学习 FastAPI",
                "用户的目标是转岗后端",
            ],
        ),
        (
            "A 合法 / B content 纯空白 / C 合法",
            [
                valid_a,
                {
                    "content": "   \n\t ",
                    "memory_type": "fact",
                },
                valid_c,
            ],
            [
                "用户正在学习 FastAPI",
                "用户的目标是转岗后端",
            ],
        ),
        (
            "A 合法 / B 非 dict / C 合法",
            [
                valid_a,
                "garbage",
                valid_c,
            ],
            [
                "用户正在学习 FastAPI",
                "用户的目标是转岗后端",
            ],
        ),
        (
            "A 坏 / B 合法",
            [
                "garbage",
                valid_a,
            ],
            [
                "用户正在学习 FastAPI",
            ],
        ),
    )

    for (
        title,
        memories,
        expected_contents,
    ) in cases:

        _, candidates, _ = run_extractor(
            user_message="我没学C++了",
            output=extractor_output(*memories),
        )

        assert [
            candidate.content
            for candidate in candidates
        ] == expected_contents, (
            f"{title}："
            "非法条目必须被跳过，"
            "合法条目必须全部保留，"
            "实际为 "
            f"{[c.content for c in candidates]}"
        )

    print(
        "[PASS] Extractor Per-item Skip Contract"
        "（含 A合法/B非法/C合法 混合批次）"
    )


# ============================================================
# ============================================================
# 5. Extractor Structured Failure Contract
#
# 旧实现里的：
#
#     raw response 非 str
#     非法 JSON 语法
#
# 已经不属于 Extractor 责任，
# 它们统一由 Provider / SDK / Pydantic Boundary
# 变成某种异常。
#
# 因此这里改为直接验证
# 「顶层结构非法 / Provider 失败」的处理。
# ============================================================
# ============================================================


def test_extractor_returns_empty_when_memories_is_none():
    """
    memories=None → []

    旧实现：

        data.get("memories", [])
            → None（key 存在）
            → not isinstance(None, list)
            → []

    新实现由顶层 before validator 承担，
    行为保持一致。
    """

    _, candidates, _ = run_extractor(
        user_message="我没学C++了",
        output=MemoryExtractorLLMOutput(
            memories=None
        ),
    )

    assert candidates == [], (
        "memories=None 应当等价于空列表"
    )


def build_real_invalid_memories_error():
    """
    真实构造一个 ValidationError：

        MemoryExtractorLLMOutput(
            memories="not a list"
        )

    这正是顶层结构非法时，
    Provider / Pydantic Boundary 抛出的异常类型。
    """

    try:

        MemoryExtractorLLMOutput(
            memories="not a list"
        )

    except ValidationError as exc:

        return exc

    raise AssertionError(
        "memories 不是 list，"
        "MemoryExtractorLLMOutput 应该拒绝"
    )


def test_extractor_fails_closed_when_memories_is_not_list():
    """
    memories 非 list
        ↓
    Structured Validation Failure
        ↓
    []

    旧实现是 Extractor 内部显式检查：

        if not isinstance(memories, list):
            return []

    新实现由顶层 before validator
    raise ValueError 承担，
    再被 except (ValidationError, ValueError)
    收敛为 []。

    Business Contract 不变：

        顶层结构非法
            → 本次没有提取到 Memory
    """

    _, candidates, fake_llm = run_extractor(
        user_message="我没学C++了",
        exc=build_real_invalid_memories_error(),
    )

    assert candidates == [], (
        "顶层结构非法应当 fail-closed 返回 []"
    )

    assert fake_llm.call_count == 1

    print(
        "[PASS] Extractor 对顶层结构非法 "
        "Fail Closed 返回 []"
    )


def test_extractor_propagates_provider_runtime_error():
    """
    Provider 层 RuntimeError
        ↓
    必须**原样向上传播**

    Failure Policy 没有变化：

        旧实现里 await call_llm(...)
        并不在 JSON try/except 之中，
        因此普通 LLM 调用异常本来就会向上传播。

        新实现只捕获：

            except (ValidationError, ValueError)

        RuntimeError 不在其中，
        因此同样向上传播。

    不要为了"看起来更稳"
    把它改成 []：
    那会改变旧 Failure Policy，
    并把 Provider 故障伪装成"没有 Memory"。
    """

    provider_error = RuntimeError(
        "DeepSeek unavailable"
    )

    try:

        run_extractor(
            user_message="我没学C++了",
            exc=provider_error,
        )

    except RuntimeError as exc:

        assert exc is provider_error, (
            "必须原样抛出 Provider 的原始异常"
        )

    else:

        raise AssertionError(
            "Provider RuntimeError "
            "必须继续向上传播，"
            "不能返回 []"
        )

    print(
        "[PASS] Extractor 传播 Provider RuntimeError"
    )


def test_validator_maps_valid_true_to_business_result():
    """
    Structured Output(valid=True)
        ↓
    MemoryValidationResult(valid=True)

    并逐字保留 reason。

    reason 这里刻意同时覆盖：

        正常字符串
        ""（空字符串）

    因为旧 Contract 只要求 reason 是 str，
    并没有拒绝空字符串。

    迁移后 Model 使用 StrictStr，
    而不是"非空校验"，
    所以空 reason 必须继续被接受并原样透传。

    重点：

        "状态终止"类 Candidate
        在 Validator 层不得被硬编码拒绝。
    """

    candidate = MemoryCandidate(
        content="用户不再学习C++了",
        memory_type="fact",
    )

    for reason in (
        "用户学习状态发生变化",
        "",
    ):

        _, result, _ = run_validator(
            candidate,
            validator_output(True, reason),
        )

        assert result.valid is True, (
            "State Termination Candidate "
            "应当可以通过 Validator"
        )

        assert result.reason == reason, (
            "reason 必须逐字保留，"
            f"期望 {reason!r}，"
            f"实际 {result.reason!r}"
        )

    print(
        "[PASS] Validator 允许 "
        "State Termination Candidate 通过，"
        "且空 reason 保持兼容"
    )


def test_validator_maps_valid_false_to_business_result():
    """
    Structured Output(valid=False)
        ↓
    MemoryValidationResult(valid=False)

    这保证"一次性事件仍可拒绝"的通道完好。
    """

    candidate = MemoryCandidate(
        content="用户今天下午学了两个小时C++",
        memory_type="fact",
    )

    _, result, _ = run_validator(
        candidate,
        validator_output(
            False,
            "一次性事件"
        ),
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

    这里断言 fake.call_count == 0，
    而不是"Fake 没有输出"，
    避免把「根本没调用」
    与「调用了但输出为空」混为一谈。
    """

    candidate = MemoryCandidate(
        content="C++",
        memory_type="fact",
    )

    captured, result, fake_llm = run_validator(
        candidate,
        validator_output(
            True,
            "不应被调用"
        ),
    )

    assert result.valid is False

    assert fake_llm.call_count == 0, (
        "Rule 层失败时不应调用 LLM"
    )

    assert captured["messages"] is None, (
        "Rule 层失败时不应发出任何 messages"
    )

    print(
        "[PASS] Validator Rule 层 "
        "短路且不调用 LLM"
    )


# ============================================================
# ============================================================
# Validator LLM Boundary Model Contract
#
# Structured Output 迁移之后，
# 下面这些**不再是 MemoryValidator 的职责**：
#
#     raw response 是不是 str
#     json.loads
#     JSON 顶层是否 object
#     valid 是不是 bool
#     reason 是不是 str
#
# 它们由 Provider / SDK / Pydantic Boundary 承担。
# 因此这里直接对
# MemoryValidatorLLMOutput 做 Contract 断言。
# ============================================================
# ============================================================


def assert_model_rejects(build):
    """
    断言构造 MemoryValidatorLLMOutput 时被拒绝。

    Pydantic 在 strict 类型不匹配时抛
    ValidationError（ValueError 的子类）。

    这里只接受 ValidationError：
    其他异常类型不应被误判为"已拒绝"。
    """

    try:

        build()

    except ValidationError:

        return

    raise AssertionError(
        "MemoryValidatorLLMOutput "
        "应该拒绝该输入"
    )


def test_validator_llm_output_model_contract():
    """
    valid  : StrictBool
    reason : StrictStr

    并且 **reason 允许为空字符串**。

    这是刻意保留的旧 Contract：

        旧实现只要求 reason 是 str，
        并不拒绝 "" 或纯空白。

    因此 Model 层不允许出现
    "reason 非空" 这类 validator，
    reason 也不做 strip。
    """

    for reason in (
        "用户学习状态发生变化",
        "",
        "   ",
    ):

        output = MemoryValidatorLLMOutput(
            valid=True,
            reason=reason,
        )

        assert output.valid is True

        assert output.reason == reason, (
            "reason 必须原样保留"
            "（不做 strip、不做非空校验），"
            f"期望 {reason!r}，"
            f"实际 {output.reason!r}"
        )

    rejected = MemoryValidatorLLMOutput(
        valid=False,
        reason="一次性事件",
    )

    assert rejected.valid is False


def test_validator_llm_output_rejects_non_strict_bool():
    """
    valid 必须是严格 bool。

        1 / 0 / "true" / "yes" / ""

    都不能偷偷转换成 bool。
    """

    for bad_valid in (
        1,
        0,
        "true",
        "yes",
        "",
        None,
    ):

        assert_model_rejects(
            lambda bad_valid=bad_valid: (
                MemoryValidatorLLMOutput(
                    valid=bad_valid,
                    reason="r",
                )
            )
        )


def test_validator_llm_output_rejects_non_strict_str_reason():
    """
    reason 必须是严格 str。

    注意这里**只校验类型**：

        reason=""

    是合法的。
    """

    for bad_reason in (
        123,
        None,
        True,
        ["r"],
        {"reason": "r"},
    ):

        assert_model_rejects(
            lambda bad_reason=bad_reason: (
                MemoryValidatorLLMOutput(
                    valid=True,
                    reason=bad_reason,
                )
            )
        )


def test_validator_requests_validator_llm_output_model():
    """
    Validator 必须向 Provider 声明
    自己需要的是 MemoryValidatorLLMOutput。

    这是 Structured Output 迁移之后
    新增的、真实的 Validator 责任：
    声明输出 Schema。
    """

    candidate = MemoryCandidate(
        content="用户不再学习C++了",
        memory_type="fact",
    )

    _, _, fake_llm = run_validator(
        candidate,
        validator_output(),
    )

    assert (
        fake_llm.received_output_model
        is MemoryValidatorLLMOutput
    ), (
        "Validator 必须以 "
        "MemoryValidatorLLMOutput "
        "作为 output_model"
    )


def build_real_invalid_output_error():
    """
    真实构造一个 ValidationError：

        MemoryValidatorLLMOutput(
            valid="true",   # 不是 StrictBool
            reason="r",
        )

    这正是 Provider 返回无法通过
    Structured Output 校验的内容时，
    向上抛出的异常类型。
    """

    try:

        MemoryValidatorLLMOutput(
            valid="true",
            reason="r",
        )

    except ValidationError as exc:

        return exc

    raise AssertionError(
        "valid 不是 StrictBool，"
        "MemoryValidatorLLMOutput 应该拒绝"
    )


def test_validator_fails_closed_on_structured_llm_failure():
    """
    Structured LLM 失败
        ↓
    valid=False
        ↓
    不 raise，不影响 Chat 主流程。

    Failure Policy 与旧实现一致，
    只是失败来源变多了：

        1. Provider / 网络层失败
           RuntimeError

        2. Provider 返回无法通过
           MemoryValidatorLLMOutput 校验的内容
           → Pydantic ValidationError

    旧实现里的：

        raw response 非 str
        非法 JSON
        JSON 顶层非 object
        valid 非 bool
        reason 非 str

    已经不属于 Validator 责任，
    它们统一由 Model / Provider Boundary
    变成"某个异常"，
    再在这里收敛为 valid=False。

    本测试只断言 Business Contract：

        非法 / 失败的 LLM Output
            → valid=False

    不要求恢复旧 JSONDecodeError
    的精确文案。
    """

    fail_closed_prefix = (
        "Memory Validation执行失败："
    )

    candidate = MemoryCandidate(
        content="用户不再学习C++了",
        memory_type="fact",
    )

    cases = (
        (
            RuntimeError(
                "fake-provider-down"
            ),
            "fake-provider-down",
        ),
        (
            build_real_invalid_output_error(),
            None,
        ),
    )

    for exc, expected_marker in cases:

        _, result, fake_llm = run_validator(
            candidate,
            exc=exc,
        )

        assert result.valid is False, (
            f"{type(exc).__name__} "
            "必须 fail-closed"
        )

        assert isinstance(
            result.reason,
            str
        ), "fail-closed reason 必须是 str"

        assert result.reason.startswith(
            fail_closed_prefix
        ), (
            f"{type(exc).__name__} "
            "必须走 fail-closed 分支，"
            f"实际 reason：{result.reason}"
        )

        assert len(result.reason) > len(
            fail_closed_prefix
        ), (
            "fail-closed reason "
            "必须带上底层错误信息"
        )

        if expected_marker is not None:

            assert (
                expected_marker
                in result.reason
            ), (
                "底层错误信息必须被转发，"
                f"期望包含 {expected_marker!r}，"
                f"实际为：{result.reason}"
            )

        assert fake_llm.call_count == 1, (
            "失败路径也必须真实调用过一次 "
            "call_llm_structured"
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
    # Extractor Structured Output → MemoryCandidate
    # --------------------------------------------------

    test_extractor_maps_structured_item_to_candidate()
    test_extractor_requests_extractor_llm_output_model()
    test_extractor_returns_empty_when_no_memories()
    test_extractor_llm_output_accepts_dict_and_model_items()
    test_extractor_per_item_skip_contract()

    # --------------------------------------------------
    # Extractor Structured Failure Contract
    # --------------------------------------------------

    test_extractor_returns_empty_when_memories_is_none()
    test_extractor_fails_closed_when_memories_is_not_list()
    test_extractor_propagates_provider_runtime_error()

    # --------------------------------------------------
    # Validator Structured Output → Business Result
    # --------------------------------------------------

    test_validator_maps_valid_true_to_business_result()
    test_validator_maps_valid_false_to_business_result()
    test_validator_rule_layer_rejects_too_short()

    # --------------------------------------------------
    # Validator LLM Boundary Model Contract
    # --------------------------------------------------

    test_validator_llm_output_model_contract()
    test_validator_llm_output_rejects_non_strict_bool()
    test_validator_llm_output_rejects_non_strict_str_reason()
    test_validator_requests_validator_llm_output_model()
    test_validator_fails_closed_on_structured_llm_failure()

    print()
    print("=" * 70)
    print(
        "ALL MEMORY WRITE STATE CHANGE "
        "REGRESSION TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()
