"""
MemoryRelevanceJudge Status Contract
[KEEP] Deterministic Tests
===========================================================

本文件覆盖 MemoryRelevanceJudge Contract 从

    texts: list[str]

升级为

    candidates: list[MemoryRelevanceCandidate]

之后的状态契约。


MemoryRelevanceCandidate：

    content
    memory_status


覆盖内容：

    1. Historical Status Preservation
    2. Current Status Preservation
    3. Prompt Semantic Contract
    4. Input Contract
    5. LLM Boundary Model Contract
    6. Output / Coverage Contract
    7. Reader → Relevance Judge
    8. Architecture Boundary


Component Responsibility（Structured Output 迁移之后）：

    MemoryRelevanceJudge 的 llm_callable 已经是：

        await llm_callable(
            messages=...,
            output_model=MemoryRelevanceLLMOutput,
        )
        → MemoryRelevanceLLMOutput

    因此 Judge **不再负责**：

        response 是否 str
        空字符串 / 空白字符串
        Markdown code fence
        json.loads 语法解析
        JSON 顶层是否 object
        decisions 是否 list
        index 类型
        selected 类型
        reason 类型 / 非空

    上面这些由 Provider / SDK / Pydantic Boundary 承担，
    本文件用 "LLM Boundary Model Contract" 分节直接测
    MemoryRelevanceLLMDecision / MemoryRelevanceLLMOutput。

    Judge 继续负责**运行时**才能确定的业务关系：

        decision 数量 == candidate 数量
        index 在本次 candidate 范围内
        index 不重复
        完整覆盖所有 candidate
        Strict Model → JudgeDecision 转换
        candidates=[] → []，且不调用 LLM
        Structured LLM / Pydantic 异常原样向上抛
        （Failure Policy 没有 fallback）
        最终结果按 index 排序


设计约束：

    deterministic
        不联网
        不调用真实 DeepSeek
        不加载真实 Embedding
        不加载真实 Reranker
        不写真实 test.db / backend.db

    LLM 通过 SyncRelevanceJudge(llm_callable=...)
    注入 Fake（Sync 只是把 coroutine 驱动到底）。


Async Contract 注意事项：

    MemoryRelevanceJudge.judge
    MemoryReader.read

    已经是 async Contract。

    本模块的 test_ 函数全部保持同步
    def test_xxx()，由 run() 把 coroutine 驱动到底：

        1. 直接 python 运行时会真的执行
        2. pytest 会原生收集执行，
           不会被当成 async test 静默跳过
        3. coroutine 一定被 await
        4. 不会产生假 PASS

    本模块不依赖 pytest-asyncio。


运行：

    .venv/Scripts/python.exe tests/memory/test_memory_relevance_judge_status_contract.py
"""


import asyncio
import sys
import json
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from pydantic import ValidationError  # noqa: E402

from backend.memory.memory_read.memory_relevance_judge import (  # noqa: E402
    MemoryRelevanceCandidate,
    MemoryRelevanceJudge,
    MemoryRelevanceLLMDecision,
    MemoryRelevanceLLMOutput,
    JudgeDecision,
)

from backend.memory.memory_read.memory_reader import (  # noqa: E402
    MemoryReader,
)

from backend.memory.memory_read.memory_query_scope import (  # noqa: E402
    MemoryQueryScopeDecision,
)

from backend.memory.memory_read.dense_retriever import (  # noqa: E402
    DenseRetrievedMemory,
)

from backend.retrieval.bm25 import (  # noqa: E402
    BM25Result,
)

from backend.retrieval.rrf import (  # noqa: E402
    RRFResult,
)

from backend.reranking.reranker import (  # noqa: E402
    RerankResult,
)


# ============================================================
# Source Paths
# ============================================================

READER_SOURCE_PATH = (
    PROJECT_ROOT
    / "backend"
    / "memory"
    / "memory_read"
    / "memory_reader.py"
)

JUDGE_SOURCE_PATH = (
    PROJECT_ROOT
    / "backend"
    / "memory"
    / "memory_read"
    / "memory_relevance_judge.py"
)

CHAT_SERVICE_SOURCE_PATH = (
    PROJECT_ROOT
    / "backend"
    / "services"
    / "chat_service.py"
)

RETRIEVAL_SOURCE_PATHS = (
    PROJECT_ROOT
    / "backend"
    / "memory"
    / "memory_read"
    / "dense_retriever.py",
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "bm25.py",
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "rrf.py",
    PROJECT_ROOT
    / "backend"
    / "reranking"
    / "reranker.py",
)


# ============================================================
# Helpers
# ============================================================


def run(coro):
    """
    在同步测试中真实执行并等待
    async production contract。

    背景：

    MemoryRelevanceJudge.judge / MemoryReader.read
    已经是 async Contract。

    但本项目当前没有 pytest-asyncio，
    因此本模块的测试函数保持同步 def test_xxx()：

        1. 直接 python 运行时会真的执行测试
        2. pytest 会原生收集执行，
           不会被当成 async test 静默跳过
        3. coroutine 一定被 await，
           不会出现 coroutine was never awaited
        4. 不会产生假 PASS

    这里只负责把 coroutine 驱动到底，
    不参与任何业务断言。
    """

    return asyncio.run(coro)


class SyncRelevanceJudge(MemoryRelevanceJudge):
    """
    真实 MemoryRelevanceJudge
    +
    同步调用适配。

    只把 async judge() 的 coroutine
    驱动到底，不复制、不绕过生产逻辑。
    """

    def judge(
        self,
        *args,
        **kwargs
    ):
        return run(
            super().judge(
                *args,
                **kwargs
            )
        )


class SyncMemoryReader(MemoryReader):
    """
    真实 MemoryReader
    +
    同步调用适配。

    只把 async read() 的 coroutine
    驱动到底，不复制、不绕过生产逻辑。
    """

    def read(
        self,
        *args,
        **kwargs
    ):
        return run(
            super().read(
                *args,
                **kwargs
            )
        )


class FakeStructuredRelevanceLLM:
    """
    Fake call_llm_structured。

    捕获 Judge 实际发出的 messages 与
    output_model，返回预设 Structured Output。

    生产 Contract：

        await self._llm_callable(
            messages=messages,
            output_model=MemoryRelevanceLLMOutput,
        )
        → MemoryRelevanceLLMOutput

    因此本 Fake 必须保持同样的
    keyword-only Calling Contract，
    并且返回 Structured Output Model，
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
            is MemoryRelevanceLLMOutput
        ), (
            "Relevance Judge 必须以 "
            "MemoryRelevanceLLMOutput "
            "作为 output_model，"
            f"实际为：{output_model}"
        )

        if self._exc is not None:

            raise self._exc

        if self._output is None:

            raise AssertionError(
                "FakeStructuredRelevanceLLM "
                "未配置输出"
            )

        return self._output


def llm_decision(
    index=0,
    selected=True,
    reason="fake reason"
):
    """
    构造单条 LLM Boundary Decision。

    这里**不做任何类型预处理**：
    非法类型应当由 Pydantic 拒绝，
    而不是被测试 Helper 提前吞掉。
    """

    return MemoryRelevanceLLMDecision(
        index=index,
        selected=selected,
        reason=reason,
    )


def llm_output(
    *decisions
):
    """
    构造顶层 Structured Output。
    """

    return MemoryRelevanceLLMOutput(
        decisions=list(decisions)
    )


def single_decision(
    index=0,
    selected=True,
    reason="fake reason"
):
    return llm_output(
        llm_decision(
            index=index,
            selected=selected,
            reason=reason,
        )
    )


def normalize(
    text
):
    """
    折叠所有空白，
    避免 Prompt 换行导致断言脆弱。
    """

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def system_prompt_of(
    messages
):
    return messages[0][
        "content"
    ]


def user_payload_of(
    messages
):
    """
    User Prompt 是一个 JSON String。
    """

    return json.loads(
        messages[1][
            "content"
        ]
    )


def build_judge(
    output
):
    """
    构造一个使用 Fake Structured LLM 的
    真实 MemoryRelevanceJudge。
    """

    fake_llm = FakeStructuredRelevanceLLM(
        output=output
    )

    judge = SyncRelevanceJudge(
        llm_callable=fake_llm
    )

    return judge, fake_llm


# ============================================================
# ============================================================
# Section 1
# Historical / Current Status Preservation
#
# 发送给 LLM 的 JSON candidate
# 必须完整保留 memory_status。
# ============================================================
# ============================================================


def test_judge_sends_historical_status_to_llm():
    """
    Candidate：

        MemoryRelevanceCandidate(
            content="用户当前正在学习AI Agent",
            memory_status="historical"
        )

    Query：

        我以前主要学什么？

    必须验证发送给 LLM 的 JSON candidate：

        {
            "content": "用户当前正在学习AI Agent",
            "memory_status": "historical"
        }

    不能丢 memory_status。
    """

    judge, fake_llm = build_judge(
        output=single_decision()
    )

    judge.judge(
        query="我以前主要学什么？",
        candidates=[
            MemoryRelevanceCandidate(
                content=(
                    "用户当前正在学习AI Agent"
                ),
                memory_status="historical"
            )
        ]
    )

    payload = user_payload_of(
        fake_llm.received_messages
    )

    assert payload["query"] == (
        "我以前主要学什么？"
    )

    assert payload[
        "candidates"
    ] == [
        {
            "index": 0,
            "content": (
                "用户当前正在学习AI Agent"
            ),
            "memory_status": (
                "historical"
            ),
        }
    ], (
        "发送给 LLM 的 candidate "
        "必须完整保留 memory_status"
    )


def test_judge_sends_current_status_to_llm():
    """
    Candidate：

        MemoryRelevanceCandidate(
            content="用户正在学习安卓应用开发",
            memory_status="current"
        )

    确认 LLM input 中：

        memory_status=current
    """

    judge, fake_llm = build_judge(
        output=single_decision()
    )

    judge.judge(
        query="我现在主要学什么？",
        candidates=[
            MemoryRelevanceCandidate(
                content=(
                    "用户正在学习安卓应用开发"
                ),
                memory_status="current"
            )
        ]
    )

    payload = user_payload_of(
        fake_llm.received_messages
    )

    assert payload[
        "candidates"
    ][0][
        "memory_status"
    ] == "current", (
        "发送给 LLM 的 candidate "
        "必须保留 memory_status=current"
    )

    assert payload[
        "candidates"
    ][0][
        "content"
    ] == "用户正在学习安卓应用开发"


def test_judge_sends_mixed_statuses_to_llm():
    """
    多个 candidate 时，
    每种状态都不能丢失，
    index 必须按顺序对应。
    """

    judge, fake_llm = build_judge(
        output=llm_output(
            llm_decision(0, True, "r0"),
            llm_decision(1, False, "r1"),
        )
    )

    judge.judge(
        query="我从以前到现在有什么变化？",
        candidates=[
            MemoryRelevanceCandidate(
                content=(
                    "用户正在学习安卓应用开发"
                ),
                memory_status="current"
            ),
            MemoryRelevanceCandidate(
                content=(
                    "用户当前正在学习AI Agent"
                ),
                memory_status="historical"
            ),
        ]
    )

    payload = user_payload_of(
        fake_llm.received_messages
    )

    assert payload[
        "candidates"
    ] == [
        {
            "index": 0,
            "content": (
                "用户正在学习安卓应用开发"
            ),
            "memory_status": "current",
        },
        {
            "index": 1,
            "content": (
                "用户当前正在学习AI Agent"
            ),
            "memory_status": "historical",
        },
    ]


# ============================================================
# ============================================================
# Section 2
# Prompt Semantic Contract
# ============================================================
# ============================================================


def build_prompt():
    """
    取得真实 system prompt（归一化后）。
    """

    judge, fake_llm = build_judge(
        output=single_decision()
    )

    judge.judge(
        query="我以前主要学什么？",
        candidates=[
            MemoryRelevanceCandidate(
                content="用户当前正在学习AI Agent",
                memory_status="historical"
            )
        ]
    )

    return normalize(
        system_prompt_of(
            fake_llm.received_messages
        )
    )


def test_prompt_defines_current_status():
    """
    Prompt 必须明确表达：

        current 表示仍代表当前状态。
    """

    prompt = build_prompt()

    assert (
        "memory_status=current"
        in prompt
    )

    assert (
        "仍然代表用户当前状态"
        in prompt
    ), (
        "Prompt 必须说明 current "
        "仍代表用户当前状态"
    )


def test_prompt_defines_historical_status():
    """
    Prompt 必须明确表达：

        historical 表示过去成立，
        但现在不再代表当前状态。
    """

    prompt = build_prompt()

    assert (
        "memory_status=historical"
        in prompt
    )

    assert (
        "曾经成立" in prompt
    ), (
        "Prompt 必须说明 historical "
        "曾经成立"
    )

    assert (
        "不再代表" in prompt
    ), (
        "Prompt 必须说明 historical "
        "不再代表用户当前状态"
    )


def test_prompt_declares_status_more_authoritative():
    """
    Prompt 必须明确表达：

        memory_status 比 content 中遗留的
        当前 / 正在 / 目前 / 现在
        更权威。
    """

    prompt = build_prompt()

    assert (
        "更加权威" in prompt
    ), (
        "Prompt 必须声明 memory_status "
        "比 content 遗留措辞更权威"
    )

    assert (
        "content" in prompt
    )


def test_prompt_declares_historical_not_auto_selected():
    """
    Prompt 必须明确表达：

        historical 不代表自动 selected=true。
    """

    prompt = build_prompt()

    assert (
        "并不意味着" in prompt
    ), (
        "Prompt 必须说明 historical "
        "不意味着自动 selected=true"
    )

    assert (
        "selected 必须为 true"
        in prompt
    )


def test_prompt_requires_query_based_relevance():
    """
    Prompt 必须要求：

        仍需要结合 Query 判断 relevance。
    """

    prompt = build_prompt()

    assert (
        "仍然需要根据当前 Query"
        in prompt
    ), (
        "Prompt 必须要求结合 Query "
        "判断 historical Memory 的实际相关性"
    )


def test_prompt_declares_no_query_scope_duty():
    """
    Relevance Judge 不负责 Query Scope。

    Prompt 必须明确禁止它判断 Query Scope。
    """

    prompt = build_prompt()

    assert (
        "判断 Query Scope" in prompt
    ), (
        "Prompt 的职责边界中必须明确"
        "禁止判断 Query Scope"
    )


def test_prompt_declares_coverage_contract():
    """
    Prompt 必须声明 Coverage Contract：

        每个 candidate 恰好一次判断，
        不遗漏，不重复。
    """

    prompt = build_prompt()

    assert (
        "Coverage Contract"
        in prompt
    )

    assert (
        "恰好返回一次判断"
        in prompt
    )


# ============================================================
# ============================================================
# Section 3
# Input Contract
# ============================================================
# ============================================================


def assert_raises(
    exception_type,
    func
):
    try:

        func()

    except exception_type:

        return

    raise AssertionError(
        f"应该抛出 {exception_type.__name__}"
    )


def test_rejects_non_list_candidates():
    """
    candidates 非 list
        ↓
    TypeError
    """

    judge = SyncRelevanceJudge(
        llm_callable=FakeStructuredRelevanceLLM()
    )

    for bad in (
        "用户正在学习安卓",
        (
            MemoryRelevanceCandidate(
                content="x",
                memory_status="current"
            ),
        ),
        None,
    ):

        assert_raises(
            TypeError,
            lambda bad=bad: judge.judge(
                query="我以前主要学什么？",
                candidates=bad
            )
        )


def test_rejects_non_candidate_element():
    """
    candidate 非 MemoryRelevanceCandidate
        ↓
    TypeError
    """

    judge = SyncRelevanceJudge(
        llm_callable=FakeStructuredRelevanceLLM()
    )

    assert_raises(
        TypeError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                "用户正在学习安卓应用开发"
            ]
        )
    )


def test_rejects_non_str_content():
    """
    content 非 str
        ↓
    TypeError
    """

    judge = SyncRelevanceJudge(
        llm_callable=FakeStructuredRelevanceLLM()
    )

    assert_raises(
        TypeError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                MemoryRelevanceCandidate(
                    content=123,
                    memory_status="current"
                )
            ]
        )
    )


def test_rejects_empty_content():
    """
    content 空
        ↓
    ValueError
    """

    judge = SyncRelevanceJudge(
        llm_callable=FakeStructuredRelevanceLLM()
    )

    assert_raises(
        ValueError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                MemoryRelevanceCandidate(
                    content="",
                    memory_status="current"
                )
            ]
        )
    )


def test_rejects_blank_content():
    """
    content 全空白
        ↓
    ValueError
    """

    judge = SyncRelevanceJudge(
        llm_callable=FakeStructuredRelevanceLLM()
    )

    assert_raises(
        ValueError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                MemoryRelevanceCandidate(
                    content="   \n\t ",
                    memory_status="current"
                )
            ]
        )
    )


def test_rejects_non_str_memory_status():
    """
    memory_status 非 str
        ↓
    TypeError
    """

    judge = SyncRelevanceJudge(
        llm_callable=FakeStructuredRelevanceLLM()
    )

    assert_raises(
        TypeError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                MemoryRelevanceCandidate(
                    content=(
                        "用户正在学习安卓应用开发"
                    ),
                    memory_status=None
                )
            ]
        )
    )


def test_rejects_illegal_memory_status():
    """
    memory_status = past
        ↓
    ValueError
    """

    judge = SyncRelevanceJudge(
        llm_callable=FakeStructuredRelevanceLLM()
    )

    assert_raises(
        ValueError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                MemoryRelevanceCandidate(
                    content=(
                        "用户正在学习安卓应用开发"
                    ),
                    memory_status="past"
                )
            ]
        )
    )


def test_empty_candidates_returns_empty_list():
    """
    candidates=[]
        ↓
    []

    且不调用 LLM。
    """

    fake_llm = FakeStructuredRelevanceLLM()

    judge = SyncRelevanceJudge(
        llm_callable=fake_llm
    )

    decisions = judge.judge(
        query="我以前主要学什么？",
        candidates=[]
    )

    assert decisions == []

    assert fake_llm.call_count == 0, (
        "空 candidates 不应该调用 LLM"
    )


def test_rejects_invalid_query():
    """
    query 非 str / 空
    """

    judge = SyncRelevanceJudge(
        llm_callable=FakeStructuredRelevanceLLM()
    )

    assert_raises(
        TypeError,
        lambda: judge.judge(
            query=123,
            candidates=[]
        )
    )

    assert_raises(
        ValueError,
        lambda: judge.judge(
            query="   ",
            candidates=[]
        )
    )


# ============================================================
# ============================================================
# Section 4
# LLM Boundary Model Contract
#
# Structured Output 迁移之后，
# 下面这些**不再是 Judge 的职责**：
#
#     response 是否 str
#     空字符串 / 空白字符串
#     Markdown code fence
#     json.loads 语法解析
#     JSON 顶层是否 object
#     decisions 是否 list
#     index / selected / reason 类型
#
# 它们由 Provider / SDK / Pydantic Boundary 承担。
# 因此这里直接对
#
#     MemoryRelevanceLLMDecision
#     MemoryRelevanceLLMOutput
#
# 做 Contract 断言，
# 不再经由 Judge 的旧 _parse_response 路径。
# ============================================================
# ============================================================


def assert_model_rejects(
    build
):
    """
    断言构造 LLM Boundary Model 时被拒绝。

    Pydantic 在 strict 类型不匹配 /
    必填字段缺失时抛 ValidationError
    （它是 ValueError 的子类）。

    这里只接受 ValidationError：
    其他异常类型不应被误判为"已拒绝"。
    """

    try:

        build()

    except ValidationError:

        return

    raise AssertionError(
        "Memory Relevance LLM Boundary Model "
        "应该拒绝该输入"
    )


def test_llm_decision_accepts_valid_payload():
    """
    合法 payload：

        index    StrictInt
        selected StrictBool
        reason   StrictStr 非空

    并且 reason 在写入业务对象前
    去除首尾空白。
    """

    decision = MemoryRelevanceLLMDecision(
        index=0,
        selected=True,
        reason="  这段 Memory 与当前问题相关  ",
    )

    assert decision.index == 0

    assert decision.selected is True

    assert decision.reason == (
        "这段 Memory 与当前问题相关"
    )


def test_llm_decision_rejects_non_strict_int_index():
    """
    index 必须是严格 int。

    bool 虽然是 int 的子类，
    StrictInt 不允许它混入；
    字符串 / 浮点 / None 同样拒绝。
    """

    for bad_index in (
        True,
        False,
        "0",
        1.0,
        None,
        [0],
    ):

        assert_model_rejects(
            lambda bad_index=bad_index: (
                MemoryRelevanceLLMDecision(
                    index=bad_index,
                    selected=True,
                    reason="r",
                )
            )
        )


def test_llm_decision_rejects_non_strict_bool_selected():
    """
    selected 必须是严格 bool。

        1 / 0 / "true" / "yes"

    都不能偷偷转换成 bool。
    """

    for bad_selected in (
        1,
        0,
        "true",
        "yes",
        "",
        None,
    ):

        assert_model_rejects(
            lambda bad_selected=bad_selected: (
                MemoryRelevanceLLMDecision(
                    index=0,
                    selected=bad_selected,
                    reason="r",
                )
            )
        )


def test_llm_decision_rejects_non_strict_str_reason():
    """
    reason 必须是严格 str。
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
                MemoryRelevanceLLMDecision(
                    index=0,
                    selected=True,
                    reason=bad_reason,
                )
            )
        )


def test_llm_decision_rejects_blank_reason():
    """
    reason 必须非空。

    空白字符串 strip 之后为空，
    因此同样被拒绝：
    合法 decision 不允许没有说明。
    """

    for bad_reason in (
        "",
        "   ",
        "\n\t  ",
    ):

        assert_model_rejects(
            lambda bad_reason=bad_reason: (
                MemoryRelevanceLLMDecision(
                    index=0,
                    selected=True,
                    reason=bad_reason,
                )
            )
        )


def test_llm_decision_requires_all_fields():
    """
    index / selected / reason
    都是必填字段。
    """

    payloads = (
        {
            "selected": True,
            "reason": "r",
        },
        {
            "index": 0,
            "reason": "r",
        },
        {
            "index": 0,
            "selected": True,
        },
    )

    for payload in payloads:

        assert_model_rejects(
            lambda payload=payload: (
                MemoryRelevanceLLMDecision(
                    **payload
                )
            )
        )


def test_llm_output_requires_decisions_list():
    """
    decisions 是必填 list。

    None / dict / str 都不是 list；
    元素本身非法时由
    MemoryRelevanceLLMDecision 拒绝。
    """

    assert_model_rejects(
        lambda: MemoryRelevanceLLMOutput()
    )

    for bad_decisions in (
        None,
        {},
        "decisions",
        123,
    ):

        assert_model_rejects(
            lambda bad_decisions=bad_decisions: (
                MemoryRelevanceLLMOutput(
                    decisions=bad_decisions
                )
            )
        )


# ============================================================
# ============================================================
# Section 5
# Output / Coverage Contract
#
# JudgeDecision:
#
#     index
#     selected
#     reason
#
# Judge 只负责运行时才能确定的
# 业务关系 + Strict Model → JudgeDecision 转换。
# ============================================================
# ============================================================


def build_real_invalid_output_error():
    """
    真实构造一个 ValidationError：

        MemoryRelevanceLLMOutput(
            decisions=[
                MemoryRelevanceLLMDecision(
                    index="0",   # 不是 StrictInt
                    ...
                )
            ]
        )

    这正是 Provider 返回无法通过
    Structured Output 校验的内容时，
    向上抛出的异常类型。
    """

    try:

        MemoryRelevanceLLMOutput(
            decisions=[
                MemoryRelevanceLLMDecision(
                    index="0",
                    selected=True,
                    reason="r",
                )
            ]
        )

    except ValidationError as exc:

        return exc

    raise AssertionError(
        "index 不是 StrictInt，"
        "MemoryRelevanceLLMDecision 应该拒绝"
    )


def test_judge_requests_relevance_llm_output_model():
    """
    Judge 必须向 Provider 声明
    自己需要的是 MemoryRelevanceLLMOutput。

    这是 Structured Output 迁移之后
    新增的、真实的 Judge 责任：
    声明输出 Schema。
    """

    judge, fake_llm = build_judge(
        output=single_decision()
    )

    judge.judge(
        query="我以前主要学什么？",
        candidates=[
            MemoryRelevanceCandidate(
                content="A",
                memory_status="current"
            )
        ]
    )

    assert (
        fake_llm.received_output_model
        is MemoryRelevanceLLMOutput
    ), (
        "Judge 必须以 MemoryRelevanceLLMOutput "
        "作为 output_model"
    )


def test_decision_success_contract():
    """
    Structured Output
        ↓
    JudgeDecision[]

    index / selected / reason
    必须完整保留。
    """

    judge, _ = build_judge(
        output=llm_output(
            llm_decision(0, True, "r0"),
            llm_decision(1, False, "r1"),
        )
    )

    decisions = judge.judge(
        query="我从以前到现在有什么变化？",
        candidates=[
            MemoryRelevanceCandidate(
                content="A",
                memory_status="current"
            ),
            MemoryRelevanceCandidate(
                content="B",
                memory_status="historical"
            ),
        ]
    )

    assert len(decisions) == 2

    for decision in decisions:

        assert isinstance(
            decision,
            JudgeDecision
        )

    assert decisions[0].index == 0
    assert decisions[0].selected is True
    assert decisions[0].reason == "r0"

    assert decisions[1].index == 1
    assert decisions[1].selected is False
    assert decisions[1].reason == "r1"


def test_decision_order_is_stable():
    """
    LLM 乱序返回时，
    最终仍整理为 0,1,2。
    """

    judge, _ = build_judge(
        output=llm_output(
            llm_decision(2, True, "r2"),
            llm_decision(0, True, "r0"),
            llm_decision(1, False, "r1"),
        )
    )

    decisions = judge.judge(
        query="我以前主要学什么？",
        candidates=[
            MemoryRelevanceCandidate(
                content="A",
                memory_status="current"
            ),
            MemoryRelevanceCandidate(
                content="B",
                memory_status="current"
            ),
            MemoryRelevanceCandidate(
                content="C",
                memory_status="historical"
            ),
        ]
    )

    assert [
        decision.index
        for decision in decisions
    ] == [0, 1, 2]


def test_decision_count_mismatch():
    """
    decision 数量 != candidate 数量
        ↓
    ValueError

    这是运行时 Coverage Contract，
    Pydantic 静态模型无法校验。
    """

    judge, _ = build_judge(
        output=single_decision()
    )

    assert_raises(
        ValueError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                MemoryRelevanceCandidate(
                    content="A",
                    memory_status="current"
                ),
                MemoryRelevanceCandidate(
                    content="B",
                    memory_status="current"
                ),
            ]
        )
    )


def test_decision_index_out_of_range():
    """
    index 超出本次 candidate 范围
        ↓
    ValueError
    """

    judge, _ = build_judge(
        output=llm_output(
            llm_decision(5, True, "r")
        )
    )

    assert_raises(
        ValueError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                MemoryRelevanceCandidate(
                    content="A",
                    memory_status="current"
                )
            ]
        )
    )


def test_decision_index_duplicate():
    """
    index 重复
        ↓
    ValueError
    """

    judge, _ = build_judge(
        output=llm_output(
            llm_decision(0, True, "r0"),
            llm_decision(0, True, "r1"),
        )
    )

    assert_raises(
        ValueError,
        lambda: judge.judge(
            query="我以前主要学什么？",
            candidates=[
                MemoryRelevanceCandidate(
                    content="A",
                    memory_status="current"
                ),
                MemoryRelevanceCandidate(
                    content="B",
                    memory_status="current"
                ),
            ]
        )
    )


def test_structured_llm_failure_propagates_without_fallback():
    """
    Structured LLM 自身抛异常
        ↓
    原异常继续向上抛。

    Failure Policy 没有 fallback：

    Judge 不允许把 LLM 失败静默吞掉，
    也不允许返回空列表或默认 decision，
    否则 Reader 会拿到"全部未选中"
    这种伪装成正常结果的失败。

    这里用两类真实异常来源：

        1. Provider / 网络层失败
           RuntimeError

        2. Provider 返回无法通过
           MemoryRelevanceLLMOutput 校验的内容
           → Pydantic ValidationError
    """

    exceptions = (
        RuntimeError(
            "DeepSeek unavailable"
        ),
        build_real_invalid_output_error(),
    )

    for exc in exceptions:

        fake_llm = FakeStructuredRelevanceLLM(
            exc=exc
        )

        judge = SyncRelevanceJudge(
            llm_callable=fake_llm
        )

        assert_raises(
            type(exc),
            lambda judge=judge: judge.judge(
                query="我以前主要学什么？",
                candidates=[
                    MemoryRelevanceCandidate(
                        content="A",
                        memory_status="current"
                    )
                ]
            )
        )

        assert fake_llm.call_count == 1, (
            "失败路径也必须真实调用过一次 "
            "LLM Boundary"
        )


# ============================================================
# ============================================================
# Section 6
# Reader → Relevance Judge
# ============================================================
# ============================================================


@dataclass
class FakeMemory:

    content: str

    memory_status: str = (
        "current"
    )

    id: int = 0

    memory_type: str = "fact"


class CapturingJudge:
    """
    Fake Relevance Judge。

    捕获 Reader 实际传入的 candidates。

    注意：

    MemoryReader 现在 await self._judge.judge(...)，
    因此本 Fake 必须保持 async Calling Contract。
    """

    def __init__(self):
        self.call_count = 0
        self.received_candidates = None
        self.received_kwargs = None

    async def judge(
        self,
        query=None,
        candidates=None,
        **kwargs
    ):
        self.call_count += 1

        self.received_candidates = (
            list(candidates)
            if candidates is not None
            else None
        )

        self.received_kwargs = kwargs

        return [
            JudgeDecision(
                index=index,
                selected=True,
                reason="fake judge"
            )
            for index in range(
                len(candidates or [])
            )
        ]


class FakeScopeJudge:
    """
    Fake Scope Judge。

    注意：

    MemoryReader 现在 await self._scope_judge.judge(...)，
    因此本 Fake 必须保持 async Calling Contract。
    """

    def __init__(self):
        self._decision = (
            MemoryQueryScopeDecision(
                scope="both",
                reason="fake scope reason",
                source="rule"
            )
        )

    async def judge(
        self,
        query
    ):
        return self._decision


class FakeRepository:

    def __init__(
        self,
        memories
    ):
        self._memories = list(
            memories
        )

    def __call__(
        self,
        db,
        user_id,
        scope=None
    ):
        return list(
            self._memories
        )


class FakeDenseRetriever:

    def search(
        self,
        query,
        memories,
        top_n
    ):
        return [
            DenseRetrievedMemory(
                index=index,
                memory=memory,
                similarity=1.0
            )
            for index, memory in enumerate(
                memories
            )
        ]


class FakeBM25Retriever:

    def index(
        self,
        texts
    ):
        return None

    def search(
        self,
        query,
        top_n
    ):
        return []


class FakeRRF:

    def __call__(
        self,
        rankings,
        top_n=None
    ):
        indexes = []

        for ranking in rankings:

            for index in ranking:

                if index not in indexes:

                    indexes.append(
                        index
                    )

        return [
            RRFResult(
                index=index,
                score=1.0
            )
            for index in indexes
        ]


class FakeReranker:

    def rerank(
        self,
        query,
        texts
    ):
        return [
            RerankResult(
                index=index,
                text=text,
                score=1.0
            )
            for index, text in enumerate(
                texts
            )
        ]


class FakeInjector:

    def build_context(
        self,
        items
    ):
        return (
            "<memory_context>fake</memory_context>"
        )


def build_reader_with_capturing_judge(
    memories
):
    capturing_judge = (
        CapturingJudge()
    )

    reader = SyncMemoryReader(
        repository_callable=(
            FakeRepository(
                memories
            )
        ),
        scope_judge=FakeScopeJudge(),
        dense_retriever=(
            FakeDenseRetriever()
        ),
        bm25_retriever=(
            FakeBM25Retriever()
        ),
        rrf_callable=FakeRRF(),
        reranker=FakeReranker(),
        judge=capturing_judge,
        injector=FakeInjector(),
    )

    return reader, capturing_judge


def test_reader_passes_relevance_candidates():
    """
    MemoryReader 必须把：

        MemoryRelevanceCandidate

    传给 Relevance Judge，

    并且：

        content 完整保留
        memory_status 完整保留
    """

    reader, capturing_judge = (
        build_reader_with_capturing_judge(
            [
                FakeMemory(
                    content=(
                        "用户正在学习安卓应用开发"
                    ),
                    memory_status="current"
                ),
                FakeMemory(
                    content=(
                        "用户当前正在学习AI Agent"
                    ),
                    memory_status=(
                        "historical"
                    )
                ),
            ]
        )
    )

    reader.read(
        db=None,
        user_id=1,
        query="我从以前到现在有什么变化？",
        top_n=5,
        top_k=3
    )

    assert (
        capturing_judge.call_count == 1
    )

    candidates = (
        capturing_judge
        .received_candidates
    )

    assert candidates is not None, (
        "MemoryReader 必须向 Judge 传 candidates"
    )

    assert len(candidates) == 2

    for candidate in candidates:

        assert isinstance(
            candidate,
            MemoryRelevanceCandidate
        ), (
            "Judge 收到的必须是 "
            "MemoryRelevanceCandidate，"
            f"实际为 {type(candidate)}"
        )

    assert candidates[0].content == (
        "用户正在学习安卓应用开发"
    )

    assert (
        candidates[0].memory_status
        == "current"
    ), (
        "current Memory 的 "
        "memory_status 必须保留"
    )

    assert candidates[1].content == (
        "用户当前正在学习AI Agent"
    )

    assert (
        candidates[1].memory_status
        == "historical"
    ), (
        "historical Memory 的 "
        "memory_status 必须保留"
    )


def test_reader_does_not_receive_texts_contract():
    """
    Reader 不应该再使用旧的 texts= Contract。
    """

    reader, capturing_judge = (
        build_reader_with_capturing_judge(
            [
                FakeMemory(
                    content="A",
                    memory_status="current"
                )
            ]
        )
    )

    reader.read(
        db=None,
        user_id=1,
        query="我现在在学什么？",
        top_n=5,
        top_k=3
    )

    assert (
        "texts"
        not in capturing_judge.received_kwargs
    ), (
        "Relevance Judge 不应再收到 texts="
    )


def test_reader_does_not_interpret_memory_status():
    """
    MemoryReader 允许：

        读取 memory.memory_status

    但不允许：

        if memory_status == ...

    或其他等价业务解释。
    """

    source = (
        READER_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forbidden = (
        "memory_status ==",
        "memory_status !=",
        "if memory_status",
        "MEMORY_STATUS_CURRENT",
        "MEMORY_STATUS_HISTORICAL",
    )

    for pattern in forbidden:

        assert pattern not in source, (
            "MemoryReader 不应解释 "
            f"memory_status：{pattern}"
        )

    # --------------------------------------------------------
    # 正向确认：
    #
    # MemoryReader 确实把 memory_status
    # 传递给 Judge / Injector
    # --------------------------------------------------------

    assert (
        "MemoryRelevanceCandidate("
        in source
    )

    assert (
        "MemoryInjectionItem("
        in source
    )


def test_reader_does_not_interpret_scope():
    """
    MemoryReader 仍然不应解释 Query Scope。
    """

    source = (
        READER_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forbidden = (
        '== "current"',
        '== "historical"',
        '== "both"',
        "MEMORY_QUERY_SCOPE_CURRENT",
        "MEMORY_QUERY_SCOPE_HISTORICAL",
        "MEMORY_QUERY_SCOPE_BOTH",
    )

    for pattern in forbidden:

        assert pattern not in source, (
            "MemoryReader 不应解释 "
            f"Query Scope：{pattern}"
        )


# ============================================================
# ============================================================
# Section 7
# Architecture Boundary
# ============================================================
# ============================================================


def test_chat_service_knows_nothing_about_scope_or_status():
    """
    ChatService 不应该知道：

        MemoryQueryScopeDecision
        MEMORY_QUERY_SCOPE_*
        memory_status
    """

    source = (
        CHAT_SERVICE_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forbidden = (
        "MemoryQueryScopeDecision",
        "MemoryQueryScopeJudge",
        "MEMORY_QUERY_SCOPE_CURRENT",
        "MEMORY_QUERY_SCOPE_HISTORICAL",
        "MEMORY_QUERY_SCOPE_BOTH",
        "memory_status",
        "scope",
    )

    for pattern in forbidden:

        assert pattern not in source, (
            "ChatService 不应出现："
            f"{pattern}"
        )


def test_relevance_judge_does_not_depend_on_query_scope():
    """
    MemoryRelevanceJudge 允许知道：

        memory_status = current / historical

    但不允许依赖：

        MemoryQueryScopeDecision
        MEMORY_QUERY_SCOPE_*
    """

    source = (
        JUDGE_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forbidden = (
        "MemoryQueryScopeDecision",
        "MEMORY_QUERY_SCOPE",
        "memory_query_scope",
    )

    for pattern in forbidden:

        assert pattern not in source, (
            "MemoryRelevanceJudge 不应依赖 "
            f"Query Scope：{pattern}"
        )

    assert (
        "memory_status" in source
    ), (
        "MemoryRelevanceJudge 应该知道 "
        "memory_status"
    )

    assert (
        "historical" in source
    )

    assert (
        "current" in source
    )


def test_relevance_judge_has_no_orm_dependency():
    """
    Relevance Judge 不应依赖 SQLAlchemy Memory ORM。
    """

    source = (
        JUDGE_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "sqlalchemy" not in source
    )

    assert (
        "backend.models.memory"
        not in source
    )


def test_retrieval_components_still_ignore_lifecycle():
    """
    Dense / BM25 / RRF / Reranker
    继续不知道 Lifecycle。
    """

    for path in RETRIEVAL_SOURCE_PATHS:

        source = path.read_text(
            encoding="utf-8"
        )

        for pattern in (
            "memory_status",
            "historical",
            "MEMORY_STATUS",
        ):

            assert pattern not in source, (
                f"{path.name} 不应出现 "
                f"Lifecycle 相关逻辑：{pattern}"
            )


# ============================================================
# ============================================================
# Test Runner
# ============================================================
# ============================================================


def collect_tests():

    current_globals = globals()

    names = [
        name
        for name in current_globals
        if name.startswith("test_")
        and callable(
            current_globals[name]
        )
    ]

    names.sort()

    return [
        (
            name,
            current_globals[name]
        )
        for name in names
    ]


def main():

    tests = collect_tests()

    passed = []
    failed = []

    print()
    print("=" * 70)
    print(
        "MEMORY RELEVANCE JUDGE "
        "STATUS CONTRACT TESTS"
    )
    print("=" * 70)

    for name, func in tests:

        try:

            func()

        except AssertionError as exc:

            failed.append(
                (
                    name,
                    f"AssertionError: {exc}"
                )
            )

            print(
                f"[FAIL] {name}"
            )

        except Exception as exc:

            failed.append(
                (
                    name,
                    f"{type(exc).__name__}: {exc}"
                )
            )

            print(
                f"[ERROR] {name}"
            )

        else:

            passed.append(
                name
            )

            print(
                f"[PASS] {name}"
            )

    print()
    print("=" * 70)
    print("FAILURE DETAIL")
    print("=" * 70)

    if not failed:

        print(
            "No failure."
        )

    for name, message in failed:

        print()
        print(f"- {name}")
        print(f"  {message}")

    print()
    print("=" * 70)
    print(
        f"TOTAL  : {len(tests)}"
    )
    print(
        f"PASSED : {len(passed)}"
    )
    print(
        f"FAILED : {len(failed)}"
    )
    print("=" * 70)

    if failed:

        return 1

    print()
    print(
        "ALL MEMORY RELEVANCE JUDGE "
        "STATUS CONTRACT TESTS PASSED"
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
