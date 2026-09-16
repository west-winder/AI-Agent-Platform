"""
Historical Memory Retrieval - Deterministic [KEEP] Tests
=========================================================

本文件覆盖：

    MemoryQueryScopeJudge
        Layer 1  Cheap Lexical Rules
        Layer 2  LLM Scope Judge + Failure Contract
        Input Contract

    MemoryRepository
        user_id + scope
            ↓
        current / historical / both

    MemoryReader
        Judge
            ↓
        decision.scope
            ↓
        Repository

    MemoryInjector
        MemoryInjectionItem
            ↓
        Memory Context

设计约束：

    deterministic
        不联网
        不调用真实 DeepSeek
        不加载真实 Embedding 模型
        不加载真实 Reranker 模型
        不写真实 backend.db / test.db

    所有外部依赖均使用 Fake Component / monkeypatch，
    Repository 测试使用隔离的 SQLite In-Memory DB。

Async Contract 注意事项：

    MemoryQueryScopeJudge.judge
    MemoryReader.read

    已经是 async Contract。

    本模块的 test_ 函数全部保持同步
    def test_xxx()，由 run() 把 coroutine 驱动到底
    （SyncQueryScopeJudge / SyncMemoryReader 只是
    把 coroutine 驱动到底的调用适配）：

        1. 直接 python 运行时会真的执行
        2. pytest 会原生收集执行，
           不会被当成 async test 静默跳过
        3. coroutine 一定被 await
        4. 不会产生假 PASS

    本模块不依赖 pytest-asyncio。

运行方式：

    1. 作为脚本运行（项目现有测试约定）：

        .venv/Scripts/python.exe tests/memory/test_memory_read_historical_scope.py

    2. 由 pytest 收集（如果环境中安装了 pytest）：

        pytest tests/memory/test_memory_read_historical_scope.py
"""


import asyncio
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from unittest import mock


# ============================================================
# Project Root Bootstrap
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from backend.database.database import Base  # noqa: E402

# ------------------------------------------------------------
# 注册全部 ORM Model
#
# 与 backend/main.py 保持一致。
# 只导入 Memory / User 会导致 relationship 解析失败。
# ------------------------------------------------------------

from backend.models import agent as agent_model  # noqa: E402,F401
from backend.models import message as message_model  # noqa: E402,F401
from backend.models import model as model_model  # noqa: E402,F401
from backend.models import user as user_model  # noqa: E402,F401
from backend.models.conversation import (  # noqa: E402
    Conversation as ConversationModel,
)

from backend.models.memory import (  # noqa: E402
    MEMORY_STATUS_CURRENT,
    MEMORY_STATUS_HISTORICAL,
    Memory,
)

from backend.memory.memory_read.memory_query_scope import (  # noqa: E402
    ALLOWED_MEMORY_QUERY_SCOPES,
    MEMORY_QUERY_SCOPE_BOTH,
    MEMORY_QUERY_SCOPE_CURRENT,
    MEMORY_QUERY_SCOPE_HISTORICAL,
    MemoryQueryScopeDecision,
)

import backend.memory.memory_read.memory_query_scope_judge as scope_judge_module  # noqa: E402

from backend.memory.memory_read.memory_query_scope_judge import (  # noqa: E402
    MemoryQueryScopeJudge,
)

from backend.memory.memory_read.memory_repository import (  # noqa: E402
    get_memories_for_read,
)

from backend.memory.memory_read.memory_reader import (  # noqa: E402
    MemoryReader,
)

from backend.memory.memory_read.memory_injector import (  # noqa: E402
    MemoryInjector,
    MemoryInjectionItem,
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

from backend.memory.memory_read.memory_relevance_judge import (  # noqa: E402
    JudgeDecision,
)


# ============================================================
# Source Paths
#
# 架构边界静态检查使用。
# ============================================================

READER_SOURCE_PATH = (
    PROJECT_ROOT
    / "backend"
    / "memory"
    / "memory_read"
    / "memory_reader.py"
)

INJECTOR_SOURCE_PATH = (
    PROJECT_ROOT
    / "backend"
    / "memory"
    / "memory_read"
    / "memory_injector.py"
)


# ============================================================
# Test Helpers
# ============================================================


def assert_scope_decision(
    decision,
    expected_scope,
    expected_source
):
    """
    统一校验一次 Scope Decision。
    """

    assert isinstance(
        decision,
        MemoryQueryScopeDecision
    ), (
        "Scope Judge 必须返回 "
        "MemoryQueryScopeDecision，"
        f"实际为：{type(decision)}"
    )

    assert decision.scope == expected_scope, (
        f"scope 期望 {expected_scope}，"
        f"实际为 {decision.scope}"
    )

    assert decision.source == expected_source, (
        f"source 期望 {expected_source}，"
        f"实际为 {decision.source}"
    )

    assert isinstance(decision.reason, str), (
        "reason 必须是 str"
    )

    assert decision.reason.strip(), (
        "reason 不能为空"
    )


# ============================================================
# Async Contract → 同步测试适配
#
# MemoryQueryScopeJudge.judge
# MemoryReader.read
#
# 都已经变成 async Contract。
#
# 本模块的测试函数保持同步 def test_xxx()：
#
#     1. 直接 python 运行时会真的执行
#     2. pytest 会原生收集执行，
#        不会被当成 async test 静默跳过
#     3. coroutine 一定被 await，
#        不会出现 coroutine was never awaited
#     4. 不会产生假 PASS
#
# 本项目当前没有 pytest-asyncio，
# 因此这里只把 coroutine 驱动到底，
# 不复制、不绕过生产逻辑。
# ============================================================


def run(coro):
    """
    在同步测试中真实执行并等待
    async production contract。
    """

    return asyncio.run(coro)


class SyncQueryScopeJudge(MemoryQueryScopeJudge):
    """
    真实 MemoryQueryScopeJudge
    +
    同步调用适配。

    只把 async judge() 的 coroutine
    驱动到底。
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
    驱动到底。
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


class FakeLLM:
    """
    Fake call_llm。

    记录自己被调用的次数与收到的 messages，
    返回预设响应或抛出预设异常。

    注意：

    MemoryQueryScopeJudge 现在
    await call_llm(messages)，

    因此本 Fake 必须保持同样的
    async Calling Contract。
    """

    def __init__(
        self,
        response=None,
        exc=None
    ):
        self._response = response
        self._exc = exc

        self.call_count = 0
        self.received_messages = None

    async def __call__(
        self,
        messages
    ):
        self.call_count += 1

        self.received_messages = messages

        if self._exc is not None:
            raise self._exc

        return self._response


def patch_call_llm(fake_llm):
    """
    将 Scope Judge 模块中的 call_llm
    替换为 Fake。
    """

    return mock.patch.object(
        scope_judge_module,
        "call_llm",
        fake_llm
    )


def llm_json(
    scope,
    reason="fake llm reason"
):
    return json.dumps(
        {
            "scope": scope,
            "reason": reason,
        },
        ensure_ascii=False
    )


# ============================================================
# SQLite In-Memory Test DB
# ============================================================


def new_session():
    """
    创建一个全新的 SQLite In-Memory Session。

    每个测试独立一个数据库，
    避免测试互相污染，
    也不会写入真实 backend.db / test.db。
    """

    engine = create_engine(
        "sqlite:///:memory:"
    )

    Base.metadata.create_all(
        engine
    )

    Session = sessionmaker(
        bind=engine
    )

    return Session()


def insert_memory(
    db,
    user_id,
    content,
    memory_status=MEMORY_STATUS_CURRENT,
    memory_type="fact"
):
    memory = Memory(
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        memory_status=memory_status,
        historical_at=None,
    )

    db.add(memory)
    db.commit()
    db.refresh(memory)

    return memory


def build_two_user_corpus(
    db
):
    """
    user 1:

        current    A / B
        historical C / D

    user 2:

        current    E
        historical F
    """

    insert_memory(
        db,
        user_id=1,
        content="A",
        memory_status=MEMORY_STATUS_CURRENT
    )

    insert_memory(
        db,
        user_id=1,
        content="B",
        memory_status=MEMORY_STATUS_CURRENT
    )

    insert_memory(
        db,
        user_id=1,
        content="C",
        memory_status=MEMORY_STATUS_HISTORICAL
    )

    insert_memory(
        db,
        user_id=1,
        content="D",
        memory_status=MEMORY_STATUS_HISTORICAL
    )

    insert_memory(
        db,
        user_id=2,
        content="E",
        memory_status=MEMORY_STATUS_CURRENT
    )

    insert_memory(
        db,
        user_id=2,
        content="F",
        memory_status=MEMORY_STATUS_HISTORICAL
    )


def contents_of(
    memories
):
    return {
        memory.content
        for memory in memories
    }


# ============================================================
# ============================================================
# Section 1
# Scope Judge - Layer 1 Lexical Rules
# ============================================================
# ============================================================


def test_rule_layer_historical_query():
    """
    Query：

        我以前主要学什么？

    预期：

        scope  = historical
        source = rule

    且 call_llm 不应被调用。
    """

    fake_llm = FakeLLM(
        response=llm_json(
            "current",
            "should not be used"
        )
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我以前主要学什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_HISTORICAL,
        "rule"
    )

    assert fake_llm.call_count == 0, (
        "规则层已经判定 historical，"
        "不应该调用 LLM"
    )


def test_rule_layer_current_query():
    """
    Query：

        我现在主要学什么？

    预期：

        scope  = current
        source = rule

    LLM 不调用。
    """

    fake_llm = FakeLLM(
        response=llm_json("historical")
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我现在主要学什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_CURRENT,
        "rule"
    )

    assert fake_llm.call_count == 0, (
        "规则层已经判定 current，"
        "不应该调用 LLM"
    )


def test_rule_layer_both_query():
    """
    Query：

        我从以前到现在最大的变化是什么？

    预期：

        scope  = both
        source = rule

    LLM 不调用。
    """

    fake_llm = FakeLLM(
        response=llm_json("historical")
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我从以前到现在"
                "最大的变化是什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_BOTH,
        "rule"
    )

    assert fake_llm.call_count == 0, (
        "规则层已经判定 both，"
        "不应该调用 LLM"
    )


def test_rule_layer_defers_mixed_markers_to_llm():
    """
    Query：

        我现在想知道我以前主要学什么？

    第一层不能简单判 both。

    必须进入 LLM。

    Fake LLM 返回 historical 后：

        scope  = historical
        source = llm
    """

    fake_llm = FakeLLM(
        response=llm_json(
            "historical",
            "用户真正询问的是过去状态"
        )
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我现在想知道"
                "我以前主要学什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_HISTORICAL,
        "llm"
    )

    assert fake_llm.call_count == 1, (
        "同时出现 现在 + 以前，"
        "且没有命中明确 both pattern，"
        "第一层必须交给 LLM"
    )


def test_rule_layer_defers_no_marker_query_to_llm():
    """
    Query：

        我为什么从 Python 转向 Java？

    第一层没有明确时间 marker。

    必须进入 LLM。

    Fake LLM 返回 both：

        scope  = both
        source = llm
    """

    fake_llm = FakeLLM(
        response=llm_json(
            "both",
            "需要对比过去与当前技术方向"
        )
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我为什么从 Python "
                "转向 Java？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_BOTH,
        "llm"
    )

    assert fake_llm.call_count == 1, (
        "没有明确时间 marker，"
        "第一层必须交给 LLM"
    )


# ============================================================
# ============================================================
# Section 2
# Scope Judge - Layer 2 LLM Valid Response
# ============================================================
# ============================================================


def test_llm_layer_scope_current():
    fake_llm = FakeLLM(
        response=llm_json("current")
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我的技术栈现状如何？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_CURRENT,
        "llm"
    )


def test_llm_layer_scope_historical():
    fake_llm = FakeLLM(
        response=llm_json("historical")
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我的技术栈经历过什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_HISTORICAL,
        "llm"
    )


def test_llm_layer_scope_both():
    fake_llm = FakeLLM(
        response=llm_json("both")
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我的技术栈经历过什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_BOTH,
        "llm"
    )


def test_llm_layer_scope_is_normalized():
    """
    LLM 返回带大小写 / 空白的合法 scope，
    应该被规范化后接受。
    """

    fake_llm = FakeLLM(
        response=llm_json("  HISTORICAL  ")
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我的技术栈经历过什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_HISTORICAL,
        "llm"
    )


def test_llm_layer_scope_kept_when_reason_missing():
    """
    scope 合法
    但 reason 缺失

    不应该丢弃合法 scope。

    例如：

        {"scope": "historical"}

    预期：

        scope  = historical
        source = llm
        reason 自动补默认说明
    """

    fake_llm = FakeLLM(
        response=json.dumps(
            {
                "scope": "historical"
            }
        )
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我的技术栈经历过什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_HISTORICAL,
        "llm"
    )

    assert "reason" in decision.reason, (
        "reason 缺失时应该自动补充默认说明，"
        f"实际为：{decision.reason}"
    )


def test_llm_layer_scope_kept_when_reason_blank():
    """
    scope 合法
    但 reason 为空字符串 / 空白

    不应该丢弃合法 scope。
    """

    fake_llm = FakeLLM(
        response=json.dumps(
            {
                "scope": "both",
                "reason": "   ",
            }
        )
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我的技术栈经历过什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_BOTH,
        "llm"
    )


# ============================================================
# ============================================================
# Section 3
# Scope Judge - Layer 2 Failure Contract
#
# 所有失败：
#
#     scope  = current
#     source = fallback
# ============================================================
# ============================================================


def assert_fallback_current(
    response=None,
    exc=None
):
    fake_llm = FakeLLM(
        response=response,
        exc=exc
    )

    with patch_call_llm(fake_llm):

        decision = (
            SyncQueryScopeJudge()
            .judge(
                "我的技术栈经历过什么？"
            )
        )

    assert_scope_decision(
        decision,
        MEMORY_QUERY_SCOPE_CURRENT,
        "fallback"
    )


def test_llm_fallback_on_empty_response():
    """
    LLM 返回空字符串。
    """

    assert_fallback_current(
        response=""
    )


def test_llm_fallback_on_blank_response():
    """
    LLM 返回全空白内容。
    """

    assert_fallback_current(
        response="   \n\t  "
    )


def test_llm_fallback_on_none_response():
    """
    LLM 返回 None。
    """

    assert_fallback_current(
        response=None
    )


def test_llm_fallback_on_non_string_response():
    """
    LLM 返回非字符串对象。
    """

    assert_fallback_current(
        response=123
    )


def test_llm_fallback_on_invalid_json():
    """
    LLM 返回非法 JSON。
    """

    assert_fallback_current(
        response="这不是 JSON"
    )


def test_llm_fallback_on_non_object_json():
    """
    LLM 返回的 JSON 不是 object。
    """

    assert_fallback_current(
        response='["historical"]'
    )


def test_llm_fallback_on_missing_scope():
    """
    JSON object 中 scope 缺失。
    """

    assert_fallback_current(
        response='{"reason": "忘了写 scope"}'
    )


def test_llm_fallback_on_illegal_scope():
    """
    scope 非法，例如 past。
    """

    assert_fallback_current(
        response=llm_json("past")
    )


def test_llm_fallback_on_exception():
    """
    call_llm 抛出异常。
    """

    assert_fallback_current(
        exc=RuntimeError(
            "DeepSeek unavailable"
        )
    )


# ============================================================
# ============================================================
# Section 4
# Scope Judge - Input Contract
# ============================================================
# ============================================================


def test_judge_rejects_non_string_query():
    """
    query 非 str
        ↓
    TypeError
    """

    judge = SyncQueryScopeJudge()

    fake_llm = FakeLLM(
        response=llm_json("current")
    )

    for bad_query in (
        123,
        None,
        ["我以前学什么"],
        {"query": "我以前学什么"},
    ):

        with patch_call_llm(fake_llm):

            try:

                judge.judge(
                    bad_query
                )

            except TypeError:

                continue

            raise AssertionError(
                f"query={bad_query!r} "
                "应该抛出 TypeError"
            )

    assert fake_llm.call_count == 0, (
        "输入校验失败时不应该调用 LLM"
    )


def test_judge_rejects_empty_query():
    """
    query 空字符串
        ↓
    ValueError
    """

    judge = SyncQueryScopeJudge()

    try:

        judge.judge("")

    except ValueError:

        return

    raise AssertionError(
        "空 query 应该抛出 ValueError"
    )


def test_judge_rejects_blank_query():
    """
    query 全空白
        ↓
    ValueError
    """

    judge = SyncQueryScopeJudge()

    for bad_query in (
        "   ",
        "\n\t  ",
    ):

        try:

            judge.judge(
                bad_query
            )

        except ValueError:

            continue

        raise AssertionError(
            f"query={bad_query!r} "
            "应该抛出 ValueError"
        )


# ============================================================
# ============================================================
# Section 5
# MemoryRepository
#
# user_id + scope
#     ↓
# 对应 Memory corpus
# ============================================================
# ============================================================


def test_repository_scope_current_only():
    """
    scope = current

    只返回 user 1 的 A / B。
    """

    db = new_session()

    build_two_user_corpus(
        db
    )

    memories = get_memories_for_read(
        db,
        1,
        scope=MEMORY_QUERY_SCOPE_CURRENT
    )

    assert contents_of(
        memories
    ) == {"A", "B"}, (
        "scope=current 只应返回 "
        f"A/B，实际为 {contents_of(memories)}"
    )


def test_repository_scope_historical_only():
    """
    scope = historical

    只返回 user 1 的 C / D。
    """

    db = new_session()

    build_two_user_corpus(
        db
    )

    memories = get_memories_for_read(
        db,
        1,
        scope=MEMORY_QUERY_SCOPE_HISTORICAL
    )

    assert contents_of(
        memories
    ) == {"C", "D"}, (
        "scope=historical 只应返回 "
        f"C/D，实际为 {contents_of(memories)}"
    )


def test_repository_scope_both():
    """
    scope = both

    返回 user 1 的 A / B / C / D。
    """

    db = new_session()

    build_two_user_corpus(
        db
    )

    memories = get_memories_for_read(
        db,
        1,
        scope=MEMORY_QUERY_SCOPE_BOTH
    )

    assert contents_of(
        memories
    ) == {"A", "B", "C", "D"}, (
        "scope=both 应返回 "
        f"A/B/C/D，实际为 {contents_of(memories)}"
    )


def test_repository_never_returns_other_users():
    """
    无论哪种 scope，
    都不能出现 user 2 的 E / F。
    """

    db = new_session()

    build_two_user_corpus(
        db
    )

    for scope in (
        MEMORY_QUERY_SCOPE_CURRENT,
        MEMORY_QUERY_SCOPE_HISTORICAL,
        MEMORY_QUERY_SCOPE_BOTH,
    ):

        memories = get_memories_for_read(
            db,
            1,
            scope=scope
        )

        contents = contents_of(
            memories
        )

        assert "E" not in contents, (
            f"scope={scope} 泄漏了 "
            "user 2 的 current Memory"
        )

        assert "F" not in contents, (
            f"scope={scope} 泄漏了 "
            "user 2 的 historical Memory"
        )


def test_repository_default_scope_is_current():
    """
    Backward Compatibility Contract

    不显式传 scope：

        get_memories_for_read(
            db,
            user_id
        )

    仍然保持 current only。
    """

    db = new_session()

    build_two_user_corpus(
        db
    )

    memories = get_memories_for_read(
        db,
        1
    )

    assert contents_of(
        memories
    ) == {"A", "B"}, (
        "默认 scope 必须保持 current only，"
        f"实际为 {contents_of(memories)}"
    )


def test_repository_rejects_invalid_scope():
    """
    scope = past

        ↓

    ValueError
    """

    db = new_session()

    build_two_user_corpus(
        db
    )

    try:

        get_memories_for_read(
            db,
            1,
            scope="past"
        )

    except ValueError:

        return

    raise AssertionError(
        "非法 scope='past' "
        "应该抛出 ValueError"
    )


def test_repository_rejects_non_string_scope():
    """
    scope 非 str

        ↓

    ValueError
    """

    db = new_session()

    build_two_user_corpus(
        db
    )

    try:

        get_memories_for_read(
            db,
            1,
            scope=None
        )

    except ValueError:

        return

    raise AssertionError(
        "scope=None 应该抛出 ValueError"
    )


# ============================================================
# ============================================================
# Section 6
# MemoryReader Orchestration
#
# 重点：
#
# 数据流，而不是 Retrieval Algorithm。
#
# MemoryReader 只应该把：
#
#     scope_decision.scope
#
# 传给 Repository。
# ============================================================
# ============================================================


@dataclass
class FakeMemory:
    """
    Controlled Test 使用的 Memory。

    只需要满足 MemoryReader 使用的
    .content / .memory_status。
    """

    content: str

    memory_status: str = (
        MEMORY_STATUS_CURRENT
    )

    id: int = 0

    memory_type: str = "fact"


class FakeScopeJudge:
    """
    Fake Scope Judge。

    不调用 DeepSeek。

    注意：

    MemoryReader 现在
    await self._scope_judge.judge(...)，

    因此本 Fake 必须保持同样的
    async Calling Contract。
    """

    def __init__(
        self,
        scope,
        reason="fake scope reason",
        source="rule"
    ):
        self._decision = (
            MemoryQueryScopeDecision(
                scope=scope,
                reason=reason,
                source=source
            )
        )

        self.call_count = 0
        self.received_queries = []

    async def judge(
        self,
        query
    ):
        self.call_count += 1

        self.received_queries.append(
            query
        )

        return self._decision


class RecordingRepository:
    """
    Fake Repository。

    记录自己实际收到的 scope。
    """

    MISSING = object()

    def __init__(
        self,
        memories
    ):
        self._memories = list(
            memories
        )

        self.call_count = 0
        self.received_user_id = None
        self.received_scope = self.MISSING
        self.received_kwargs = None

    def __call__(
        self,
        db,
        user_id,
        scope=MISSING
    ):
        self.call_count += 1

        self.received_user_id = user_id
        self.received_scope = scope

        return list(
            self._memories
        )


class FakeDenseRetriever:
    """
    Fake Dense Retriever。

    不加载真实 Embedding。
    """

    def __init__(self):
        self.call_count = 0

    def search(
        self,
        query,
        memories,
        top_n
    ):
        self.call_count += 1

        return [
            DenseRetrievedMemory(
                index=index,
                memory=memory,
                similarity=float(
                    len(memories) - index
                )
            )
            for index, memory in enumerate(
                memories
            )
        ]


class FakeBM25Retriever:
    """
    Fake BM25 Retriever。

    不构建真实 BM25 索引。
    """

    def __init__(self):
        self.call_count = 0

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
        self.call_count += 1

        return []


class FakeRRF:
    """
    Fake RRF。

    默认返回 Dense / BM25 ranking 的并集。

    forced_results 不为 None 时，
    直接返回指定结果，
    用于构造 Early Return 场景。
    """

    def __init__(
        self,
        forced_results=None
    ):
        self._forced_results = (
            forced_results
        )

        self.call_count = 0
        self.received_rankings = None

    def __call__(
        self,
        rankings,
        top_n=None
    ):
        self.call_count += 1

        self.received_rankings = [
            list(ranking)
            for ranking in rankings
        ]

        if self._forced_results is not None:

            return list(
                self._forced_results
            )

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
    """
    Fake Reranker。

    不加载 Cross Encoder。
    """

    def __init__(self):
        self.call_count = 0

    def rerank(
        self,
        query,
        texts
    ):
        self.call_count += 1

        return [
            RerankResult(
                index=index,
                text=text,
                score=float(
                    len(texts) - index
                )
            )
            for index, text in enumerate(
                texts
            )
        ]


class FakeJudge:
    """
    Fake Relevance Judge。

    不调用 DeepSeek。

    MemoryRelevanceJudge Contract 已演进为：

        candidates: list[MemoryRelevanceCandidate]

    这里只记录收到的 candidates，
    不解释 memory_status。

    注意：

    MemoryReader 现在 await self._judge.judge(...)，
    因此本 Fake 必须保持 async Calling Contract。
    """

    def __init__(
        self,
        selected=True
    ):
        self._selected = selected

        self.call_count = 0
        self.received_candidates = None

    async def judge(
        self,
        query,
        candidates
    ):
        self.call_count += 1

        self.received_candidates = list(
            candidates
        )

        return [
            JudgeDecision(
                index=index,
                selected=self._selected,
                reason="fake judge"
            )
            for index in range(
                len(candidates)
            )
        ]


class FakeInjector:
    """
    Fake Injector。

    只记录自己收到的内容。
    """

    def __init__(self):
        self.call_count = 0
        self.received_items = None

    def build_context(
        self,
        items
    ):
        self.call_count += 1

        self.received_items = list(
            items
        )

        return "<memory_context>fake</memory_context>"


def build_reader(
    scope,
    memories,
    rrf_results=None,
    injector=None
):
    """
    构造一个全 Fake 的 MemoryReader。

    返回：

        reader
        scope_judge
        repository
    """

    scope_judge = FakeScopeJudge(
        scope=scope
    )

    repository = RecordingRepository(
        memories
    )

    reader = SyncMemoryReader(
        repository_callable=repository,
        scope_judge=scope_judge,
        dense_retriever=FakeDenseRetriever(),
        bm25_retriever=FakeBM25Retriever(),
        rrf_callable=FakeRRF(
            forced_results=rrf_results
        ),
        reranker=FakeReranker(),
        judge=FakeJudge(),
        injector=(
            injector
            if injector is not None
            else FakeInjector()
        ),
    )

    return (
        reader,
        scope_judge,
        repository
    )


def assert_scope_decision_preserved(
    result,
    expected_scope
):
    """
    MemoryReadResult.scope_decision
    必须完整保留：

        scope
        reason
        source
    """

    decision = (
        result.scope_decision
    )

    assert decision is not None, (
        "MemoryReadResult.scope_decision "
        "不能为 None"
    )

    assert isinstance(
        decision,
        MemoryQueryScopeDecision
    ), (
        "scope_decision 必须是 "
        "MemoryQueryScopeDecision"
    )

    assert decision.scope == expected_scope, (
        f"scope_decision.scope 期望 "
        f"{expected_scope}，"
        f"实际为 {decision.scope}"
    )

    assert decision.reason == (
        "fake scope reason"
    ), (
        "scope_decision.reason 必须完整保留，"
        f"实际为 {decision.reason}"
    )

    assert decision.source == "rule", (
        "scope_decision.source 必须完整保留，"
        f"实际为 {decision.source}"
    )


def test_reader_passes_current_scope_to_repository():
    """
    Fake ScopeJudge 返回 current

        ↓

    Repository 必须收到 current。
    """

    reader, _, repository = (
        build_reader(
            scope=MEMORY_QUERY_SCOPE_CURRENT,
            memories=[
                FakeMemory(
                    "用户现在在学习 LangGraph"
                )
            ]
        )
    )

    result = reader.read(
        db=None,
        user_id=1,
        query="我现在在学什么？",
        top_n=5,
        top_k=3
    )

    assert (
        repository.received_scope
        == MEMORY_QUERY_SCOPE_CURRENT
    ), (
        "Repository 应该收到 current，"
        f"实际为 {repository.received_scope}"
    )

    assert_scope_decision_preserved(
        result,
        MEMORY_QUERY_SCOPE_CURRENT
    )


def test_reader_passes_historical_scope_to_repository():
    """
    Fake ScopeJudge 返回 historical

        ↓

    Repository 必须收到 historical。
    """

    reader, _, repository = (
        build_reader(
            scope=(
                MEMORY_QUERY_SCOPE_HISTORICAL
            ),
            memories=[
                FakeMemory(
                    "用户以前主要学习 Java",
                    memory_status=(
                        MEMORY_STATUS_HISTORICAL
                    )
                )
            ]
        )
    )

    result = reader.read(
        db=None,
        user_id=1,
        query="我以前主要学什么？",
        top_n=5,
        top_k=3
    )

    assert (
        repository.received_scope
        == MEMORY_QUERY_SCOPE_HISTORICAL
    ), (
        "Repository 应该收到 historical，"
        f"实际为 {repository.received_scope}"
    )

    assert_scope_decision_preserved(
        result,
        MEMORY_QUERY_SCOPE_HISTORICAL
    )


def test_reader_passes_both_scope_to_repository():
    """
    Fake ScopeJudge 返回 both

        ↓

    Repository 必须收到 both。
    """

    reader, _, repository = (
        build_reader(
            scope=MEMORY_QUERY_SCOPE_BOTH,
            memories=[
                FakeMemory(
                    "用户现在在学习 LangGraph"
                ),
                FakeMemory(
                    "用户以前主要学习 Java",
                    memory_status=(
                        MEMORY_STATUS_HISTORICAL
                    )
                ),
            ]
        )
    )

    result = reader.read(
        db=None,
        user_id=1,
        query="我从以前到现在有什么变化？",
        top_n=5,
        top_k=3
    )

    assert (
        repository.received_scope
        == MEMORY_QUERY_SCOPE_BOTH
    ), (
        "Repository 应该收到 both，"
        f"实际为 {repository.received_scope}"
    )

    assert_scope_decision_preserved(
        result,
        MEMORY_QUERY_SCOPE_BOTH
    )


def test_reader_calls_repository_once_with_scope():
    """
    Repository 只应被调用一次，
    并且 scope 以 keyword 形式传入。
    """

    reader, scope_judge, repository = (
        build_reader(
            scope=MEMORY_QUERY_SCOPE_BOTH,
            memories=[
                FakeMemory(
                    "用户现在在学习 LangGraph"
                )
            ]
        )
    )

    reader.read(
        db=None,
        user_id=7,
        query="我现在在学什么？",
        top_n=5,
        top_k=3
    )

    assert scope_judge.call_count == 1, (
        "Scope Judge 应该只被调用一次"
    )

    assert repository.call_count == 1, (
        "Repository 应该只被调用一次"
    )

    assert (
        repository.received_user_id == 7
    ), (
        "Repository 应该收到原始 user_id"
    )

    assert (
        repository.received_scope
        is not RecordingRepository.MISSING
    ), (
        "MemoryReader 必须显式把 "
        "decision.scope 传给 Repository"
    )


def test_reader_preserves_scope_decision_on_empty_corpus():
    """
    Repository 返回 []

        ↓

    Early Return

    不能因为 Pipeline 提前返回
    而丢失 scope_decision。
    """

    reader, _, repository = (
        build_reader(
            scope=MEMORY_QUERY_SCOPE_HISTORICAL,
            memories=[]
        )
    )

    result = reader.read(
        db=None,
        user_id=1,
        query="我以前主要学什么？",
        top_n=5,
        top_k=3
    )

    assert result.items == [], (
        "Empty Corpus 应返回空 items"
    )

    assert result.memory_context == "", (
        "Empty Corpus 应返回空 context"
    )

    assert (
        repository.received_scope
        == MEMORY_QUERY_SCOPE_HISTORICAL
    ), (
        "即使 Early Return，"
        "Repository 也应该收到 historical"
    )

    assert_scope_decision_preserved(
        result,
        MEMORY_QUERY_SCOPE_HISTORICAL
    )


def test_reader_preserves_scope_decision_on_empty_rrf():
    """
    RRF 返回 []

        ↓

    Early Return

    不能因为 Pipeline 提前返回
    而丢失 scope_decision。
    """

    reader, _, repository = (
        build_reader(
            scope=MEMORY_QUERY_SCOPE_BOTH,
            memories=[
                FakeMemory(
                    "用户现在在学习 LangGraph"
                )
            ],
            rrf_results=[]
        )
    )

    result = reader.read(
        db=None,
        user_id=1,
        query="我从以前到现在有什么变化？",
        top_n=5,
        top_k=3
    )

    assert result.items == [], (
        "RRF 为空时应返回空 items"
    )

    assert_scope_decision_preserved(
        result,
        MEMORY_QUERY_SCOPE_BOTH
    )


def test_reader_preserves_scope_decision_on_full_path():
    """
    完整 Pipeline 路径下，
    scope_decision 也必须完整保留。
    """

    reader, _, _ = build_reader(
        scope=MEMORY_QUERY_SCOPE_BOTH,
        memories=[
            FakeMemory(
                "用户现在在学习 LangGraph"
            ),
            FakeMemory(
                "用户以前主要学习 Java",
                memory_status=(
                    MEMORY_STATUS_HISTORICAL
                )
            ),
        ]
    )

    result = reader.read(
        db=None,
        user_id=1,
        query="我从以前到现在有什么变化？",
        top_n=5,
        top_k=3
    )

    assert len(result.items) == 2, (
        "Full Path 应返回全部 Candidate"
    )

    assert result.memory_context, (
        "Full Path 应生成 memory_context"
    )

    assert_scope_decision_preserved(
        result,
        MEMORY_QUERY_SCOPE_BOTH
    )


# ============================================================
# ============================================================
# Section 7
# MemoryInjector
#
# 根据当前真实 MemoryInjectionItem Contract 测试。
# ============================================================
# ============================================================


def test_injector_current_item():
    """
    current item

    输出 context 中明确存在：

        memory_status: current
    """

    injector = MemoryInjector()

    context = injector.build_context(
        [
            MemoryInjectionItem(
                content="用户现在在学习 LangGraph",
                memory_status=(
                    MEMORY_STATUS_CURRENT
                )
            )
        ]
    )

    assert (
        "memory_status: current"
        in context
    ), (
        "context 中必须明确出现 "
        "memory_status: current"
    )

    assert (
        "用户现在在学习 LangGraph"
        in context
    )


def test_injector_historical_item():
    """
    historical item

    输出 context 中明确存在：

        memory_status: historical
    """

    injector = MemoryInjector()

    context = injector.build_context(
        [
            MemoryInjectionItem(
                content="用户以前主要学习 Java",
                memory_status=(
                    MEMORY_STATUS_HISTORICAL
                )
            )
        ]
    )

    assert (
        "memory_status: historical"
        in context
    ), (
        "context 中必须明确出现 "
        "memory_status: historical"
    )

    assert (
        "用户以前主要学习 Java"
        in context
    )


def test_injector_historical_header_warns_not_current():
    """
    Context Header 必须明确说明：

        historical 不代表用户当前状态。
    """

    injector = MemoryInjector()

    context = injector.build_context(
        [
            MemoryInjectionItem(
                content="用户以前主要学习 Java",
                memory_status=(
                    MEMORY_STATUS_HISTORICAL
                )
            )
        ]
    )

    header = context.split(
        "<memory_context>"
    )[0]

    assert (
        "不再代表用户当前状态"
        in header
    ), (
        "Context Header 必须明确说明 "
        "historical 不代表用户当前状态"
    )


def test_injector_mixed_statuses():
    """
    current + historical 混合

    两条状态都不能丢失。
    """

    injector = MemoryInjector()

    context = injector.build_context(
        [
            MemoryInjectionItem(
                content="用户现在在学习 LangGraph",
                memory_status=(
                    MEMORY_STATUS_CURRENT
                )
            ),
            MemoryInjectionItem(
                content="用户以前主要学习 Java",
                memory_status=(
                    MEMORY_STATUS_HISTORICAL
                )
            ),
        ]
    )

    assert (
        "memory_status: current"
        in context
    )

    assert (
        "memory_status: historical"
        in context
    )

    assert (
        "用户现在在学习 LangGraph"
        in context
    )

    assert (
        "用户以前主要学习 Java"
        in context
    )


def test_injector_empty_list():
    """
    空 list

        ↓

    ""
    """

    injector = MemoryInjector()

    assert (
        injector.build_context(
            []
        )
        == ""
    )


def test_injector_rejects_wrong_item_type():
    """
    错误 item 类型

        ↓

    TypeError
    """

    injector = MemoryInjector()

    try:

        injector.build_context(
            [
                "用户现在在学习 LangGraph"
            ]
        )

    except TypeError:

        return

    raise AssertionError(
        "str item 应该抛出 TypeError"
    )


def test_injector_rejects_empty_content():
    """
    空 content

        ↓

    ValueError
    """

    injector = MemoryInjector()

    try:

        injector.build_context(
            [
                MemoryInjectionItem(
                    content="   ",
                    memory_status=(
                        MEMORY_STATUS_CURRENT
                    )
                )
            ]
        )

    except ValueError:

        return

    raise AssertionError(
        "空 content 应该抛出 ValueError"
    )


def test_injector_rejects_invalid_memory_status():
    """
    非法 memory_status

        ↓

    ValueError
    """

    injector = MemoryInjector()

    try:

        injector.build_context(
            [
                MemoryInjectionItem(
                    content="用户现在在学习 LangGraph",
                    memory_status="past"
                )
            ]
        )

    except ValueError:

        return

    raise AssertionError(
        "非法 memory_status 应该抛出"
        " ValueError"
    )


def test_injector_has_no_sqlalchemy_dependency():
    """
    Injector 不应依赖：

        SQLAlchemy Memory ORM
    """

    source = (
        INJECTOR_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert "sqlalchemy" not in source, (
        "MemoryInjector 不应依赖 SQLAlchemy"
    )

    assert (
        "backend.models.memory"
        not in source
    ), (
        "MemoryInjector 不应依赖 "
        "Memory ORM"
    )

    assert "import Memory" not in source, (
        "MemoryInjector 不应导入 Memory ORM"
    )


# ============================================================
# ============================================================
# Section 8
# Architecture Boundary
#
# 关注点分离静态检查。
# ============================================================
# ============================================================


def test_reader_does_not_branch_on_scope_literals():
    """
    MemoryReader 不应该自行解释：

        current / historical / both

    它只应该：

        调 Scope Judge
        保存 Scope Decision
        把 decision.scope 传给 Repository
    """

    source = (
        READER_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forbidden_patterns = (
        '== "current"',
        '== "historical"',
        '== "both"',
        '!= "current"',
        '!= "historical"',
        '!= "both"',
        "MEMORY_QUERY_SCOPE_HISTORICAL",
        "MEMORY_QUERY_SCOPE_BOTH",
    )

    for pattern in forbidden_patterns:

        assert pattern not in source, (
            "MemoryReader 不应出现 "
            f"scope 业务分支：{pattern}"
        )


def test_injector_does_not_know_query_scope():
    """
    Injector 允许知道：

        memory_status = current / historical

    但不应知道 Query Scope：

        both
    """

    source = (
        INJECTOR_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "MEMORY_QUERY_SCOPE"
        not in source
    ), (
        "MemoryInjector 不应依赖 "
        "Memory Query Scope 常量"
    )

    assert (
        "memory_query_scope"
        not in source
    ), (
        "MemoryInjector 不应依赖 "
        "Memory Query Scope 模块"
    )

    assert (
        MEMORY_QUERY_SCOPE_BOTH
        not in source
    ), (
        "MemoryInjector 不应知道 "
        "Query Scope：both"
    )


# ============================================================
# ============================================================
# Section 9
# Reader → Injector Status Contract
#
# 期望契约：
#
# MemoryReader 最终交给 Injector 的
# 应该是 MemoryInjectionItem，
# 并且保留每条 Memory 自身的
# memory_status。
# ============================================================
# ============================================================


def test_reader_preserves_memory_status_into_context():
    """
    完整 Pipeline
    +
    真实 MemoryInjector

    最终 memory_context 应该保留：

        memory_status: current
        memory_status: historical
    """

    reader, _, _ = build_reader(
        scope=MEMORY_QUERY_SCOPE_BOTH,
        memories=[
            FakeMemory(
                "用户现在在学习 LangGraph"
            ),
            FakeMemory(
                "用户以前主要学习 Java",
                memory_status=(
                    MEMORY_STATUS_HISTORICAL
                )
            ),
        ],
        injector=MemoryInjector()
    )

    result = reader.read(
        db=None,
        user_id=1,
        query="我从以前到现在有什么变化？",
        top_n=5,
        top_k=3
    )

    assert (
        "memory_status: current"
        in result.memory_context
    ), (
        "memory_context 应保留 "
        "memory_status: current"
    )

    assert (
        "memory_status: historical"
        in result.memory_context
    ), (
        "memory_context 应保留 "
        "memory_status: historical"
    )


# ============================================================
# ============================================================
# Test Runner
#
# 项目现有测试采用脚本式运行。
#
# 这里同时保持 pytest 可收集。
# ============================================================
# ============================================================


def collect_tests():
    """
    收集本模块中所有 test_ 函数。

    按函数名排序，保证执行顺序确定。
    """

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
        "HISTORICAL MEMORY RETRIEVAL "
        "DETERMINISTIC TESTS"
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
        "ALL HISTORICAL MEMORY RETRIEVAL "
        "TESTS PASSED"
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
