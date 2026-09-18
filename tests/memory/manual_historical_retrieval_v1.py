"""
Historical Memory Retrieval V1
[MANUAL KEEP] 真实运行验证
===========================================================

本文件不属于每次自动回归。

原因：

    真实 Embedding
    真实 BM25
    真实 Reranker
    真实 DeepSeek Scope Judge
    真实 DeepSeek Relevance Judge
    真实 SQLite Read

可能存在 LLM nondeterminism。


本轮使用的全部组件都是真实组件：

    MemoryQueryScopeJudge     真实
    MemoryRepository          真实
    DenseMemoryRetriever      真实
    BM25Retriever             真实
    reciprocal_rank_fusion    真实
    CrossEncoderReranker      真实
    MemoryRelevanceJudge      真实
    MemoryInjector            真实


唯一的两个 Wrapper 都是纯 Pass-through Recorder：

    RecordingRepository
        记录 corpus，然后原样调用真实 repository

    RecordingInjector
        校验 item 类型，记录 status，
        然后原样调用真实 MemoryInjector

两者都不改变任何行为。


不写数据库：

    只运行 Memory Read。
    运行前后会对比 memories 表行数 / 最大 id / 文件哈希。


运行：

    .venv/Scripts/python.exe tests/memory/manual_historical_retrieval_v1.py
"""


import asyncio
import sys
import hashlib
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from backend.database.database import (  # noqa: E402
    SessionLocal,
)

from backend.memory.memory_read.memory_reader import (  # noqa: E402
    MemoryReader,
)

from backend.memory.memory_read.memory_repository import (  # noqa: E402
    get_memories_for_read,
)

from backend.memory.memory_read.memory_injector import (  # noqa: E402
    MemoryInjector,
    MemoryInjectionItem,
)

import backend.memory.memory_read.memory_query_scope_judge as scope_judge_module  # noqa: E402

from backend.services.llm_service import (  # noqa: E402
    call_llm_structured as real_call_llm_structured,
)


# ============================================================
# Async Contract → 同步调用适配
#
# MemoryReader.read 已经是 async Contract。
#
# 本文件是 [MANUAL KEEP] 真实运行验证，
# main() 保持同步（脚本式运行），
# 这里只把 coroutine 驱动到底。
# ============================================================


def run(coro):
    """
    在同步脚本中真实执行并等待
    async production contract。
    """

    return asyncio.run(coro)


class SyncMemoryReader(MemoryReader):
    """
    真实 MemoryReader
    +
    同步调用适配。
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


DB_PATH = PROJECT_ROOT / "test.db"

REAL_USER_ID = 1

TOP_N = 20

TOP_K = 5


# ============================================================
# Cases
# ============================================================

CASES = [
    {
        "name": "CASE 1 - current / rule",
        "query": "我现在主要在学什么？",
        "expected_scope": "current",
        "expected_source": "rule",
        "expected_corpus_statuses": {
            "current"
        },
    },
    {
        "name": "CASE 2 - historical / rule",
        "query": "我以前主要在学什么？",
        "expected_scope": "historical",
        "expected_source": "rule",
        "expected_corpus_statuses": {
            "historical"
        },
    },
    {
        "name": "CASE 3 - both / rule",
        "query": "我从以前到现在最大的变化是什么？",
        "expected_scope": "both",
        "expected_source": "rule",
        "expected_corpus_statuses": {
            "current",
            "historical",
        },
    },
    {
        "name": "CASE 4 - both / llm",
        "query": (
            "我为什么从 AI Agent "
            "转向安卓开发？"
        ),
        "expected_scope": "both",
        "expected_source": "llm",
        "expected_corpus_statuses": {
            "current",
            "historical",
        },
    },
]


# ============================================================
# Pass-through Recorders
# ============================================================


class RecordingRepository:
    """
    纯 Pass-through。

    记录 Repository 返回的 corpus，
    然后原样返回。
    """

    def __init__(
        self,
        real_repository
    ):
        self._real_repository = (
            real_repository
        )

        self.received_scope = None
        self.received_user_id = None
        self.last_corpus = None

    def __call__(
        self,
        db,
        user_id,
        scope=None
    ):
        self.received_user_id = user_id
        self.received_scope = scope

        corpus = self._real_repository(
            db,
            user_id,
            scope=scope
        )

        self.last_corpus = list(
            corpus
        )

        return corpus


class RecordingInjector:
    """
    纯 Pass-through。

    校验 Injector 实际收到的对象类型，
    记录 content / memory_status，
    然后原样调用真实 MemoryInjector。
    """

    def __init__(self):
        self._real_injector = (
            MemoryInjector()
        )

        self.received_items = []
        self.all_items_are_injection_items = (
            True
        )

    def build_context(
        self,
        items
    ):
        for item in items:

            if not isinstance(
                item,
                MemoryInjectionItem
            ):
                self.all_items_are_injection_items = (
                    False
                )

        self.received_items = [
            (
                item.content,
                item.memory_status
            )
            for item in items
        ]

        return (
            self._real_injector
            .build_context(
                items
            )
        )


class RecordingScopeLLM:
    """
    纯 Pass-through。

    记录 Scope Judge 是否真的调用了 LLM，
    然后原样调用真实 call_llm_structured。

    注意：

    MemoryQueryScopeJudge 现在：

        await call_llm_structured(
            messages=...,
            output_model=MemoryQueryScopeLLMOutput,
        )

    因此本 Recorder 必须保持同样的
    keyword-only Calling Contract，
    并且原样透传 output_model，
    由真实 structured_completion 完成
    Provider 侧解析与校验。
    """

    def __init__(
        self,
        real_call_llm_structured
    ):
        self._real_call_llm_structured = (
            real_call_llm_structured
        )

        self.call_count = 0

    async def __call__(
        self,
        *,
        messages,
        output_model,
        model=None
    ):
        self.call_count += 1

        return await self._real_call_llm_structured(
            messages=messages,
            output_model=output_model,
            model=model,
        )


# ============================================================
# DB Snapshot
# ============================================================


def db_snapshot():
    """
    只读读取 memories 表状态指纹。
    """

    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    try:

        cur = conn.cursor()

        cur.execute(
            "SELECT COUNT(*) FROM memories"
        )

        row_count = cur.fetchone()[0]

        cur.execute(
            "SELECT COALESCE(MAX(id), 0) "
            "FROM memories"
        )

        max_id = cur.fetchone()[0]

        cur.execute(
            "SELECT id, user_id, "
            "memory_status, content "
            "FROM memories "
            "ORDER BY id"
        )

        rows = cur.fetchall()

    finally:

        conn.close()

    file_hash = hashlib.sha256(
        DB_PATH.read_bytes()
    ).hexdigest()

    return {
        "row_count": row_count,
        "max_id": max_id,
        "rows": rows,
        "file_hash": file_hash,
    }


# ============================================================
# Print Helpers
# ============================================================


def print_section(
    title
):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_corpus(
    corpus
):
    """
    Repository Corpus

    id / memory_status / content
    """

    if not corpus:

        print(
            "  [EMPTY] Repository 返回空 corpus"
        )

        return

    for memory in corpus:

        print(
            f"  - id={memory.id} "
            f"status={memory.memory_status} "
            f"content={memory.content}"
        )


def print_items(
    items
):
    """
    最终 Rerank / Judge 结果。
    """

    if not items:

        print(
            "  [EMPTY] 没有进入最终 items"
        )

        return

    for index, item in enumerate(
        items,
        start=1
    ):

        print(
            f"  [{index}] "
            f"id={item.memory.id} "
            f"status={item.memory.memory_status}"
        )

        print(
            f"      content       : "
            f"{item.memory.content}"
        )

        print(
            f"      rerank_score  : "
            f"{item.rerank_score}"
        )

        print(
            f"      judge_selected: "
            f"{item.judge_selected}"
        )

        print(
            f"      judge_reason  : "
            f"{item.judge_reason}"
        )


def status_distribution(
    rows
):
    distribution = {}

    for row in rows:

        key = (
            row.memory_status
            if hasattr(
                row,
                "memory_status"
            )
            else row
        )

        distribution[key] = (
            distribution.get(
                key,
                0
            )
            + 1
        )

    return distribution


# ============================================================
# Main
# ============================================================


def main():
    print_section(
        "HISTORICAL MEMORY RETRIEVAL V1 "
        "[MANUAL KEEP] REAL RUN"
    )

    print()
    print(
        f"DB          : {DB_PATH}"
    )
    print(
        f"USER_ID     : {REAL_USER_ID}"
    )
    print(
        f"TOP_N       : {TOP_N}"
    )
    print(
        f"TOP_K       : {TOP_K}"
    )

    # --------------------------------------------------------
    # DB Before
    # --------------------------------------------------------

    before = db_snapshot()

    print()
    print(
        "[DB BEFORE] "
        f"row_count={before['row_count']} "
        f"max_id={before['max_id']}"
    )
    print(
        "[DB BEFORE] "
        f"sha256={before['file_hash'][:16]}..."
    )

    # --------------------------------------------------------
    # Real Components
    # --------------------------------------------------------

    scope_llm_recorder = (
        RecordingScopeLLM(
            real_call_llm_structured
        )
    )

    scope_judge_module.call_llm_structured = (
        scope_llm_recorder
    )

    repository_recorder = (
        RecordingRepository(
            get_memories_for_read
        )
    )

    injector_recorder = (
        RecordingInjector()
    )

    reader = SyncMemoryReader(
        repository_callable=(
            repository_recorder
        ),
        injector=injector_recorder,
    )

    db = SessionLocal()

    results = []

    try:

        for case in CASES:

            print_section(
                case["name"]
            )

            query = case["query"]

            print()
            print("1. QUERY")
            print(f"   {query}")

            scope_llm_before = (
                scope_llm_recorder
                .call_count
            )

            # --------------------------------------------
            # Real Memory Read
            # --------------------------------------------

            result = reader.read(
                db=db,
                user_id=REAL_USER_ID,
                query=query,
                top_n=TOP_N,
                top_k=TOP_K
            )

            scope_llm_calls = (
                scope_llm_recorder
                .call_count
                - scope_llm_before
            )

            # --------------------------------------------
            # 2. Scope Decision
            # --------------------------------------------

            decision = (
                result.scope_decision
            )

            print()
            print("2. SCOPE DECISION")
            print(
                f"   scope  : {decision.scope}"
            )
            print(
                f"   source : {decision.source}"
            )
            print(
                f"   reason : {decision.reason}"
            )
            print(
                "   scope llm calls: "
                f"{scope_llm_calls}"
            )

            # --------------------------------------------
            # 3. Repository Corpus
            # --------------------------------------------

            corpus = (
                repository_recorder
                .last_corpus
            )

            distribution = (
                status_distribution(
                    corpus
                )
            )

            print()
            print("3. REPOSITORY CORPUS")
            print(
                "   repository received scope: "
                f"{repository_recorder.received_scope}"
            )
            print(
                "   status distribution: "
                f"{distribution}"
            )

            print_corpus(
                corpus
            )

            # --------------------------------------------
            # 4. Final Items
            # --------------------------------------------

            print()
            print(
                "4. FINAL ITEMS "
                "(RERANK + JUDGE)"
            )

            print_items(
                result.items
            )

            # --------------------------------------------
            # 5. memory_context
            # --------------------------------------------

            print()
            print("5. MEMORY CONTEXT")

            if result.memory_context:

                print(
                    result.memory_context
                )

            else:

                print(
                    "   [EMPTY]"
                )

            # --------------------------------------------
            # 6. Injector Contract
            # --------------------------------------------

            print()
            print(
                "6. INJECTOR CONTRACT"
            )
            print(
                "   all items are "
                "MemoryInjectionItem: "
                f"{injector_recorder.all_items_are_injection_items}"
            )
            print(
                "   injected "
                "(content, memory_status):"
            )

            for (
                content,
                memory_status
            ) in (
                injector_recorder
                .received_items
            ):

                print(
                    f"     - [{memory_status}] "
                    f"{content}"
                )

            results.append(
                {
                    "case": case["name"],
                    "query": query,
                    "scope": decision.scope,
                    "source": decision.source,
                    "scope_llm_calls": (
                        scope_llm_calls
                    ),
                    "corpus_distribution": (
                        distribution
                    ),
                    "corpus_ids": [
                        memory.id
                        for memory in corpus
                    ],
                    "selected": [
                        (
                            item.memory.id,
                            item.memory.memory_status,
                            item.memory.content,
                        )
                        for item in result.items
                        if item.judge_selected
                        is True
                    ],
                    "context": (
                        result.memory_context
                    ),
                }
            )

    finally:

        db.close()

        scope_judge_module.call_llm_structured = (
            real_call_llm_structured
        )

    # --------------------------------------------------------
    # DB After
    # --------------------------------------------------------

    after = db_snapshot()

    print_section("DB WRITE CHECK")

    print()
    print(
        "[DB AFTER] "
        f"row_count={after['row_count']} "
        f"max_id={after['max_id']}"
    )
    print(
        "[DB AFTER] "
        f"sha256={after['file_hash'][:16]}..."
    )
    print()

    if (
        before["rows"]
        == after["rows"]
    ):

        print(
            "[PASS] memories 表内容完全一致，"
            "未写入"
        )

    else:

        print(
            "[FAIL] memories 表内容发生变化"
        )

    if (
        before["file_hash"]
        == after["file_hash"]
    ):

        print(
            "[PASS] DB 文件哈希一致，未写入"
        )

    else:

        print(
            "[WARN] DB 文件哈希变化 "
            "（可能只是 SQLite 内部页/时间戳）"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_section("SUMMARY")

    for item in results:

        print()
        print(
            f"- {item['case']}"
        )
        print(
            f"    query      : "
            f"{item['query']}"
        )
        print(
            f"    scope      : "
            f"{item['scope']}"
        )
        print(
            f"    source     : "
            f"{item['source']}"
        )
        print(
            f"    scope llm  : "
            f"{item['scope_llm_calls']}"
        )
        print(
            f"    corpus     : "
            f"{item['corpus_distribution']} "
            f"ids={item['corpus_ids']}"
        )
        print(
            f"    selected   : "
            f"{item['selected']}"
        )

    print()
    print("=" * 78)
    print("MANUAL KEEP REAL RUN FINISHED")
    print("=" * 78)

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
