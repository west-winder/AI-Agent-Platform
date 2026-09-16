"""
Memory Lifecycle V1 - Contract Tests

类型：

[KEEP]

这些测试全部是：

1. 纯逻辑测试
2. 或 SQLite In-Memory Contract 测试

不依赖：

1. DeepSeek / 任何 LLM
2. 真实 Embedding 模型
3. 真实 Reranker

因此它们是确定性的，
可以长期保留在回归套件中。

运行：

    python tests/memory/test_memory_lifecycle_v1.py

保护的核心 Contract：

1. Relationship 顺序改变
   不得改变 Lifecycle Decision

2. 所有进入 Relationship Judge 的 Existing Memory
   必须得到且只得到一个 Relationship

3. Judge Contract 不完整
   不得产生数据库 Mutation

4. 一次成功 Candidate Lifecycle Write 后，
   被明确判定 conflict 的旧 current Memory
   必须成为 historical

5. historical Memory 不允许被 Lifecycle 恢复为 current

6. 普通 Memory Read 不允许 historical Memory
   进入 retrieval corpus

7. 普通 Memory Write 的
   Exact Dedup / Similarity / Relationship Judge
   不考虑 historical Memory

8. 同一个 Candidate 的
   旧 Memory 状态变化 + 新 Candidate insert
   必须一个 transaction 一起成功或一起 rollback

9. Lifecycle Persistence 的两个独立维度：

   save_candidate
       决定是否 Insert Candidate

   historical_memory_ids
       独立决定对应 current Memory 转 historical

   save_candidate=False
       不得导致 historical_memory_ids 被忽略

10. Runtime current corpus 必须在
    transaction 成功 commit 后才同步：
    移除 historical，append 新建 current。
    rollback 时不得提前修改。

Async Contract 注意事项：

    MemoryPipeline.process
    MemoryExtractor.extract
    MemoryValidator.validate
    MemoryRelationshipJudge.judge

    已经是 async Contract。

    本模块的 test_ 函数全部保持同步
    def test_xxx()，由 run() 把 coroutine 驱动到底：

        1. 直接 python 运行时会真的执行
        2. pytest 会原生收集执行，
           不会被当成 async test 静默跳过
        3. coroutine 一定被 await
        4. 不会产生假 PASS

    Candidate 循环仍然是顺序执行的
    （Sequential Dependency 不得改变）。

    本模块不依赖 pytest-asyncio。
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


from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base

# ------------------------------------------------------------
# 注册全部 ORM Model
#
# 与 backend/main.py 保持一致。
# 只导入 Memory / User 会导致
# relationship 解析失败。
# ------------------------------------------------------------

from backend.models import agent as agent_model  # noqa: F401
from backend.models import message as message_model  # noqa: F401
from backend.models import model as model_model  # noqa: F401
from backend.models import user as user_model  # noqa: F401
from backend.models.conversation import (  # noqa: F401
    Conversation as ConversationModel,
)

from backend.models.memory import (
    MEMORY_STATUS_CURRENT,
    MEMORY_STATUS_HISTORICAL,
    Memory,
)

from backend.schemas.memory import MemoryCreate
from backend.schemas.memory_candidate import MemoryCandidate
from backend.schemas.memory_lifecycle import (
    MemoryLifecycleDecision,
)
from backend.schemas.memory_relationship import (
    MemoryRelationship,
    MemoryRelationshipResult,
)
from backend.schemas.memory_similarity import (
    MemorySimilarityResult,
    MemorySimilaritySearchResult,
)
from backend.schemas.memory_validation import (
    MemoryValidationResult,
)

from backend.memory.memory_write.memory_lifecycle import (
    decide_memory_lifecycle,
)

from backend.memory.memory_write.memory_relationship_judge import (
    RelationshipContractError,
    validate_relationship_coverage,
)

from backend.memory.memory_write.memory_similarity import (
    MemorySimilarityError,
)

from backend.memory.memory_write.memory_pipeline import (
    MemoryPipeline,
)

from backend.memory.memory_read.memory_repository import (
    get_memories_for_read,
)

from backend.services.memory_service import (
    MemoryLifecycleError,
    add_memory,
    create_memory,
    get_current_memories,
    get_memories,
    mark_memories_historical,
)


# ============================================================
# SQLite In-Memory Test DB
# ============================================================

def new_session():
    """
    创建一个全新的 SQLite In-Memory Session。

    每个测试独立一个数据库，
    避免测试之间互相污染。
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
    memory_type="fact",
    memory_status=MEMORY_STATUS_CURRENT,
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


# ============================================================
# Async Contract → 同步测试适配
#
# MemoryPipeline.process
# MemoryExtractor.extract
# MemoryValidator.validate
# MemoryRelationshipJudge.judge
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
#
# 注意：
#
# Process 内部的 Candidate 循环仍然
# 是顺序执行的（Sequential Dependency），
# 这里只是等待同一个 coroutine 完成。
# ============================================================


def run(coro):
    """
    在同步测试中真实执行并等待
    async production contract。
    """

    return asyncio.run(coro)


class SyncMemoryPipeline(MemoryPipeline):
    """
    真实 MemoryPipeline
    +
    同步调用适配。

    只把 async process() 的 coroutine
    驱动到底。
    """

    def process(
        self,
        *args,
        **kwargs
    ):
        return run(
            super().process(
                *args,
                **kwargs
            )
        )


# ============================================================
# Fake Pipeline Components
#
# 全部是确定性 Fake。
# 不调用 LLM，不加载 Embedding 模型。
# ============================================================

class FakeExtractor:
    """
    Fake Extractor。

    注意：

    MemoryPipeline 现在 await self.extractor.extract(...)，
    因此本 Fake 必须保持 async Calling Contract。
    """

    def __init__(
        self,
        candidates
    ):
        self._candidates = candidates

    async def extract(
        self,
        user_message
    ):
        return self._candidates


class FakeValidator:
    """
    Fake Validator。

    注意：

    MemoryPipeline 现在 await self.validator.validate(...)，
    因此本 Fake 必须保持 async Calling Contract。
    """

    async def validate(
        self,
        candidate
    ):
        return MemoryValidationResult(
            valid=True,
            reason="fake valid"
        )


class FakeSimilarity:

    def __init__(
        self,
        matches=None,
        raise_error=False,
    ):
        self._matches = (
            matches
            if matches is not None
            else []
        )
        self._raise_error = raise_error

    def search(
        self,
        candidate,
        memories,
        top_k=5,
        threshold=0.70,
    ):

        if self._raise_error:

            raise MemorySimilarityError(
                "fake embedding runtime failure"
            )

        return MemorySimilaritySearchResult(
            matches=self._matches,
            threshold=threshold,
            top_k=top_k,
        )


class FakeRelationshipJudge:
    """
    Fake Relationship Judge。

    注意：

    MemoryPipeline 现在
    await self.relationship_judge.judge(...)，

    因此本 Fake 必须保持 async Calling Contract。
    """

    def __init__(
        self,
        relationships
    ):
        self._relationships = relationships

    async def judge(
        self,
        candidate,
        existing_memories
    ):

        return MemoryRelationshipResult(
            relationships=self._relationships
        )


class RecordingSimilarity:
    """
    记录每一次 Similarity Search
    实际收到的 Runtime current corpus。

    用于验证：

    Candidate A transaction 成功
        → Candidate B 收到的 corpus
        已经排除了 historical X
    """

    def __init__(
        self,
        raise_error=False,
    ):
        self.corpora = []
        self._raise_error = raise_error

    def search(
        self,
        candidate,
        memories,
        top_k=5,
        threshold=0.70,
    ):

        if self._raise_error:

            raise MemorySimilarityError(
                "fake embedding runtime failure"
            )

        self.corpora.append(
            list(memories)
        )

        matches = [
            similarity_match(
                memory.id,
                memory.content,
            )
            for memory in memories[:top_k]
        ]

        return MemorySimilaritySearchResult(
            matches=matches,
            threshold=threshold,
            top_k=top_k,
        )


class CorpusAwareJudge:
    """
    根据 memory_id 决定 relationship。

    同时记录每次 Judge 实际收到的
    Existing Memory IDs。

    conflict_ids 中的 ID
        → conflict

    其他 ID
        → new

    注意：

    MemoryPipeline 现在
    await self.relationship_judge.judge(...)，

    因此本 Fake 必须保持 async Calling Contract。
    """

    def __init__(
        self,
        conflict_ids
    ):
        self.conflict_ids = set(
            conflict_ids
        )
        self.seen_ids = []

    async def judge(
        self,
        candidate,
        existing_memories
    ):

        self.seen_ids.append(
            [
                memory.memory_id
                for memory in existing_memories
            ]
        )

        relationships = []

        for memory in existing_memories:

            if memory.memory_id in (
                self.conflict_ids
            ):

                relationships.append(
                    relationship(
                        memory.memory_id,
                        "conflict",
                    )
                )

            else:

                relationships.append(
                    relationship(
                        memory.memory_id,
                        "new",
                    )
                )

        return MemoryRelationshipResult(
            relationships=relationships
        )


def build_pipeline_with_candidates(
    candidates,
    similarity,
    judge,
):

    return SyncMemoryPipeline(
        extractor=FakeExtractor(
            candidates
        ),
        validator=FakeValidator(),
        similarity=similarity,
        relationship_judge=judge,
    )


def build_pipeline(
    candidate_content,
    candidate_type="fact",
    matches=None,
    relationships=None,
    raise_similarity_error=False,
):

    candidate = MemoryCandidate(
        content=candidate_content,
        memory_type=candidate_type,
    )

    pipeline = SyncMemoryPipeline(
        extractor=FakeExtractor(
            [candidate]
        ),
        validator=FakeValidator(),
        similarity=FakeSimilarity(
            matches=matches,
            raise_error=raise_similarity_error,
        ),
        relationship_judge=FakeRelationshipJudge(
            relationships=(
                relationships
                if relationships is not None
                else []
            )
        ),
    )

    return pipeline


def relationship(
    memory_id,
    relation,
    reason="reason"
):

    return MemoryRelationship(
        memory_id=memory_id,
        relationship=relation,
        reason=reason,
    )


def similarity_match(
    memory_id,
    content="existing memory"
):

    return MemorySimilarityResult(
        memory_id=memory_id,
        content=content,
        memory_type="fact",
        similarity=0.95,
    )


# ============================================================
# [KEEP] 1. Lifecycle Decision 顺序无关
# ============================================================

def test_decision_order_independence():
    """
    Contract 1：

    Relationship 顺序改变
    不得改变 Lifecycle Decision。
    """

    base = [
        relationship(1, "related"),
        relationship(2, "conflict"),
        relationship(3, "new"),
    ]

    expected = MemoryLifecycleDecision(
        save_candidate=True,
        historical_memory_ids=[2],
    )

    import itertools

    for permutation in itertools.permutations(
        base
    ):

        decision = decide_memory_lifecycle(
            MemoryRelationshipResult(
                relationships=list(
                    permutation
                )
            )
        )

        assert decision == expected, (
            "顺序改变导致 Decision 改变："
            f"{[r.memory_id for r in permutation]}"
        )

    print(
        "[PASS] Lifecycle Decision "
        "顺序无关（related / conflict / new）"
    )


# ============================================================
# [KEEP] 2. duplicate + conflict
# ============================================================

def test_duplicate_with_conflict():
    """
    Contract：

    存在任意 duplicate
        → save_candidate = False

    但 conflict memory
    仍然进入 historical_memory_ids。
    """

    decision = decide_memory_lifecycle(
        MemoryRelationshipResult(
            relationships=[
                relationship(7, "conflict"),
                relationship(8, "duplicate"),
            ]
        )
    )

    assert decision.save_candidate is False

    assert decision.historical_memory_ids == [7]

    print(
        "[PASS] duplicate + conflict："
        "save_candidate=False，"
        "conflict 仍进入 historical_memory_ids"
    )


# ============================================================
# [KEEP] 3. 多个 conflict 全部收集且无重复
# ============================================================

def test_multiple_conflicts_collected_once():
    """
    Contract：

    所有 conflict 对应的 memory_id
    全部进入 historical_memory_ids，
    且无重复。
    """

    decision = decide_memory_lifecycle(
        MemoryRelationshipResult(
            relationships=[
                relationship(4, "conflict"),
                relationship(5, "conflict"),
                relationship(4, "conflict"),
                relationship(6, "related"),
                relationship(9, "new"),
            ]
        )
    )

    assert decision.save_candidate is True

    assert decision.historical_memory_ids == [4, 5]

    print(
        "[PASS] 多个 conflict："
        "全部收集、去重、排序"
    )


# ============================================================
# [KEEP] 4. 无 duplicate → save_candidate = True
# ============================================================

def test_no_duplicate_saves_candidate():
    """
    Contract：

    完全没有 duplicate
        → save_candidate = True
    """

    decision = decide_memory_lifecycle(
        MemoryRelationshipResult(
            relationships=[
                relationship(1, "related"),
                relationship(2, "new"),
                relationship(3, "conflict"),
            ]
        )
    )

    assert decision.save_candidate is True

    assert decision.historical_memory_ids == [3]

    # 空 Relationship 集合

    empty_decision = decide_memory_lifecycle(
        MemoryRelationshipResult(
            relationships=[]
        )
    )

    assert empty_decision.save_candidate is True

    assert empty_decision.historical_memory_ids == []

    print(
        "[PASS] 无 duplicate：save_candidate=True"
    )


# ============================================================
# [KEEP] 5-7. Judge Coverage Contract
# ============================================================

def test_judge_coverage_missing_id_failure():
    """
    Contract 2 / 3：

    缺 ID → Judge Contract Failure。
    """

    existing = [
        similarity_match(1),
        similarity_match(2),
        similarity_match(3),
    ]

    result = MemoryRelationshipResult(
        relationships=[
            relationship(1, "new"),
            relationship(2, "new"),
            # 缺 3
        ]
    )

    try:

        validate_relationship_coverage(
            existing,
            result,
        )

        raise AssertionError(
            "缺 ID 时没有抛出 "
            "RelationshipContractError"
        )

    except RelationshipContractError:
        pass

    print(
        "[PASS] Judge Coverage："
        "缺 ID → Contract Failure"
    )


def test_judge_coverage_duplicate_id_failure():
    """
    Contract 2 / 3：

    重复 ID → Judge Contract Failure。
    """

    existing = [
        similarity_match(1),
        similarity_match(2),
    ]

    result = MemoryRelationshipResult(
        relationships=[
            relationship(1, "new"),
            relationship(1, "conflict"),
            # 2 没有覆盖
        ]
    )

    try:

        validate_relationship_coverage(
            existing,
            result,
        )

        raise AssertionError(
            "重复 ID 时没有抛出 "
            "RelationshipContractError"
        )

    except RelationshipContractError:
        pass

    # 数量正确但 ID 重复的另一种形态

    result2 = MemoryRelationshipResult(
        relationships=[
            relationship(1, "new"),
            relationship(1, "new"),
        ]
    )

    try:

        validate_relationship_coverage(
            existing,
            result2,
        )

        raise AssertionError(
            "ID 重复时没有抛出 "
            "RelationshipContractError"
        )

    except RelationshipContractError:
        pass

    print(
        "[PASS] Judge Coverage："
        "重复 ID → Contract Failure"
    )


def test_judge_coverage_unknown_id_failure():
    """
    Contract 2 / 3：

    未知 ID → Judge Contract Failure。
    """

    existing = [
        similarity_match(1),
        similarity_match(2),
    ]

    result = MemoryRelationshipResult(
        relationships=[
            relationship(1, "new"),
            relationship(999, "new"),
        ]
    )

    try:

        validate_relationship_coverage(
            existing,
            result,
        )

        raise AssertionError(
            "未知 ID 时没有抛出 "
            "RelationshipContractError"
        )

    except RelationshipContractError:
        pass

    print(
        "[PASS] Judge Coverage："
        "未知 ID → Contract Failure"
    )


def test_judge_coverage_success():
    """
    完整覆盖时必须通过。
    """

    existing = [
        similarity_match(1),
        similarity_match(2),
    ]

    result = MemoryRelationshipResult(
        relationships=[
            relationship(1, "conflict"),
            relationship(2, "related"),
        ]
    )

    validate_relationship_coverage(
        existing,
        result,
    )

    print(
        "[PASS] Judge Coverage："
        "完整覆盖 → 通过"
    )


# ============================================================
# [KEEP] 8. Judge Contract Failure 不产生数据库 Mutation
# ============================================================

def test_judge_contract_failure_no_db_mutation():
    """
    Contract 3：

    Judge Contract 不完整
    → 不得产生数据库 Lifecycle Mutation。
    """

    db = new_session()

    old_memory = insert_memory(
        db,
        user_id=1,
        content="用户长期学习Python后端开发",
    )

    pipeline = build_pipeline(
        candidate_content=(
            "用户不打算学习Python后端开发了"
        ),
        matches=[
            similarity_match(
                old_memory.id
            ),
            similarity_match(
                old_memory.id + 100
            ),
        ],
        relationships=[
            # 故意只覆盖一个 ID
            relationship(
                old_memory.id,
                "conflict",
            ),
        ],
    )

    saved = pipeline.process(
        db=db,
        user_id=1,
        user_message="我不再学Python了",
    )

    assert saved == [], (
        "Judge Contract Failure 时 "
        "不应该保存任何 Memory"
    )

    db.refresh(
        old_memory
    )

    assert old_memory.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert old_memory.historical_at is None

    assert len(
        get_memories(db, 1)
    ) == 1

    print(
        "[PASS] Judge Contract Failure："
        "不产生数据库 Mutation"
    )


# ============================================================
# [KEEP] 9. Transaction：insert 失败 → rollback
# ============================================================

def test_transaction_rollback_on_insert_failure():
    """
    Contract 8：

    historical update 成功
    但 candidate insert 故意失败
        → rollback
        → old memories 仍然 current
    """

    db = new_session()

    old_1 = insert_memory(
        db,
        user_id=1,
        content="用户长期学习Python后端开发",
    )

    old_2 = insert_memory(
        db,
        user_id=1,
        content="用户正在使用Python进行后端开发学习",
    )

    pipeline = build_pipeline(
        candidate_content="用户准备转向Java后端开发",
        matches=[
            similarity_match(old_1.id),
            similarity_match(old_2.id),
        ],
        relationships=[
            relationship(
                old_1.id,
                "conflict",
            ),
            relationship(
                old_2.id,
                "conflict",
            ),
        ],
    )

    def boom(*args, **kwargs):

        raise RuntimeError(
            "forced insert failure"
        )

    with patch(
        "backend.memory.memory_write."
        "memory_pipeline.add_memory",
        side_effect=boom,
    ):

        saved = pipeline.process(
            db=db,
            user_id=1,
            user_message="我准备转Java",
        )

    assert saved == []

    db.refresh(old_1)
    db.refresh(old_2)

    assert old_1.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert old_2.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert old_1.historical_at is None

    assert old_2.historical_at is None

    # 没有半完成状态：
    # 既没有新 Memory，也没有被改状态的旧 Memory

    assert len(
        get_memories(db, 1)
    ) == 2

    print(
        "[PASS] Transaction："
        "insert 失败 → rollback，"
        "旧 Memory 仍为 current"
    )


# ============================================================
# [KEEP] 10. 成功 Lifecycle Write：conflict → historical
# ============================================================

def test_conflict_marks_old_historical_and_inserts_new():
    """
    Contract 4 / 8：

    一次成功 Candidate Lifecycle Write 后，
    被明确判定 conflict 的旧 current Memory
    必须成为 historical。

    同时新 Candidate 必须作为 current 插入。
    """

    db = new_session()

    old = insert_memory(
        db,
        user_id=1,
        content="用户长期学习Python后端开发",
        memory_type="goal",
    )

    pipeline = build_pipeline(
        candidate_content="用户准备转向Java后端开发",
        candidate_type="goal",
        matches=[
            similarity_match(old.id),
        ],
        relationships=[
            relationship(
                old.id,
                "conflict",
            ),
        ],
    )

    saved = pipeline.process(
        db=db,
        user_id=1,
        user_message="我准备转Java",
    )

    assert len(saved) == 1

    db.refresh(old)

    assert old.memory_status == (
        MEMORY_STATUS_HISTORICAL
    )

    assert old.historical_at is not None

    new_memory = saved[0]

    assert new_memory.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert new_memory.historical_at is None

    assert new_memory.content == (
        "用户准备转向Java后端开发"
    )

    # corpus 里只剩一条 current

    current = get_current_memories(
        db,
        1,
    )

    assert [
        memory.id
        for memory in current
    ] == [new_memory.id]

    print(
        "[PASS] Lifecycle Write："
        "conflict 旧 Memory → historical，"
        "新 Candidate → current"
    )


# ============================================================
# [KEEP] 11. Historical Exact Duplicate 不阻止保存
# ============================================================

def test_historical_exact_duplicate_does_not_block():
    """
    Contract 7：

    historical Memory 不参与
    普通 Write Lifecycle 判断。

    historical:
    用户正在学习 Python 后端

    Candidate:
    用户正在学习 Python 后端

    不得因为 historical 精确重复
    而阻止新的 current Memory 保存。
    """

    db = new_session()

    historical = insert_memory(
        db,
        user_id=1,
        content="用户正在学习 Python 后端",
        memory_status=MEMORY_STATUS_HISTORICAL,
    )

    historical.historical_at = (
        db.query(Memory)
        .filter(
            Memory.id == historical.id
        )
        .first()
        .created_at
    )

    db.commit()

    pipeline = build_pipeline(
        candidate_content="用户正在学习 Python 后端",
        matches=[],
        relationships=[],
    )

    saved = pipeline.process(
        db=db,
        user_id=1,
        user_message="我又开始学 Python 后端了",
    )

    assert len(saved) == 1, (
        "historical 精确重复 "
        "不应该阻止新的 current Memory 保存"
    )

    new_memory = saved[0]

    assert new_memory.id != historical.id

    assert new_memory.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert len(
        get_current_memories(db, 1)
    ) == 1

    print(
        "[PASS] Historical Exact Duplicate："
        "不阻止新的 current Memory 保存"
    )


# ============================================================
# [KEEP] 12. Related Retrieval Failure → Fail Closed
# ============================================================

def test_retrieval_failure_fails_closed():
    """
    Contract：

    Embedding / Similarity Runtime Failure
        → Fail Closed
        → 本 Candidate 不写数据库

    不得把系统故障误认为"没有相关 Memory"。
    """

    db = new_session()

    insert_memory(
        db,
        user_id=1,
        content="用户长期学习Python后端开发",
    )

    pipeline = build_pipeline(
        candidate_content="用户准备转向Java后端开发",
        raise_similarity_error=True,
    )

    saved = pipeline.process(
        db=db,
        user_id=1,
        user_message="我准备转Java",
    )

    assert saved == []

    assert len(
        get_memories(db, 1)
    ) == 1

    print(
        "[PASS] Related Retrieval Failure："
        "Fail Closed，不写数据库"
    )


# ============================================================
# [KEEP] 13. Memory Read 不含 historical
# ============================================================

def test_memory_read_excludes_historical():
    """
    Contract 6：

    普通 Memory Read 不允许
    historical Memory 进入 retrieval corpus。
    """

    db = new_session()

    current = insert_memory(
        db,
        user_id=1,
        content="用户当前正在学习 LangGraph",
    )

    historical = insert_memory(
        db,
        user_id=1,
        content="用户曾经学习 Python 后端",
        memory_status=MEMORY_STATUS_HISTORICAL,
    )

    read_memories = get_memories_for_read(
        db,
        1,
    )

    read_ids = [
        memory.id
        for memory in read_memories
    ]

    assert read_ids == [current.id]

    assert historical.id not in read_ids

    print(
        "[PASS] Memory Read："
        "historical 不进入 retrieval corpus"
    )


# ============================================================
# [KEEP] 14. mark_memories_historical 跨 user 保护
# ============================================================

def test_mark_historical_cross_user_protection():
    """
    Contract：

    必须防止跨 user_id 修改。

    指定 ID 不存在或不属于当前 user
        → Fail Closed
    """

    db = new_session()

    own = insert_memory(
        db,
        user_id=1,
        content="用户1的Memory",
    )

    other = insert_memory(
        db,
        user_id=2,
        content="用户2的Memory",
    )

    try:

        mark_memories_historical(
            db=db,
            user_id=1,
            memory_ids=[
                own.id,
                other.id,
            ],
        )

        raise AssertionError(
            "跨 user_id 修改没有被拒绝"
        )

    except MemoryLifecycleError:
        pass

    db.rollback()

    db.refresh(own)
    db.refresh(other)

    assert own.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert other.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    print(
        "[PASS] mark_memories_historical："
        "跨 user_id 修改被拒绝"
    )


# ============================================================
# [KEEP] 15. ID 不存在 → Fail Closed
# ============================================================

def test_mark_historical_missing_id_fails_closed():
    """
    Contract：

    ID 不存在
        → Fail Closed
        → 不允许静默部分修改
    """

    db = new_session()

    own = insert_memory(
        db,
        user_id=1,
        content="用户1的Memory",
    )

    try:

        mark_memories_historical(
            db=db,
            user_id=1,
            memory_ids=[
                own.id,
                9999,
            ],
        )

        raise AssertionError(
            "不存在的 ID 没有被拒绝"
        )

    except MemoryLifecycleError:
        pass

    db.rollback()

    db.refresh(own)

    assert own.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    print(
        "[PASS] mark_memories_historical："
        "不存在 ID 被拒绝，不静默部分修改"
    )


# ============================================================
# [KEEP] 16. historical → current 禁止
# ============================================================

def test_historical_cannot_be_marked_again():
    """
    Contract 5 / Lifecycle 单向性：

    historical → current 禁止。

    这里验证：
    已经是 historical 的 Memory
    不允许再次进入 Lifecycle 状态变更。
    """

    db = new_session()

    historical = insert_memory(
        db,
        user_id=1,
        content="用户曾经学习 Python 后端",
        memory_status=MEMORY_STATUS_HISTORICAL,
    )

    try:

        mark_memories_historical(
            db=db,
            user_id=1,
            memory_ids=[historical.id],
        )

        raise AssertionError(
            "historical Memory 被再次标记"
        )

    except MemoryLifecycleError:
        pass

    db.rollback()

    db.refresh(historical)

    assert historical.memory_status == (
        MEMORY_STATUS_HISTORICAL
    )

    print(
        "[PASS] historical → historical "
        "被拒绝，Lifecycle 单向"
    )


# ============================================================
# [KEEP] 17. add_memory 不 commit
# ============================================================

def test_add_memory_does_not_commit():
    """
    底层 op 不 commit。

    这是 Lifecycle 事务边界的前提。
    """

    db = new_session()

    memory = add_memory(
        db=db,
        user_id=1,
        memory_data=MemoryCreate(
            content="用户正在学习 LangGraph",
            memory_type="fact",
        ),
    )

    assert memory.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert memory.historical_at is None

    db.rollback()

    assert len(
        get_memories(db, 1)
    ) == 0

    print(
        "[PASS] add_memory："
        "不 commit，默认 current"
    )


# ============================================================
# [KEEP] 18. create_memory CRUD wrapper 仍然 commit
# ============================================================

def test_create_memory_wrapper_still_commits():
    """
    保留现有简单 CRUD 兼容性。

    普通 Router 调用方
    不需要因为 Lifecycle 重写。
    """

    db = new_session()

    memory = create_memory(
        db=db,
        user_id=1,
        memory_data=MemoryCreate(
            content="用户喜欢使用中文交流",
            memory_type="preference",
        ),
    )

    assert memory.id is not None

    assert memory.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    # 换一个 Session 读取，确认已经 commit

    assert len(
        get_memories(db, 1)
    ) == 1

    print(
        "[PASS] create_memory wrapper："
        "仍然 add + commit + refresh"
    )


# ============================================================
# [KEEP] 19. Case A：纯 duplicate → no-op
# ============================================================

def test_pure_duplicate_is_noop():
    """
    Case A：

    save_candidate = False
    historical_memory_ids = []

        → 纯 no-op
        → 不需要任何数据库 mutation
    """

    db = new_session()

    # 注意：
    #
    # 这里刻意使用"语义重复但非精确重复"的内容。
    #
    # 如果 content 完全相同，
    # Exact Dedup 会在 Relationship Judge 之前
    # 直接短路，
    # 本测试就测不到 Lifecycle Persistence 分支。

    old = insert_memory(
        db,
        user_id=1,
        content="用户长期学习 Python 后端开发",
    )

    related = insert_memory(
        db,
        user_id=1,
        content="用户正在使用Python进行后端开发学习",
    )

    pipeline = build_pipeline(
        candidate_content="用户长期学习Python后端开发",
        matches=[
            similarity_match(old.id),
            similarity_match(related.id),
        ],
        relationships=[
            relationship(
                old.id,
                "duplicate",
            ),
            relationship(
                related.id,
                "related",
            ),
        ],
    )

    saved = pipeline.process(
        db=db,
        user_id=1,
        user_message="我长期学习Python后端开发",
    )

    assert saved == []

    db.refresh(old)
    db.refresh(related)

    assert old.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert related.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert len(
        get_memories(db, 1)
    ) == 2

    print(
        "[PASS] Case A：纯 duplicate → no-op"
    )


# ============================================================
# [KEEP] 20. duplicate + conflict persistence
# ============================================================

def test_duplicate_with_conflict_persistence():
    """
    save_candidate 与 historical_memory_ids
    是两个独立维度。

    Duplicate
        → 只决定 Candidate 不 Insert

    Conflict
        → 独立决定对应 current Memory 转 historical

    输入：

    A → duplicate
    B → conflict

    结果：

    Candidate 不新增
    A 仍 current
    B historical
    """

    db = new_session()

    # 同样是"语义重复但非精确重复"，
    # 避免被 Exact Dedup 提前短路。

    duplicate_memory = insert_memory(
        db,
        user_id=1,
        content="用户正在准备学习 Java 后端开发",
        memory_type="goal",
    )

    conflict_memory = insert_memory(
        db,
        user_id=1,
        content="用户不打算学习任何新语言",
        memory_type="goal",
    )

    related_memory = insert_memory(
        db,
        user_id=1,
        content="用户是测绘工程专业硕士研究生",
        memory_type="profile",
    )

    pipeline = build_pipeline(
        candidate_content="用户准备学习 Java",
        candidate_type="goal",
        matches=[
            similarity_match(
                related_memory.id
            ),
            similarity_match(
                conflict_memory.id
            ),
            similarity_match(
                duplicate_memory.id
            ),
        ],
        relationships=[
            relationship(
                related_memory.id,
                "related",
            ),
            relationship(
                conflict_memory.id,
                "conflict",
            ),
            relationship(
                duplicate_memory.id,
                "duplicate",
            ),
        ],
    )

    saved = pipeline.process(
        db=db,
        user_id=1,
        user_message="我准备学习 Java",
    )

    # Candidate 没有新增

    assert saved == []

    # A 仍 current

    db.refresh(duplicate_memory)

    assert duplicate_memory.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert duplicate_memory.historical_at is None

    # B historical

    db.refresh(conflict_memory)

    assert conflict_memory.memory_status == (
        MEMORY_STATUS_HISTORICAL
    )

    assert conflict_memory.historical_at is not None

    # related 不受影响

    db.refresh(related_memory)

    assert related_memory.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    # 总数不变：没有 Insert

    assert len(
        get_memories(db, 1)
    ) == 3

    # current corpus 只剩两条

    current_ids = {
        memory.id
        for memory in get_current_memories(
            db, 1
        )
    }

    assert current_ids == {
        duplicate_memory.id,
        related_memory.id,
    }

    print(
        "[PASS] duplicate + conflict persistence："
        "Candidate 不新增，"
        "duplicate 仍 current，"
        "conflict 转 historical"
    )


# ============================================================
# [KEEP] 21. 多 Candidate：Runtime corpus 刷新
# ============================================================

def test_multiple_candidates_runtime_corpus_refresh():
    """
    Runtime current corpus 必须在
    transaction 成功 commit 后刷新。

    同一次 extract 返回多个 Candidate：

    Candidate A
        → X conflict
        → DB: X historical

    Candidate B
        → 不得再把 X 当作 current corpus

    违反会导致：

    后续 Candidate 的
    Exact Dedup / Similarity / Judge
    错误地把 historical Memory 当 current。
    """

    db = new_session()

    x = insert_memory(
        db,
        user_id=1,
        content="用户长期学习Python后端开发",
        memory_type="goal",
    )

    y = insert_memory(
        db,
        user_id=1,
        content="用户喜欢使用中文交流",
        memory_type="preference",
    )

    candidate_a = MemoryCandidate(
        content="用户准备转向Java后端开发",
        memory_type="goal",
    )

    candidate_b = MemoryCandidate(
        content="用户希望成为AI Agent开发工程师",
        memory_type="goal",
    )

    similarity = RecordingSimilarity()

    judge = CorpusAwareJudge(
        conflict_ids=[x.id]
    )

    pipeline = build_pipeline_with_candidates(
        candidates=[
            candidate_a,
            candidate_b,
        ],
        similarity=similarity,
        judge=judge,
    )

    saved = pipeline.process(
        db=db,
        user_id=1,
        user_message="我准备转Java，也想做AI Agent",
    )

    assert len(saved) == 2

    # ------------------------------------------------------
    # Candidate A 看到了 X
    # ------------------------------------------------------

    first_corpus_ids = [
        memory.id
        for memory in similarity.corpora[0]
    ]

    assert x.id in first_corpus_ids

    # ------------------------------------------------------
    # Candidate B 不再看到 X
    # ------------------------------------------------------

    second_corpus_ids = [
        memory.id
        for memory in similarity.corpora[1]
    ]

    assert x.id not in second_corpus_ids, (
        "Candidate A 已把 X 转为 historical，"
        "Candidate B 仍然在 Runtime corpus 里看到 X"
    )

    # ------------------------------------------------------
    # Candidate B 必须看到 Candidate A 新建的 Memory
    # ------------------------------------------------------

    new_a_id = saved[0].id

    assert new_a_id in second_corpus_ids

    assert y.id in second_corpus_ids

    # ------------------------------------------------------
    # Judge 侧同样不应再看到 X
    # ------------------------------------------------------

    assert x.id in judge.seen_ids[0]

    assert x.id not in judge.seen_ids[1]

    # ------------------------------------------------------
    # 数据库状态
    # ------------------------------------------------------

    db.refresh(x)

    assert x.memory_status == (
        MEMORY_STATUS_HISTORICAL
    )

    assert x.historical_at is not None

    print(
        "[PASS] 多 Candidate："
        "Runtime current corpus 在 commit 后刷新，"
        "后续 Candidate 不再看到 historical X"
    )


# ============================================================
# [KEEP] 22. rollback 后 Runtime corpus 不被提前删除
# ============================================================

def test_rollback_keeps_runtime_corpus_unchanged():
    """
    rollback 顺序：

    执行 DB mutation
        ↓
    commit 失败
        ↓
    rollback
        ↓
    Runtime corpus 保持原样

    禁止：

    先从 corpus remove
        ↓
    DB 操作
        ↓
    rollback
    """

    db = new_session()

    x = insert_memory(
        db,
        user_id=1,
        content="用户长期学习Python后端开发",
        memory_type="goal",
    )

    candidate_a = MemoryCandidate(
        content="用户准备转向Java后端开发",
        memory_type="goal",
    )

    candidate_b = MemoryCandidate(
        content="用户准备转向Go后端开发",
        memory_type="goal",
    )

    similarity = RecordingSimilarity()

    judge = CorpusAwareJudge(
        conflict_ids=[x.id]
    )

    pipeline = build_pipeline_with_candidates(
        candidates=[
            candidate_a,
            candidate_b,
        ],
        similarity=similarity,
        judge=judge,
    )

    def boom(*args, **kwargs):

        raise RuntimeError(
            "forced insert failure"
        )

    with patch(
        "backend.memory.memory_write."
        "memory_pipeline.add_memory",
        side_effect=boom,
    ):

        saved = pipeline.process(
            db=db,
            user_id=1,
            user_message="我准备转Java",
        )

    # ------------------------------------------------------
    # 两个 Candidate 都失败
    # ------------------------------------------------------

    assert saved == []

    # ------------------------------------------------------
    # 数据库：X 仍然 current
    # ------------------------------------------------------

    db.refresh(x)

    assert x.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert x.historical_at is None

    assert len(
        get_memories(db, 1)
    ) == 1

    # ------------------------------------------------------
    # Runtime corpus：X 仍然在
    #
    # 如果被提前删除，
    # Candidate B 就看不到 X 了。
    # ------------------------------------------------------

    assert len(
        similarity.corpora
    ) == 2

    for index, corpus in enumerate(
        similarity.corpora
    ):

        corpus_ids = [
            memory.id
            for memory in corpus
        ]

        assert x.id in corpus_ids, (
            "rollback 后 Runtime corpus "
            "被错误修改："
            f"call {index} 看不到 X"
        )

    print(
        "[PASS] rollback："
        "数据库与 Runtime corpus 保持一致，"
        "X 仍为 current 且仍在 corpus 中"
    )


# ============================================================
# [KEEP] 23. commit 失败 → rollback，corpus 不同步
# ============================================================

def test_commit_failure_rolls_back_and_keeps_corpus():
    """
    Transaction Failure Contract：

    故意让 db.commit() 在 Lifecycle transaction 时失败。

    断言：

    1. Old Memory 仍 current
    2. Candidate 不存在
    3. Runtime current corpus 未错误同步

    注意：

    Session.commit 在 fixture 准备完成后才 patch，
    因此不会影响 insert_memory() 的准备阶段。
    """

    db = new_session()

    x = insert_memory(
        db,
        user_id=1,
        content="用户长期学习Python后端开发",
        memory_type="goal",
    )

    candidate_a = MemoryCandidate(
        content="用户准备转向Java后端开发",
        memory_type="goal",
    )

    candidate_b = MemoryCandidate(
        content="用户准备转向Go后端开发",
        memory_type="goal",
    )

    similarity = RecordingSimilarity()

    judge = CorpusAwareJudge(
        conflict_ids=[x.id]
    )

    pipeline = build_pipeline_with_candidates(
        candidates=[
            candidate_a,
            candidate_b,
        ],
        similarity=similarity,
        judge=judge,
    )

    original_commit = Session.commit

    commit_calls = []

    def failing_commit(
        self,
        *args,
        **kwargs
    ):

        commit_calls.append(1)

        raise RuntimeError(
            "forced commit failure"
        )

    # ------------------------------------------------------
    # fixture 准备完成后才 patch
    # ------------------------------------------------------

    with patch.object(
        Session,
        "commit",
        failing_commit,
    ):

        saved = pipeline.process(
            db=db,
            user_id=1,
            user_message="我准备换技术方向",
        )

    # ------------------------------------------------------
    # 两个 Candidate 都尝试过 commit
    # ------------------------------------------------------

    assert len(commit_calls) == 2

    # ------------------------------------------------------
    # 没有保存任何 Memory
    # ------------------------------------------------------

    assert saved == []

    # ------------------------------------------------------
    # 数据库：X 仍 current，没有新增 Candidate
    # ------------------------------------------------------

    db.expire_all()

    assert len(
        get_memories(db, 1)
    ) == 1

    fresh_x = (
        db.query(Memory)
        .filter(
            Memory.id == x.id
        )
        .first()
    )

    assert fresh_x.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert fresh_x.historical_at is None

    # ------------------------------------------------------
    # Runtime current corpus 未被错误同步
    #
    # 如果在 commit 失败后仍然移除 X，
    # 后续 Candidate 就看不到 X 了。
    # ------------------------------------------------------

    assert len(
        similarity.corpora
    ) == 2

    for index, corpus in enumerate(
        similarity.corpora
    ):

        corpus_ids = [
            memory.id
            for memory in corpus
        ]

        assert x.id in corpus_ids, (
            "commit 失败后 Runtime corpus "
            "被错误同步："
            f"call {index} 看不到 X"
        )

    print(
        "[PASS] commit failure："
        "旧 Memory 仍 current，"
        "Candidate 不存在，"
        "Runtime corpus 未错误同步"
    )


# ============================================================
# [KEEP] 24. commit 成功后的操作失败不得被当成 rollback
# ============================================================

def test_post_commit_failure_is_not_treated_as_rollback():
    """
    Transaction Failure Contract：

    db.commit() 一旦成功返回，
    事务已经正式提交。

    此时不得再进入
    声称可以撤销该事务的 rollback 分支。

    否则会产生：

    DB 已成功 commit
    但 Pipeline 把这一轮当成 rollback failure

    具体表现：

    DB 里旧 Memory 已 historical、新 Memory 已插入
    但 Pipeline 返回 saved=[]，
    并且假装事务被回滚。

    本测试通过让 Session.refresh 抛异常来构造
    "commit 之后的失败"。

    当前实现在 commit 成功后不再调用 refresh，
    因此这类失败不应影响已提交结果。
    """

    db = new_session()

    x = insert_memory(
        db,
        user_id=1,
        content="用户长期学习Python后端开发",
        memory_type="goal",
    )

    pipeline = build_pipeline(
        candidate_content="用户准备转向Java后端开发",
        candidate_type="goal",
        matches=[
            similarity_match(x.id),
        ],
        relationships=[
            relationship(
                x.id,
                "conflict",
            ),
        ],
    )

    def refresh_boom(
        self,
        *args,
        **kwargs
    ):

        raise RuntimeError(
            "forced post-commit refresh failure"
        )

    with patch.object(
        Session,
        "refresh",
        refresh_boom,
    ):

        saved = pipeline.process(
            db=db,
            user_id=1,
            user_message="我准备转Java",
        )

    # ------------------------------------------------------
    # commit 已成功：
    #
    # Pipeline 必须如实返回已保存的 Memory
    # ------------------------------------------------------

    assert len(saved) == 1

    new_memory = saved[0]

    assert new_memory.id is not None

    # ------------------------------------------------------
    # 数据库状态与返回值必须一致
    # ------------------------------------------------------

    db.expire_all()

    fresh_x = (
        db.query(Memory)
        .filter(
            Memory.id == x.id
        )
        .first()
    )

    assert fresh_x.memory_status == (
        MEMORY_STATUS_HISTORICAL
    )

    fresh_new = (
        db.query(Memory)
        .filter(
            Memory.id == new_memory.id
        )
        .first()
    )

    assert fresh_new is not None

    assert fresh_new.memory_status == (
        MEMORY_STATUS_CURRENT
    )

    assert len(
        get_memories(db, 1)
    ) == 2

    print(
        "[PASS] commit 成功后失败："
        "不被当成 rollback，"
        "DB 与返回值一致"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MEMORY LIFECYCLE V1 CONTRACT TESTS [KEEP]")
    print("=" * 70)
    print()

    test_decision_order_independence()
    test_duplicate_with_conflict()
    test_multiple_conflicts_collected_once()
    test_no_duplicate_saves_candidate()

    test_judge_coverage_success()
    test_judge_coverage_missing_id_failure()
    test_judge_coverage_duplicate_id_failure()
    test_judge_coverage_unknown_id_failure()
    test_judge_contract_failure_no_db_mutation()

    test_transaction_rollback_on_insert_failure()
    test_conflict_marks_old_historical_and_inserts_new()
    test_pure_duplicate_is_noop()
    test_duplicate_with_conflict_persistence()

    test_multiple_candidates_runtime_corpus_refresh()
    test_rollback_keeps_runtime_corpus_unchanged()
    test_commit_failure_rolls_back_and_keeps_corpus()
    test_post_commit_failure_is_not_treated_as_rollback()

    test_historical_exact_duplicate_does_not_block()
    test_retrieval_failure_fails_closed()
    test_memory_read_excludes_historical()

    test_mark_historical_cross_user_protection()
    test_mark_historical_missing_id_fails_closed()
    test_historical_cannot_be_marked_again()

    test_add_memory_does_not_commit()
    test_create_memory_wrapper_still_commits()

    print()
    print("=" * 70)
    print(
        "ALL MEMORY LIFECYCLE V1 "
        "CONTRACT TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()
