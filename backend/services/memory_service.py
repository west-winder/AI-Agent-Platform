from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.models.memory import (
    MEMORY_STATUS_CURRENT,
    MEMORY_STATUS_HISTORICAL,
    Memory,
)
from backend.schemas.memory import MemoryCreate, MemoryUpdate


# ==================================================
# Lifecycle Persistence Failure
# ==================================================

class MemoryLifecycleError(Exception):
    """
    Lifecycle 持久化操作失败。

    出现这个异常意味着：

    本次 Lifecycle 持久化操作
    没有满足前置条件，
    因此拒绝执行任何修改。

    Fail Closed：

    不允许静默部分修改。
    """


# ==================================================
# 底层 Lifecycle / Persistence Operation
#
# 这一层不 commit。
#
# 目的：
#
# 让 Lifecycle Pipeline 可以在
# 一个事务内组合多个底层操作，
# 由 Pipeline 统一 commit / rollback。
# ==================================================

def add_memory(
    db: Session,
    user_id: int,
    memory_data: MemoryCreate
) -> Memory:
    """
    构造一条 Memory 并加入当前 Session。

    不 commit。

    默认：

    memory_status = current
    historical_at = None

    注意：

    Lifecycle V1 只允许新增 current Memory。

    如果旧状态重新出现，
    正确做法是：

    新建一条 current Memory，
    而不是恢复 historical Memory。
    """

    memory = Memory(
        user_id=user_id,
        content=memory_data.content,
        memory_type=memory_data.memory_type,
        memory_status=MEMORY_STATUS_CURRENT,
        historical_at=None,
    )

    db.add(memory)

    return memory


def mark_memories_historical(
    db: Session,
    user_id: int,
    memory_ids,
    historical_at: datetime | None = None,
) -> list[Memory]:
    """
    将指定 Memory 标记为 historical。

    不 commit。

    前置条件（Fail Closed）：

    1. memory_ids 不能包含重复 ID
    2. 所有 memory_id 必须存在
    3. 所有 Memory 必须属于指定 user_id
    4. 所有 Memory 当前必须是 current

    任何一条不满足：

    raise MemoryLifecycleError
    不修改任何数据。

    注意：

    Lifecycle V1 禁止
    historical → current。

    因此本函数是单向操作，
    不提供恢复接口。
    """

    requested_ids = list(
        memory_ids or []
    )

    # --------------------------------------------------
    # 空集合：不产生任何操作
    # --------------------------------------------------

    if not requested_ids:
        return []

    # --------------------------------------------------
    # ID 类型与重复检查
    # --------------------------------------------------

    for memory_id in requested_ids:

        if not isinstance(
            memory_id,
            int
        ):

            raise MemoryLifecycleError(
                "memory_id 必须是 int："
                f"{memory_id!r}"
            )

    if len(
        set(requested_ids)
    ) != len(
        requested_ids
    ):

        raise MemoryLifecycleError(
            "memory_ids 包含重复 ID："
            f"{requested_ids}"
        )

    # --------------------------------------------------
    # 只查询指定 user_id 的 Memory
    #
    # 防止跨 user_id 修改。
    # --------------------------------------------------

    memories = (
        db.query(Memory)
        .filter(
            Memory.id.in_(requested_ids),
            Memory.user_id == user_id,
        )
        .all()
    )

    # --------------------------------------------------
    # 数量一致性检查
    #
    # 覆盖：
    #
    # 1. ID 不存在
    # 2. ID 存在但属于其他 user_id
    # --------------------------------------------------

    if len(memories) != len(requested_ids):

        found_ids = {
            memory.id
            for memory in memories
        }

        missing_ids = [
            memory_id
            for memory_id in requested_ids
            if memory_id not in found_ids
        ]

        raise MemoryLifecycleError(
            "部分 Memory 不存在 "
            "或不属于当前 user_id："
            f"{missing_ids}"
        )

    # --------------------------------------------------
    # 状态检查
    #
    # 只允许 current → historical
    # --------------------------------------------------

    for memory in memories:

        if memory.memory_status != (
            MEMORY_STATUS_CURRENT
        ):

            raise MemoryLifecycleError(
                "只允许将 current Memory "
                "标记为 historical："
                f"id={memory.id} "
                f"status={memory.memory_status}"
            )

    # --------------------------------------------------
    # 所有检查通过后才修改
    #
    # 避免"部分修改后才发现失败"。
    # --------------------------------------------------

    if historical_at is None:

        historical_at = datetime.now(
            timezone.utc
        )

    for memory in memories:

        memory.memory_status = (
            MEMORY_STATUS_HISTORICAL
        )

        memory.historical_at = historical_at

    return memories


# ==================================================
# 创建Memory
# ==================================================

def create_memory(
    db: Session,
    user_id: int,
    memory_data: MemoryCreate
):
    """
    创建一条Memory。

    这是一个简单 CRUD wrapper：

    add
        ↓
    commit
        ↓
    refresh

    普通 Router 调用方继续使用它，
    不需要因为 Lifecycle 重写。

    注意：

    Lifecycle Pipeline 不应该使用本函数，
    因为它会自己 commit，
    破坏 Lifecycle 的事务边界。

    Lifecycle Pipeline 应使用：

    add_memory()
    """

    memory = add_memory(
        db=db,
        user_id=user_id,
        memory_data=memory_data,
    )

    db.commit()
    db.refresh(memory)

    return memory


# ==================================================
# 查询用户的Memory
# ==================================================

def get_memories(
    db: Session,
    user_id: int
):
    """
    获取指定用户的全部Memory。

    注意：

    这里不过滤 memory_status。

    普通 CRUD / Debug 场景
    仍然需要看到全部 Memory，
    包括 historical。

    Memory Write Lifecycle 应使用：

    get_current_memories()
    """

    return (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id
        )
        .order_by(
            Memory.created_at.desc()
        )
        .all()
    )


# ==================================================
# 查询用户的 current Memory
# ==================================================

def get_current_memories(
    db: Session,
    user_id: int
):
    """
    获取指定用户当前仍然有效的 Memory。

    只返回：

    memory_status == current

    Memory Write Lifecycle 的
    Write Corpus 只使用 current Memory：

    1. Exact Dedup
    2. Related Retrieval
    3. Relationship Judge

    historical Memory 不参与
    普通 Write Lifecycle 判断。
    """

    return (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id,
            Memory.memory_status
            == MEMORY_STATUS_CURRENT,
        )
        .order_by(
            Memory.created_at.desc()
        )
        .all()
    )


# ==================================================
# 查询单条Memory
# ==================================================

def get_memory_by_id(
    db: Session,
    memory_id: int,
    user_id: int
):
    """
    获取指定用户的一条Memory。

    同时限制user_id，
    防止用户访问其他用户的Memory。
    """

    return (
        db.query(Memory)
        .filter(
            Memory.id == memory_id,
            Memory.user_id == user_id
        )
        .first()
    )


# ==================================================
# 更新Memory
# ==================================================

def update_memory(
    db: Session,
    memory: Memory,
    memory_data: MemoryUpdate
):
    """
    更新Memory。

    注意：

    MemoryUpdate 不包含
    memory_status / historical_at，
    因此普通 Update
    无法修改 Lifecycle 状态。
    """

    update_data = memory_data.model_dump(
        exclude_unset=True
    )

    for field, value in update_data.items():
        setattr(memory, field, value)

    db.commit()
    db.refresh(memory)

    return memory


# ==================================================
# 删除Memory
# ==================================================

def delete_memory(
    db: Session,
    memory: Memory
):
    """
    删除Memory。
    """

    db.delete(memory)
    db.commit()
