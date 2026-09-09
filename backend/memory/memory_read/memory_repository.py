from sqlalchemy.orm import Session

from backend.models.memory import (
    MEMORY_STATUS_CURRENT,
    Memory,
)


def get_memories_for_read(
    db: Session,
    user_id: int
):
    """
    获取指定用户的 Memory，供 Memory Read 使用。

    当前职责：

    SQLite
        ↓
    查询指定用户的 Memory
        ↓
    返回 Memory ORM 对象列表

    Memory Lifecycle V1：

    普通 Memory Read 的 retrieval corpus
    只包括：

    memory_status == current

    historical Memory 不允许
    进入普通 retrieval corpus。

    本次不在上层做过滤，
    也不修改：

    Dense
    BM25
    RRF
    Reranker
    Judge
    Injector

    只在数据源层过滤。

    本模块不负责：

    1. Embedding
    2. Similarity Search
    3. Ranking
    4. Top-N
    5. Reranking
    6. LLM Judge
    7. Memory Injection
    8. Historical Query Detection（本次不实现）
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