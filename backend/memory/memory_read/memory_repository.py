from sqlalchemy.orm import Session

from backend.models.memory import (
    MEMORY_STATUS_CURRENT,
    MEMORY_STATUS_HISTORICAL,
    Memory,
)

from backend.memory.memory_read.memory_query_scope import (
    ALLOWED_MEMORY_QUERY_SCOPES,
    MEMORY_QUERY_SCOPE_BOTH,
    MEMORY_QUERY_SCOPE_CURRENT,
    MEMORY_QUERY_SCOPE_HISTORICAL,
)


def get_memories_for_read(
    db: Session,
    user_id: int,
    scope: str = MEMORY_QUERY_SCOPE_CURRENT,
):
    """
    获取指定用户的 Memory，供 Memory Read 使用。

    当前职责：

    user_id
    +
    retrieval scope
        ↓
    SQLite
        ↓
    根据用户和 Memory Status
    查询 Retrieval Corpus
        ↓
    返回 Memory ORM 对象列表

    支持的 scope：

    current
        只返回 current Memory。

    historical
        只返回 historical Memory。

    both
        返回 current + historical Memory。

    默认：

    scope = current

    因此旧调用：

        get_memories_for_read(
            db,
            user_id
        )

    仍然保持 current-only 行为。

    Memory Lifecycle V1：

    本次仍然只在数据源层过滤，
    不在上层做过滤。

    本模块只负责：

    根据已经确定好的 scope，
    从数据库中取得对应 Memory。

    本模块不负责：

    1. Query 语义理解
    2. Historical Query Detection
    3. LLM Scope Judge
    4. Embedding
    5. Similarity Search
    6. BM25
    7. RRF
    8. Ranking
    9. Reranking
    10. Relevance Judge
    11. Memory Injection
    """

    if scope not in ALLOWED_MEMORY_QUERY_SCOPES:
        raise ValueError(
            f"不支持的 Memory Query Scope：{scope}"
        )

    # 无论哪种 scope，
    # 都只能读取指定 user_id 的 Memory。
    query = (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id
        )
    )

    # current：
    # 只允许当前仍然有效的 Memory
    # 进入 Retrieval Corpus。
    if scope == MEMORY_QUERY_SCOPE_CURRENT:
        query = query.filter(
            Memory.memory_status
            == MEMORY_STATUS_CURRENT
        )

    # historical：
    # 只允许历史 Memory
    # 进入 Retrieval Corpus。
    elif scope == MEMORY_QUERY_SCOPE_HISTORICAL:
        query = query.filter(
            Memory.memory_status
            == MEMORY_STATUS_HISTORICAL
        )

    # both：
    # 明确列出两种状态，
    # 而不是简单地取消 status filter。
    elif scope == MEMORY_QUERY_SCOPE_BOTH:
        query = query.filter(
            Memory.memory_status.in_(
                [
                    MEMORY_STATUS_CURRENT,
                    MEMORY_STATUS_HISTORICAL,
                ]
            )
        )

    return (
        query
        .order_by(
            Memory.created_at.desc()
        )
        .all()
    )
