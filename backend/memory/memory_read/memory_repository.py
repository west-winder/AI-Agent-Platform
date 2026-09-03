from sqlalchemy.orm import Session

from backend.models.memory import Memory


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

    本模块不负责：

    1. Embedding
    2. Similarity Search
    3. Ranking
    4. Top-N
    5. Reranking
    6. LLM Judge
    7. Memory Injection
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