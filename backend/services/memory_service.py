from sqlalchemy.orm import Session

from backend.models.memory import Memory
from backend.schemas.memory import MemoryCreate, MemoryUpdate


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
    """

    memory = Memory(
        user_id=user_id,
        content=memory_data.content,
        memory_type=memory_data.memory_type
    )

    db.add(memory)
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