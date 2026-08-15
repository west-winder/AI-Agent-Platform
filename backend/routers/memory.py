from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.schemas.memory import (
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse
)
from backend.services.memory_service import (
    create_memory,
    get_memories,
    get_memory_by_id,
    update_memory,
    delete_memory
)


router = APIRouter(
    prefix="/memories",
    tags=["Memory"]
)


# ==================================================
# 当前用户
# ==================================================

USER_ID = 1


# ==================================================
# 创建Memory
# ==================================================

@router.post(
    "",
    response_model=MemoryResponse
)
def create(
    memory_data: MemoryCreate,
    db: Session = Depends(get_db)
):
    return create_memory(
        db,
        USER_ID,
        memory_data
    )


# ==================================================
# 获取当前用户所有Memory
# ==================================================

@router.get(
    "",
    response_model=list[MemoryResponse]
)
def get_all(
    db: Session = Depends(get_db)
):
    return get_memories(
        db,
        USER_ID
    )


# ==================================================
# 获取单条Memory
# ==================================================

@router.get(
    "/{memory_id}",
    response_model=MemoryResponse
)
def get_one(
    memory_id: int,
    db: Session = Depends(get_db)
):

    memory = get_memory_by_id(
        db,
        memory_id,
        USER_ID
    )

    if not memory:
        raise HTTPException(
            status_code=404,
            detail="Memory not found"
        )

    return memory


# ==================================================
# 更新Memory
# ==================================================

@router.patch(
    "/{memory_id}",
    response_model=MemoryResponse
)
def update(
    memory_id: int,
    memory_data: MemoryUpdate,
    db: Session = Depends(get_db)
):

    memory = get_memory_by_id(
        db,
        memory_id,
        USER_ID
    )

    if not memory:
        raise HTTPException(
            status_code=404,
            detail="Memory not found"
        )

    return update_memory(
        db,
        memory,
        memory_data
    )


# ==================================================
# 删除Memory
# ==================================================

@router.delete(
    "/{memory_id}"
)
def delete(
    memory_id: int,
    db: Session = Depends(get_db)
):

    memory = get_memory_by_id(
        db,
        memory_id,
        USER_ID
    )

    if not memory:
        raise HTTPException(
            status_code=404,
            detail="Memory not found"
        )

    delete_memory(
        db,
        memory
    )

    return {
        "message": "Memory deleted successfully"
    }