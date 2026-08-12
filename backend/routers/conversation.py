from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.schemas.conversation import ConversationCreate, ConversationResponse
from backend.services.conversation_service import (
    create_conversation,
    get_conversations,
    get_conversation,
    delete_conversation,
)

router = APIRouter(prefix="/conversations", tags=["Conversation"])


@router.post("", response_model=ConversationResponse)
# 创建新的对话接口
def post_conversation(
    conv_in: ConversationCreate,
    user_id: int = Query(..., description="ID of the user creating the conversation"),
    db: Session = Depends(get_db),
):
    conv = create_conversation(db, user_id, conv_in)
    return conv


@router.get("", response_model=list[ConversationResponse])
# 查询对话列表接口
def list_conversations(
    user_id: Optional[int] = Query(None, description="Optional user_id to filter conversations"),
    db: Session = Depends(get_db),
):
    convs = get_conversations(db, user_id)
    return convs


@router.get("/{id}", response_model=ConversationResponse)
# 获取单条对话接口
def read_conversation(id: int = Path(...), db: Session = Depends(get_db)):
    conv = get_conversation(db, id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.delete("/{id}")
# 删除对话接口
def remove_conversation(id: int = Path(...), db: Session = Depends(get_db)):
    ok = delete_conversation(db, id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}
