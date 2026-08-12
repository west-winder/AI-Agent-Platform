from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.database import get_db

from backend.schemas.message import MessageResponse

from backend.services.message_service import (
    get_messages_by_conversation
)


router = APIRouter(
    prefix="/messages",
    tags=["messages"]
)


# 查询指定会话下的消息列表
def get_messages(
    conversation_id: int,
    db: Session = Depends(get_db)
):

    return get_messages_by_conversation(
        db,
        conversation_id
    )