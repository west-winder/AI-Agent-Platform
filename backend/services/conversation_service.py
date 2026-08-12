from typing import List, Optional
from sqlalchemy.orm import Session

from backend.models.conversation import Conversation
from backend.schemas.conversation import ConversationCreate


# 创建新的对话记录
def create_conversation(db: Session, user_id: int, conv_in: ConversationCreate) -> Conversation:
    conv = Conversation(
        user_id=user_id,
        agent_id=conv_in.agent_id,
        title=conv_in.title
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


# 根据用户ID获取对话列表
def get_conversations(db: Session, user_id: Optional[int] = None) -> List[Conversation]:
    q = db.query(Conversation)
    if user_id is not None:
        q = q.filter(Conversation.user_id == user_id)
    return q.all()


# 根据对话ID获取单条对话
def get_conversation(db: Session, id: int) -> Optional[Conversation]:
    return db.query(Conversation).filter(Conversation.id == id).first()


# 删除指定对话
def delete_conversation(db: Session, id: int) -> bool:
    conv = db.query(Conversation).filter(Conversation.id == id).first()
    if not conv:
        return False
    db.delete(conv)
    db.commit()
    return True
