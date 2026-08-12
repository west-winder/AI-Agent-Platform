from sqlalchemy.orm import Session

from backend.models.message import Message
from backend.schemas.message import MessageCreate


# 创建一条消息并写入数据库
def create_message(
    db: Session,
    message: MessageCreate,
    conversation_id: int
):

    db_message = Message(
        conversation_id=conversation_id,
        role=message.role,
        content=message.content
    )


    db.add(db_message)

    db.commit()

    db.refresh(db_message)


    return db_message


# 按会话ID查询消息列表
def get_messages_by_conversation(
    db: Session,
    conversation_id: int
):

    return (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(
            Message.created_at
        )
        .all()
    )