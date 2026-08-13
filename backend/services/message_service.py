from sqlalchemy.orm import Session

from backend.models.message import Message
from backend.schemas.message import MessageCreate


def create_message(
    db: Session,
    message: MessageCreate,
    conversation_id: int,
    role: str
):

    db_message = Message(
        conversation_id=conversation_id,
        role=role,
        content=message.content
    )


    db.add(db_message)

    db.commit()

    db.refresh(db_message)

    return db_message



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