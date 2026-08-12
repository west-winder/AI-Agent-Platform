from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from backend.database.database import Base


class Message(Base):

    __tablename__ = "messages"


    id = Column(
        Integer,
        primary_key=True,
        index=True
    )


    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=False
    )


    role = Column(
        String(20),
        nullable=False
    )


    content = Column(
        Text,
        nullable=False
    )


    token_count = Column(
        Integer,
        nullable=True
    )


    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


    conversation = relationship(
        "Conversation",
        back_populates="messages"
    )