from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database.database import Base


from sqlalchemy import Index


class Conversation(Base):

    __tablename__ = "conversations"

    __table_args__ = (
        Index(
            "idx_conversation_user_time",
            "user_id",
            "last_message_time"
        ),
    )


    id = Column(
        Integer,
        primary_key=True
    )


    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )


    agent_id = Column(
        Integer,
        ForeignKey("agents.id"),
        nullable=False
    )


    title = Column(
        String(200),
        nullable=True
    )


    summary = Column(
        Text,
        nullable=True
    )


    agent_snapshot = Column(
        JSON,
        nullable=True
    )


    status = Column(
        String(20),
        default="active",
        nullable=False
    )


    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )


    last_message_time = Column(
        DateTime(timezone=True),
        nullable=True
    )


    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True
    )


    user = relationship(
        "User",
        back_populates="conversations"
    )


    agent = relationship(
        "Agent",
        back_populates="conversations"
    )


    

    messages = relationship(
    "Message",
    back_populates="conversation"
    )