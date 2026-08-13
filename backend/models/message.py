from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Index
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database.database import Base


class Message(Base):

    __tablename__ = "messages"

    # 联合索引：
    # 优化根据conversation查询历史消息
    __table_args__ = (
        Index(
            "idx_messages_conversation_time",
            "conversation_id",
            "created_at"
        ),
    )


    id = Column(
        Integer,
        primary_key=True
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


    # 数据库字段：metadata
    # Python属性：message_metadata
    message_metadata = Column(
        "metadata",
        JSON,
        nullable=True
    )


    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


    conversation = relationship(
        "Conversation",
        back_populates="messages"
    )