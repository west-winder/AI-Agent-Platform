from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database.database import Base


class Agent(Base):

    __tablename__ = "agents"


    id = Column(
        Integer,
        primary_key=True,
        index=True
    )


    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )


    name = Column(
        String(100),
        nullable=False
    )


    system_prompt = Column(
        Text,
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


    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True
    )


    user = relationship(
        "User",
        back_populates="agents"
    )


    conversations = relationship(
    "Conversation",
    back_populates="agent"
    )