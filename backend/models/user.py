from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from backend.database.database import Base


class User(Base):

    __tablename__ = "users"


    id = Column(
        Integer,
        primary_key=True
    )


    username = Column(
        String(50),
        nullable=False
    )


    email = Column(
        String(100),
        nullable=True
    )


    conversations = relationship(
        "Conversation",
        back_populates="user"
    )


    agents = relationship(
        "Agent",
        back_populates="user"
    )