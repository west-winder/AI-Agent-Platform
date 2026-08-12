from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.database.database import Base


class Model(Base):

    __tablename__ = "models"


    id = Column(
        Integer,
        primary_key=True,
        index=True
    )


    provider = Column(
        String(50),
        nullable=False
    )


    model_name = Column(
        String(100),
        nullable=False
    )


    display_name = Column(
        String(100),
        nullable=True
    )


    api_endpoint = Column(
        String(255),
        nullable=True
    )


    context_window = Column(
        Integer,
        nullable=True
    )


    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


    agents = relationship(
    "Agent",
    back_populates="model"
)