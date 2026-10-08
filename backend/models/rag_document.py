from sqlalchemy import Column, String, CheckConstraint
from sqlalchemy.orm import relationship
from backend.database.database import Base


class DocumentORM(Base):
    __tablename__ = "documents"

    __table_args__ = (
        CheckConstraint(
            "file_type IN ('txt', 'markdown', 'pdf')",
            name="ck_documents_file_type",
        ),
    )

    document_id = Column(
        String(128),
        primary_key=True,
    )

    file_name = Column(
        String(255),
        nullable=False,
    )

    file_type = Column(
        String(20),
        nullable=False,
    )

    chunks = relationship(
        "ChunkORM",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )