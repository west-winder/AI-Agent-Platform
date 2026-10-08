from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from backend.database.database import Base


class ChunkORM(Base):
    __tablename__ = "chunks"

    # Chunk 唯一标识
    chunk_id = Column(
        String(255),
        primary_key=True,
    )

    # 所属 Document
    document_id = Column(
        String(128),
        ForeignKey(
            "documents.document_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # Chunk 正文
    content = Column(
        Text,
        nullable=False,
    )

    # ORM 对象关系
    document = relationship(
        "DocumentORM",
        back_populates="chunks",
    )


    embeddings = relationship(
        "ChunkEmbeddingORM",
        back_populates="chunk",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )