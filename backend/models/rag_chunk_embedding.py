from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import VECTOR

from backend.database.database import Base


class ChunkEmbeddingORM(Base):
    __tablename__ = "chunk_embeddings"

    # 联合主键第一部分：所属 Chunk
    chunk_id = Column(
        String(255),
        ForeignKey(
            "chunks.chunk_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    # 联合主键第二部分：Embedding Model
    embedding_model = Column(
        String(255),
        primary_key=True,
    )

    # 向量数据，不固定维度
    embedding = Column(
        VECTOR(),
        nullable=False,
    )

    # ORM 对象关系
    chunk = relationship(
        "ChunkORM",
        back_populates="embeddings",
    )