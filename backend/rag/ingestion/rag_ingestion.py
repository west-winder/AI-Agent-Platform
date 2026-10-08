
from dataclasses import dataclass
from typing import Literal

import numpy as np

from backend.database.database import SessionLocal
from backend.models.rag_document import DocumentORM
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM

from backend.rag.contracts import Document
from backend.rag.ingestion.chunker import chunk_document
from backend.rag.ingestion.chunk_embedding import build_chunk_vectors
from backend.rag.ingestion.ingestion_validator import (
    validate_ingestion_data,
)


# ==================================================
# Ingestion Result Contract
# ==================================================

@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    status: Literal["success", "empty"]
    chunk_count: int


# ==================================================
# RAG Ingestion Orchestrator
# ==================================================

class RAGIngestionService:
    """
    Persistent RAG Ingestion V1。

    职责：
    1. 编排 Chunking
    2. 编排 Embedding
    3. 调用 IngestionValidator
    4. 管理整次数据库事务
    5. 返回明确的 IngestionResult

    不负责：
    - Chunking 算法
    - Embedding 算法
    - Vector Search
    - Retrieval
    - LLM Generation
    """

    def __init__(
        self,
        embedder,
        session_factory=SessionLocal,
    ):
        self._embedder = embedder
        self._session_factory = session_factory

    def ingest(
        self,
        document: Document,
        raw_text: str,
        chunk_size: int,
    ) -> IngestionResult:
        """
        导入一个 Document。

        规则：
        - Empty Document -> empty，不写数据库
        - Duplicate Document -> 数据库主键拒绝
        - Validation Failure -> raise，不写数据库
        - Persistence Failure -> rollback + raise
        - Success -> 一次性 commit
        """

        # ------------------------------------------
        # 1. Chunking
        # ------------------------------------------

        chunks = chunk_document(
            document=document,
            raw_text=raw_text,
            chunk_size=chunk_size,
        )

        # ------------------------------------------
        # 2. Empty Document
        # ------------------------------------------

        if not chunks:
            return IngestionResult(
                document_id=document.document_id,
                status="empty",
                chunk_count=0,
            )

        # ------------------------------------------
        # 3. Corpus-side Embedding
        # ------------------------------------------

        chunk_vectors = build_chunk_vectors(
            chunks=chunks,
            embedder=self._embedder,
        )

        # ------------------------------------------
        # 4. Model Identity + Validation
        # ------------------------------------------

        embedding_model = self._embedder.model_name
        embedding_dimension = self._embedder.dimension

        validate_ingestion_data(
            chunks=chunks,
            chunk_vectors=chunk_vectors,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

        # ------------------------------------------
        # 5. Atomic Persistence
        # ------------------------------------------

        # 只有在全部数据准备并校验通过后，
        # 才开启数据库事务。

        with self._session_factory.begin() as session:

            # 5.1 Document
            session.add(
                DocumentORM(
                    document_id=document.document_id,
                    file_name=document.file_name,
                    file_type=document.file_type,
                )
            )

            session.flush()

            # 5.2 Chunks
            session.add_all([
                ChunkORM(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                )
                for chunk in chunks
            ])

            session.flush()

            # 5.3 Chunk Embeddings
            session.add_all([
                ChunkEmbeddingORM(
                    chunk_id=chunk.chunk_id,
                    embedding_model=embedding_model,
                    embedding=np.asarray(
                        chunk_vectors[chunk.chunk_id],
                        dtype=float,
                    ).tolist(),
                )
                for chunk in chunks
            ])

            # 退出上下文时：
            # 成功 -> COMMIT
            # 异常 -> ROLLBACK

        # ------------------------------------------
        # 6. Success
        # ------------------------------------------

        return IngestionResult(
            document_id=document.document_id,
            status="success",
            chunk_count=len(chunks),
        )
