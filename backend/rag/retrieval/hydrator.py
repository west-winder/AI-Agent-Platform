from typing import TYPE_CHECKING

from backend.rag.contracts import (
    RagRetrievedChunk,
    RetrievedCandidate,
)

# Hydrator 只依赖 Repository Contract：
# 这里仅做静态类型引用，运行时不 import repositories，
# 因此 Hydrator 不会传递性依赖 SQLAlchemy / Session / ORM。
if TYPE_CHECKING:
    from backend.rag.retrieval.repositories import (
        ChunkRepository,
        DocumentRepository,
    )


class RAGHydrator:
    """
    RAG Retrieval Pipeline 的 Hydration Stage。

    它是流水线的最后一个转换阶段。

    职责：

    RetrievedCandidate[]
        ↓
    ChunkRepository 批量查询 Chunk
        ↓
    DocumentRepository 批量查询 Document
        ↓
    一致性检查
        ↓
    按 RetrievedCandidate 顺序恢复 Ranking
        ↓
    RagRetrievedChunk[]

    它只依赖 Repository Contract：

    生产注入 PostgreSQL Repository，
    测试注入 InMemory Repository。

    本组件不感知：
    - PostgreSQL
    - dict
    - SQLAlchemy
    - 任何 storage implementation

    也不负责：

    1. Query Embedding
    2. Retrieval（Dense / Sparse）
    3. Ranking（不重排、不算分）
    4. Context Assembly
    5. LLM Generation
    """

    def __init__(
        self,
        chunk_repository: "ChunkRepository",
        document_repository: "DocumentRepository",
    ):
        self._chunk_repository = chunk_repository
        self._document_repository = document_repository

    def hydrate(
        self,
        candidates: list[RetrievedCandidate],
    ) -> list[RagRetrievedChunk]:
        """
        按输入顺序，把 RetrievedCandidate 恢复成完整 RagRetrievedChunk。

        参数：
            candidates:
                Retrieval / Ranking 阶段给出的候选，
                已携带 1-based 业务排名。

        返回：
            与 candidates 顺序一致的 RagRetrievedChunk 列表。
            本方法不重排、不算分、不改变顺序。
        """

        # ==========================================
        # 1. Empty Retrieval 是合法状态
        # ==========================================

        if not candidates:
            return []

        # ==========================================
        # 2. 提取 RetrievedCandidate 中的 chunk_id
        # ==========================================

        chunk_ids = [
            candidate.chunk_id
            for candidate in candidates
        ]

        # ==========================================
        # 3. Batch Query：Chunk
        # ==========================================

        chunks = self._chunk_repository.get_by_ids(
            chunk_ids
        )

        chunk_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

        # ==========================================
        # 4. Chunk 一致性检查
        # ==========================================

        missing_chunk_ids = [
            chunk_id
            for chunk_id in chunk_ids
            if chunk_id not in chunk_by_id
        ]

        if missing_chunk_ids:
            raise KeyError(
                "Hydration failed: "
                "RetrievedCandidate references missing chunks: "
                f"{missing_chunk_ids}"
            )

        # ==========================================
        # 5. 提取 document_id
        # ==========================================

        document_ids = list(
            dict.fromkeys(
                chunk_by_id[chunk_id].document_id
                for chunk_id in chunk_ids
            )
        )

        # ==========================================
        # 6. Batch Query：Document
        # ==========================================

        documents = self._document_repository.get_by_ids(
            document_ids
        )

        document_by_id = {
            document.document_id: document
            for document in documents
        }

        # ==========================================
        # 7. Document 一致性检查
        # ==========================================

        missing_document_ids = [
            document_id
            for document_id in document_ids
            if document_id not in document_by_id
        ]

        if missing_document_ids:
            raise KeyError(
                "Hydration failed: "
                "chunks reference missing documents: "
                f"{missing_document_ids}"
            )

        # ==========================================
        # 8. Order Preservation + Hydration
        # ==========================================

        retrieved_chunks: list[RagRetrievedChunk] = []

        for candidate in candidates:

            chunk = chunk_by_id[
                candidate.chunk_id
            ]

            document = document_by_id[
                chunk.document_id
            ]

            retrieved_chunks.append(
                RagRetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    source=document.file_name,
                    retrieval_rank=candidate.rank,
                )
            )

        return retrieved_chunks
