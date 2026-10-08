from backend.rag.contracts import (
    RetrievedChunk,
    SearchResult,
)
from backend.rag.retrieval.repositories import (
    ChunkRepository,
    DocumentRepository,
)


class PostgreSQLHydrator:
    """
    Persistent RAG Hydration 组件。

    职责：

    SearchResult[]
        ↓
    ChunkRepository 批量查询 Chunk
        ↓
    DocumentRepository 批量查询 Document
        ↓
    一致性检查
        ↓
    按 SearchResult 顺序恢复 Ranking
        ↓
    RetrievedChunk[]

    不负责：
    - Query Embedding
    - Vector Search
    - SQL 查询细节
    - Ranking 算法
    - Context Assembly
    - LLM Generation
    """

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        document_repository: DocumentRepository,
    ):
        self._chunk_repository = chunk_repository
        self._document_repository = document_repository

    def hydrate(
        self,
        search_results: list[SearchResult],
    ) -> list[RetrievedChunk]:

        # ==========================================
        # 1. Empty Retrieval 是合法状态
        # ==========================================

        if not search_results:
            return []

        # ==========================================
        # 2. 提取 SearchResult 中的 chunk_id
        # ==========================================

        chunk_ids = [
            result.chunk_id
            for result in search_results
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
                "SearchResult references missing chunks: "
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

        retrieved_chunks: list[RetrievedChunk] = []

        for search_result in search_results:

            chunk = chunk_by_id[
                search_result.chunk_id
            ]

            document = document_by_id[
                chunk.document_id
            ]

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    source=document.file_name,
                    score=search_result.score,
                )
            )

        return retrieved_chunks