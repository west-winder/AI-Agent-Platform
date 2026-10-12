from backend.rag.contracts import (
    Chunk,
    DenseRetrieverResult,
    Document,
    RagRetrievedChunk,
)


class RAGHydrator:
    """
    RAG Retrieval 阶段的 Hydration 组件。

    职责：

    DenseRetrieverResult[]
        ↓
    根据 chunk_id 查找 Chunk
        ↓
    根据 Chunk.document_id 查找 Document
        ↓
    组合成 RagRetrievedChunk[]
        ↓
    返回给 RAGRetriever

    本组件不负责：

    1. Query Embedding
    2. Vector Search
    3. Ranking
    4. Context Assembly
    5. LLM Generation
    """

    def hydrate(
        self,
        search_results: list[DenseRetrieverResult],
        chunk_store: dict[str, Chunk],
        document_store: dict[str, Document],
    ) -> list[RagRetrievedChunk]:
        """
        将轻量 DenseRetrieverResult 恢复成完整 RagRetrievedChunk。

        参数：
            search_results:
                Dense Retriever 返回的检索结果。

            chunk_store:
                chunk_id -> Chunk

            document_store:
                document_id -> Document

        返回：
            与 search_results 顺序一致的 RagRetrievedChunk 列表。
        """

        # --------------------------------------------------
        # 1. Empty Retrieval 是合法状态
        # --------------------------------------------------

        if not search_results:
            return []

        # --------------------------------------------------
        # 2. 按 DenseRetrieverResult 顺序逐个 Hydration
        # --------------------------------------------------

        retrieved_chunks: list[RagRetrievedChunk] = []

        for search_result in search_results:

            # ----------------------------------------------
            # 2.1 DenseRetrieverResult -> Chunk
            # ----------------------------------------------

            chunk = chunk_store.get(
                search_result.chunk_id
            )

            if chunk is None:
                raise KeyError(
                    "Hydration failed: "
                    f"chunk '{search_result.chunk_id}' "
                    "does not exist in chunk_store"
                )

            # ----------------------------------------------
            # 2.2 Chunk -> Document
            # ----------------------------------------------

            document = document_store.get(
                chunk.document_id
            )

            if document is None:
                raise KeyError(
                    "Hydration failed: "
                    f"document '{chunk.document_id}' "
                    "does not exist in document_store"
                )

            # ----------------------------------------------
            # 2.3 组合完整 RagRetrievedChunk
            # ----------------------------------------------

            retrieved_chunk = RagRetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                source=document.file_name,
                retrieval_rank=search_result.dense_rank,
            )

            retrieved_chunks.append(
                retrieved_chunk
            )

        # --------------------------------------------------
        # 3. 返回完整 Retrieval Result
        # --------------------------------------------------

        return retrieved_chunks