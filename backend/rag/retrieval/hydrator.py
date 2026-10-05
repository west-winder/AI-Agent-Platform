from backend.rag.contracts import (
    Chunk,
    Document,
    RetrievedChunk,
    SearchResult,
)


class RAGHydrator:
    """
    RAG Retrieval 阶段的 Hydration 组件。

    职责：

    SearchResult[]
        ↓
    根据 chunk_id 查找 Chunk
        ↓
    根据 Chunk.document_id 查找 Document
        ↓
    组合成 RetrievedChunk[]
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
        search_results: list[SearchResult],
        chunk_store: dict[str, Chunk],
        document_store: dict[str, Document],
    ) -> list[RetrievedChunk]:
        """
        将轻量 SearchResult 恢复成完整 RetrievedChunk。

        参数：
            search_results:
                Exact Search 返回的检索结果。

            chunk_store:
                chunk_id -> Chunk

            document_store:
                document_id -> Document

        返回：
            与 search_results 顺序一致的 RetrievedChunk 列表。
        """

        # --------------------------------------------------
        # 1. Empty Retrieval 是合法状态
        # --------------------------------------------------

        if not search_results:
            return []

        # --------------------------------------------------
        # 2. 按 SearchResult 顺序逐个 Hydration
        # --------------------------------------------------

        retrieved_chunks: list[RetrievedChunk] = []

        for search_result in search_results:

            # ----------------------------------------------
            # 2.1 SearchResult -> Chunk
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
            # 2.3 组合完整 RetrievedChunk
            # ----------------------------------------------

            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                source=document.file_name,
                score=search_result.score,
            )

            retrieved_chunks.append(
                retrieved_chunk
            )

        # --------------------------------------------------
        # 3. 返回完整 Retrieval Result
        # --------------------------------------------------

        return retrieved_chunks