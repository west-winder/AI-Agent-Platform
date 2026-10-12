from backend.rag.contracts import (
    Chunk,
    Document,
    RagRetrievedChunk,
    RetrievedCandidate,
)


class RAGHydrator:
    """
    RAG Retrieval Pipeline 的 Hydration Stage。

    它是流水线的最后一个转换阶段。

    职责：

    RetrievedCandidate[]
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
    2. Retrieval（Dense / Sparse）
    3. Ranking（不重排、不算分）
    4. Context Assembly
    5. LLM Generation

    它只做一件事：

    按输入顺序把 chunk_id 恢复成完整的 RagRetrievedChunk，
    并把上游 Retrieval / Ranking 阶段给出的业务排名
    原样带入 retrieval_rank。
    """

    def hydrate(
        self,
        candidates: list[RetrievedCandidate],
        chunk_store: dict[str, Chunk],
        document_store: dict[str, Document],
    ) -> list[RagRetrievedChunk]:
        """
        按输入顺序，把 RetrievedCandidate 恢复成完整 RagRetrievedChunk。

        参数：
            candidates:
                Retrieval / Ranking 阶段给出的候选，
                已携带 1-based 业务排名。

            chunk_store:
                chunk_id -> Chunk

            document_store:
                document_id -> Document

        返回：
            与 candidates 顺序一致的 RagRetrievedChunk 列表。
            本方法不重排、不算分、不改变顺序。
        """

        # --------------------------------------------------
        # 1. Empty Retrieval 是合法状态
        # --------------------------------------------------

        if not candidates:
            return []

        # --------------------------------------------------
        # 2. 按 RetrievedCandidate 顺序逐个 Hydration
        # --------------------------------------------------

        retrieved_chunks: list[RagRetrievedChunk] = []

        for candidate in candidates:

            # ----------------------------------------------
            # 2.1 RetrievedCandidate -> Chunk
            # ----------------------------------------------

            chunk = chunk_store.get(
                candidate.chunk_id
            )

            if chunk is None:
                raise KeyError(
                    "Hydration failed: "
                    f"chunk '{candidate.chunk_id}' "
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
                retrieval_rank=candidate.rank,
            )

            retrieved_chunks.append(
                retrieved_chunk
            )

        # --------------------------------------------------
        # 3. 返回完整 Retrieval Result
        # --------------------------------------------------

        return retrieved_chunks