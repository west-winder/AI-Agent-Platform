from backend.rag.contracts import (
    Chunk,
    FusedRetrieverResult,
    RerankDocument,
)


class RerankDocumentBuilder:
    """
    把 Fusion 输出转换成 Reranker 输入。

    职责：

    FusedRetrieverResult[]（只有 chunk_id / rank / score）
        ↓
    按 chunk_id 获取文本
        ↓
    RerankDocument[]

    本组件不负责：

    1. Retrieval
    2. Ranking
    3. 打分
    4. Hydration
    5. Context Assembly

    有了本组件，Reranker 就不需要访问
    Database / chunk_store / Document store。

    它保持 Fusion 的既有顺序，
    因此 RerankDocument 的位置就是 previous_rank。
    """

    def build(
        self,
        fused_results: list[FusedRetrieverResult],
        chunk_store: dict[str, Chunk],
    ) -> list[RerankDocument]:
        """
        按 chunk_id 取回文本，构造 RerankDocument[]。

        参数：
            fused_results:
                Fusion 输出。

            chunk_store:
                chunk_id -> Chunk

        返回：
            与 fused_results 顺序一致的 RerankDocument 列表。
        """

        if not isinstance(fused_results, list):
            raise TypeError(
                "fused_results must be "
                "list[FusedRetrieverResult]"
            )

        if not isinstance(chunk_store, dict):
            raise TypeError(
                "chunk_store must be dict[str, Chunk]"
            )

        # --------------------------------------------------
        # Empty Retrieval 是合法状态
        # --------------------------------------------------

        if not fused_results:
            return []

        documents: list[RerankDocument] = []

        for result in fused_results:

            chunk = chunk_store.get(result.chunk_id)

            if chunk is None:
                raise KeyError(
                    "RerankDocument build failed: "
                    f"chunk '{result.chunk_id}' "
                    "does not exist in chunk_store"
                )

            documents.append(
                RerankDocument(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                )
            )

        return documents
