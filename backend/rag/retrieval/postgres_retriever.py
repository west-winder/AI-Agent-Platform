from backend.rag.contracts import (
    DenseRetrieverResult,
    RagRetrievedChunk,
    RetrievedCandidate,
)
from backend.rag.retrieval.dense_retriever import (
    PostgreSQLVectorSearch,
)
from backend.rag.retrieval.postgres_hydrator import (
    PostgreSQLHydrator,
)


class PostgreSQLRAGRetriever:
    """
    PostgreSQL Persistent Retrieval Pipeline 的 Orchestrator。

    Pipeline Stages：

    Query
        ↓
    Embedding Stage          (_embed_query)
        ↓
    Retrieval Stage          (_retrieve_candidates)
        ↓
    DenseRetrieverResult[]
        ↓
    << Future Fusion / Reranker Insertion Point >>
        ↓
    RetrievedCandidate[]
        ↓
    Hydration Stage          (_hydrate)
        ↓
    RagRetrievedChunk[]

    本组件只负责编排，不实现：
    - Embedding 算法
    - Vector Similarity
    - SQL 查询细节
    - Hydration 细节

    当前只实现 Dense Retrieval。
    Sparse Retrieval（BM25）、Fusion（RRF）与 Reranker
    会在未来插入到上述 Insertion Point，本类结构已为其预留位置。
    """

    def __init__(
        self,
        embedder,
        vector_search: PostgreSQLVectorSearch,
        hydrator: PostgreSQLHydrator,
    ):
        self._embedder = embedder
        self._vector_search = vector_search
        self._hydrator = hydrator

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RagRetrievedChunk]:
        """
        执行一次完整的 Persistent RAG Retrieval。

        外部契约不变：
            输入 Query，
            返回按 Retrieval Ranking 排序的 RagRetrievedChunk[]。
        """

        # ==========================================
        # 1. Embedding Stage
        # ==========================================

        query_vector = self._embed_query(query)

        # ==========================================
        # 2. Retrieval Stage
        # ==========================================

        dense_results = self._retrieve_candidates(
            query_vector=query_vector,
            top_k=top_k,
        )

        # ==========================================
        # 3. Future Fusion / Reranker Insertion Point
        # ==========================================
        #
        # 未来阶段在这里插入：
        #
        #     DenseRetrieverResult[]  +  BM25RetrieverResult[]
        #         ↓  Fusion(RRF)
        #     FusedRetrieverResult[]
        #         ↓  Reranker
        #     RerankedResult[]
        #
        # 插入后只需把最终结果适配成 RetrievedCandidate[]，
        # 下游 Hydration Stage 无需任何改动。

        candidates = [
            RetrievedCandidate(
                chunk_id=result.chunk_id,
                rank=result.dense_rank,
            )
            for result in dense_results
        ]

        # ==========================================
        # 4. Hydration Stage
        # ==========================================

        return self._hydrate(candidates)

    def _embed_query(
        self,
        query: str,
    ):
        """
        Embedding Stage：把 Query 转换成 Query Vector。
        """

        return self._embedder.embed(query)

    def _retrieve_candidates(
        self,
        query_vector,
        top_k: int,
    ) -> list[DenseRetrieverResult]:
        """
        Retrieval Stage：候选获取阶段。

        它代表「获取候选 Chunk」这一阶段，
        不绑定具体 Retrieval 算法。

        当前实现内部使用 Persistent Dense Retrieval。
        """

        return self._vector_search.search(
            query_vector=query_vector,
            embedding_model=(
                self._embedder.model_name
            ),
            top_k=top_k,
        )

    def _hydrate(
        self,
        candidates: list[RetrievedCandidate],
    ) -> list[RagRetrievedChunk]:
        """
        Hydration Stage：把候选恢复成完整 RagRetrievedChunk[]。
        """

        return self._hydrator.hydrate(candidates)
