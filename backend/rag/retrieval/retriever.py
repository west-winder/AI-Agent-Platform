import numpy as np

from backend.rag.contracts import (
    Chunk,
    Document,
    RetrievedChunk,
)
from backend.rag.retrieval.exact_search import ExactDenseSearch
from backend.rag.retrieval.hydrator import RAGHydrator


class RAGRetriever:
    """
    RAG Retrieval 阶段的 Orchestrator。

    职责：

    Query
        ↓
    Embedder
        ↓
    Query Vector
        ↓
    ExactDenseSearch
        ↓
    SearchResult[]
        ↓
    RAGHydrator
        ↓
    RetrievedChunk[]

    本组件只负责编排 Retrieval Pipeline。

    不负责：

    1. Embedding 算法实现
    2. Similarity 计算
    3. Ranking 算法实现
    4. Hydration 细节
    5. Context Assembly
    6. LLM Generation
    """

    def __init__(
        self,
        embedder,
        exact_search: ExactDenseSearch,
        hydrator: RAGHydrator,
    ):
        """
        通过 Dependency Injection 注入 Retrieval 所需组件。
        """

        self._embedder = embedder
        self._exact_search = exact_search
        self._hydrator = hydrator

    def retrieve(
        self,
        query: str,
        chunk_vectors: dict[str, np.ndarray],
        chunk_store: dict[str, Chunk],
        document_store: dict[str, Document],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        执行一次完整的 RAG Retrieval。

        返回：
            按 Retrieval Ranking 排序的 RetrievedChunk[]。
        """

        # --------------------------------------------------
        # 1. Query -> Embedding
        # --------------------------------------------------

        query_vector = self._embedder.embed(
            query
        )

        # --------------------------------------------------
        # 2. Exact Dense Search
        # --------------------------------------------------

        search_results = self._exact_search.search(
            query_vector=query_vector,
            chunk_vectors=chunk_vectors,
            top_k=top_k,
        )

        # --------------------------------------------------
        # 3. Hydration
        # --------------------------------------------------

        retrieved_chunks = self._hydrator.hydrate(
            search_results=search_results,
            chunk_store=chunk_store,
            document_store=document_store,
        )

        # --------------------------------------------------
        # 4. Return Retrieval Result
        # --------------------------------------------------

        return retrieved_chunks