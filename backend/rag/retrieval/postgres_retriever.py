from backend.rag.contracts import RetrievedChunk
from backend.rag.retrieval.postgres_vector_search import (
    PostgreSQLVectorSearch,
)
from backend.rag.retrieval.postgres_hydrator import (
    PostgreSQLHydrator,
)


class PostgreSQLRAGRetriever:
    """
    PostgreSQL Persistent Retrieval Orchestrator。

    职责：

    Query
        ↓
    Embedder
        ↓
    Query Vector
        ↓
    PostgreSQLVectorSearch
        ↓
    SearchResult[]
        ↓
    PostgreSQLHydrator
        ↓
    RetrievedChunk[]

    本组件只负责编排，不实现：
    - Embedding 算法
    - Vector Similarity
    - SQL 查询细节
    - Hydration 细节
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
    ) -> list[RetrievedChunk]:

        # ==========================================
        # 1. Query -> Embedding
        # ==========================================

        query_vector = self._embedder.embed(
            query
        )

        # ==========================================
        # 2. Persistent Vector Search
        # ==========================================

        search_results = (
            self._vector_search.search(
                query_vector=query_vector,
                embedding_model=(
                    self._embedder.model_name
                ),
                top_k=top_k,
            )
        )

        # ==========================================
        # 3. Persistent Hydration
        # ==========================================

        retrieved_chunks = (
            self._hydrator.hydrate(
                search_results
            )
        )

        return retrieved_chunks