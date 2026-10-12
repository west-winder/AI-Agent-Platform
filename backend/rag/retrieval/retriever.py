import numpy as np

from backend.rag.contracts import (
    BM25RetrieverResult,
    Chunk,
    DenseRetrieverResult,
    Document,
    FusedRetrieverResult,
    RagRetrievedChunk,
    RerankedResult,
    RetrievedCandidate,
)
from backend.rag.retrieval.bm25_retriever import BM25Retriever
from backend.rag.retrieval.exact_search import ExactDenseSearch
from backend.rag.retrieval.fusion import RRFFusion
from backend.rag.retrieval.hydrator import RAGHydrator
from backend.rag.retrieval.rerank_document_builder import (
    RerankDocumentBuilder,
)
from backend.rag.retrieval.reranker_adapter import (
    RerankerAdapter,
)


class RAGRetriever:
    """
    RAG Hybrid Retrieval Pipeline 的 Orchestrator。

    Pipeline Stages：

    Query
        ↓
    Embedding Stage          (_embed_query)
        ↓
    Retrieval Stage          (_dense_retrieve_candidates
                              + _retrieve_sparse_candidates)
        ↓
    DenseRetrieverResult[]  +  BM25RetrieverResult[]
        ↓
    Fusion Stage             (_fuse)
        ↓
    FusedRetrieverResult[]
        ↓
    Rerank Stage             (_rerank)
        ↓
    RerankedResult[]
        ↓
    RetrievedCandidate[]
        ↓
    Hydration Stage          (_hydrate)
        ↓
    RagRetrievedChunk[]

    本组件只负责编排 Retrieval Pipeline。

    不负责：

    1. Embedding 算法实现
    2. Similarity 计算
    3. BM25 打分
    4. RRF 融合算法实现
    5. Rerank 模型
    6. Hydration 细节
    7. Context Assembly
    8. LLM Generation

    它没有第二套 Hybrid Pipeline：
    Sparse Retrieval / Fusion / Rerank 都是本 Pipeline 的阶段。
    """

    def __init__(
        self,
        embedder,
        exact_search: ExactDenseSearch,
        bm25_retriever: BM25Retriever,
        fusion: RRFFusion,
        rerank_document_builder: RerankDocumentBuilder,
        reranker: RerankerAdapter,
        hydrator: RAGHydrator,
    ):
        """
        通过 Dependency Injection 注入 Pipeline 各阶段组件。
        """

        self._embedder = embedder
        self._exact_search = exact_search
        self._bm25_retriever = bm25_retriever
        self._fusion = fusion
        self._rerank_document_builder = (
            rerank_document_builder
        )
        self._reranker = reranker
        self._hydrator = hydrator

    def retrieve(
        self,
        query: str,
        chunk_vectors: dict[str, np.ndarray],
        chunk_store: dict[str, Chunk],
        document_store: dict[str, Document],
        top_k: int = 5,
    ) -> list[RagRetrievedChunk]:
        """
        执行一次完整的 Hybrid RAG Retrieval。

        外部契约不变：
            输入 Query，
            返回按最终 Ranking 排序的 RagRetrievedChunk[]。
        """

        # --------------------------------------------------
        # 1. Embedding Stage
        # --------------------------------------------------

        query_vector = self._embed_query(query)

        # --------------------------------------------------
        # 2. Retrieval Stage：Dense
        # --------------------------------------------------

        dense_results = self._dense_retrieve_candidates(
            query_vector=query_vector,
            chunk_vectors=chunk_vectors,
            top_k=top_k,
        )

        # --------------------------------------------------
        # 3. Retrieval Stage：Sparse（BM25）
        # --------------------------------------------------

        bm25_results = self._retrieve_sparse_candidates(
            query=query,
            chunk_store=chunk_store,
            top_k=top_k,
        )

        # --------------------------------------------------
        # 4. Fusion Stage
        # --------------------------------------------------

        fused_results = self._fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        # --------------------------------------------------
        # 5. Rerank Stage
        # --------------------------------------------------

        reranked_results = self._rerank(
            query=query,
            fused_results=fused_results,
            chunk_store=chunk_store,
        )

        # --------------------------------------------------
        # 6. RerankedResult[] -> RetrievedCandidate[]
        # --------------------------------------------------
        #
        # Hydrator 只认 RetrievedCandidate，
        # 不感知上游是 Dense / BM25 / Fusion / Reranker。
        # 最终条数在这里统一裁剪为 top_k。

        candidates = [
            RetrievedCandidate(
                chunk_id=result.chunk_id,
                rank=result.rerank_rank,
            )
            for result in reranked_results[:top_k]
        ]

        # --------------------------------------------------
        # 7. Hydration Stage
        # --------------------------------------------------

        return self._hydrate(
            candidates=candidates,
            chunk_store=chunk_store,
            document_store=document_store,
        )

    def _embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Embedding Stage：把 Query 转换成 Query Vector。
        """

        return self._embedder.embed(query)

    def _dense_retrieve_candidates(
        self,
        query_vector: np.ndarray,
        chunk_vectors: dict[str, np.ndarray],
        top_k: int,
    ) -> list[DenseRetrieverResult]:
        """
        Retrieval Stage：候选获取阶段（Dense 路径）。

        与 _retrieve_sparse_candidates 命名对称。

        当前 Dense 路径内部使用 Dense Retrieval。
        """

        return self._exact_search.search(
            query_vector=query_vector,
            chunk_vectors=chunk_vectors,
            top_k=top_k,
        )

    def _retrieve_sparse_candidates(
        self,
        query: str,
        chunk_store: dict[str, Chunk],
        top_k: int,
    ) -> list[BM25RetrieverResult]:
        """
        Retrieval Stage：候选获取阶段（Sparse 路径）。

        BM25 需要 chunk 文本，
        但调用方不应该知道 chunk_store：

        文本在内部从 chunk_store 派生，
        因此 retrieve() 的外部签名保持不变。
        """

        chunk_texts = {
            chunk_id: chunk.content
            for chunk_id, chunk in chunk_store.items()
        }

        return self._bm25_retriever.search(
            query=query,
            chunk_texts=chunk_texts,
            top_k=top_k,
        )

    def _fuse(
        self,
        dense_results: list[DenseRetrieverResult],
        bm25_results: list[BM25RetrieverResult],
    ) -> list[FusedRetrieverResult]:
        """
        Fusion Stage：融合 Dense 与 BM25 候选。

        这里不裁剪：
        先保留完整候选池交给 Reranker，
        最终条数在进入 Hydration 前统一裁剪。
        """

        return self._fusion.fuse(
            dense_results,
            bm25_results,
        )

    def _rerank(
        self,
        query: str,
        fused_results: list[FusedRetrieverResult],
        chunk_store: dict[str, Chunk],
    ) -> list[RerankedResult]:
        """
        Rerank Stage：

        FusedRetrieverResult[]
            ↓
        RerankDocumentBuilder（按 chunk_id 取文本）
            ↓
        RerankerAdapter（二次排序）
            ↓
        RerankedResult[]

        Reranker 不访问 chunk_store。
        """

        documents = self._rerank_document_builder.build(
            fused_results,
            chunk_store,
        )

        return self._reranker.rerank(
            query,
            documents,
        )

    def _hydrate(
        self,
        candidates: list[RetrievedCandidate],
        chunk_store: dict[str, Chunk],
        document_store: dict[str, Document],
    ) -> list[RagRetrievedChunk]:
        """
        Hydration Stage：把候选恢复成完整 RagRetrievedChunk[]。
        """

        return self._hydrator.hydrate(
            candidates=candidates,
            chunk_store=chunk_store,
            document_store=document_store,
        )
