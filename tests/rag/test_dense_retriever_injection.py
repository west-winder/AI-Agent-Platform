"""
RAGRetriever 的 DenseRetriever 注入契约测试（确定性，不联网 / 不依赖数据库）。

验证 RAGRetriever 只按 DenseRetriever Contract 调用 dense 路径：

1. Query -> Embedder -> Query Vector
2. Query Vector + top_k -> 注入的 DenseRetriever
3. DenseRetrieverResult.dense_rank -> RetrievedCandidate.rank
4. Empty Retrieval 合法
5. Dense 失败必须在 Hydration 之前向上传播

持久化形态（真实 PostgreSQLVectorSearch + RAGHydrator）见
test_rag_retriever_postgres.py。
"""

import numpy as np
import pytest

from backend.rag.contracts import (
    Chunk,
    DenseRetrieverResult,
    RagRetrievedChunk,
)
from backend.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)
from backend.rag.retrieval.fusion import RRFFusion
from backend.rag.retrieval.rerank_document_builder import (
    RerankDocumentBuilder,
)
from backend.rag.retrieval.reranker_adapter import (
    RerankerAdapter,
)
from backend.rag.retrieval.retriever import RAGRetriever


# ==================================================
# Fake Components
# ==================================================

class FakeEmbedder:
    model_name = "test-model"

    def __init__(self):
        self.received_query = None

    def embed(self, query):
        self.received_query = query
        return np.array([1.0, 0.0, 0.0])


class FakeDenseRetriever:
    """
    DenseRetriever Contract 的 Fake 实现。

    只暴露 Contract 要求的 search(query_vector, top_k)，
    RAGRetriever 不应该知道它背后是 dict 还是 PostgreSQL。
    """

    def __init__(self):
        self.received_query_vector = None
        self.received_top_k = None

    def search(
        self,
        query_vector,
        top_k=5,
    ):
        self.received_query_vector = query_vector
        self.received_top_k = top_k

        return [
            DenseRetrieverResult(
                chunk_id="chunk_2",
                dense_score=0.9,
                dense_rank=1,
            ),
            DenseRetrieverResult(
                chunk_id="chunk_1",
                dense_score=0.8,
                dense_rank=2,
            ),
        ]


class FakeRerankItem:
    """
    模拟 backend.reranking.reranker.RerankResult 的字段契约。
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score


class PreservingReranker:
    """
    保持 Fusion 顺序，避免测试关注真实 Rerank 模型。
    """

    def rerank(self, query, texts):
        return [
            FakeRerankItem(
                index=index,
                score=1.0 / (index + 1),
            )
            for index, text in enumerate(texts)
        ]


class FakeHydrator:
    """
    记录进入 Hydration 的候选，返回固定结果。
    """

    def __init__(self):
        self.received_candidates = None

    def hydrate(
        self,
        candidates,
    ):
        self.received_candidates = candidates

        if not candidates:
            return []

        return [
            RagRetrievedChunk(
                chunk_id="chunk_2",
                document_id="doc_1",
                content="Content 2",
                source="knowledge.txt",
                retrieval_rank=1,
            ),
            RagRetrievedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Content 1",
                source="knowledge.txt",
                retrieval_rank=2,
            ),
        ]


CHUNK_STORE = {
    "chunk_1": Chunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        content="Content 1",
    ),
    "chunk_2": Chunk(
        chunk_id="chunk_2",
        document_id="doc_1",
        content="Content 2",
    ),
}


def build_retriever(
    embedder,
    dense_retriever,
    hydrator,
) -> RAGRetriever:
    return RAGRetriever(
        embedder=embedder,
        dense_retriever=dense_retriever,
        bm25_retriever=BM25Retriever(),
        fusion=RRFFusion(),
        rerank_document_builder=RerankDocumentBuilder(),
        reranker=RerankerAdapter(PreservingReranker()),
        hydrator=hydrator,
    )


# ==================================================
# Case 1：完整编排流程
# ==================================================

def test_retriever_orchestration():

    embedder = FakeEmbedder()
    dense_retriever = FakeDenseRetriever()
    hydrator = FakeHydrator()

    retriever = build_retriever(
        embedder,
        dense_retriever,
        hydrator,
    )

    results = retriever.retrieve(
        query="What is RAG?",
        chunk_store=CHUNK_STORE,
        top_k=2,
    )

    # Query -> Embedder
    assert embedder.received_query == "What is RAG?"

    # Query Vector -> DenseRetriever Contract
    np.testing.assert_allclose(
        dense_retriever.received_query_vector,
        [1.0, 0.0, 0.0],
    )

    # top_k 正确传递
    assert dense_retriever.received_top_k == 2

    # DenseRetrieverResult[] -> RetrievedCandidate[] -> Hydrator
    assert [
        item.chunk_id
        for item in hydrator.received_candidates
    ] == [
        "chunk_2",
        "chunk_1",
    ]

    # dense_rank 必须被适配成候选的业务排名
    assert [
        item.rank
        for item in hydrator.received_candidates
    ] == [
        1,
        2,
    ]

    # 最终 RagRetrievedChunk[]
    assert [
        item.chunk_id
        for item in results
    ] == [
        "chunk_2",
        "chunk_1",
    ]


# ==================================================
# Case 2：Empty Retrieval
# ==================================================

class EmptyDenseRetriever:

    def search(
        self,
        query_vector,
        top_k=5,
    ):
        return []


def test_empty_retrieval():

    hydrator = FakeHydrator()

    retriever = build_retriever(
        FakeEmbedder(),
        EmptyDenseRetriever(),
        hydrator,
    )

    results = retriever.retrieve(
        query="nothing",
        chunk_store={},
        top_k=5,
    )

    assert hydrator.received_candidates == []
    assert results == []


# ==================================================
# Case 3：Dense Retrieval Failure 必须传播
# ==================================================

class FailingDenseRetriever:

    def search(
        self,
        query_vector,
        top_k=5,
    ):
        raise RuntimeError(
            "simulated dense retrieval failure"
        )


class ForbiddenHydrator:

    def hydrate(
        self,
        candidates,
    ):
        raise AssertionError(
            "Hydrator must not run after search failure"
        )


def test_search_failure_propagates():

    retriever = build_retriever(
        FakeEmbedder(),
        FailingDenseRetriever(),
        ForbiddenHydrator(),
    )

    with pytest.raises(
        RuntimeError,
        match="simulated dense retrieval failure",
    ):
        retriever.retrieve(
            query="test",
            chunk_store=CHUNK_STORE,
            top_k=5,
        )
