import numpy as np

from backend.rag.contracts import (
    Chunk,
    Document,
    RagRetrievedChunk,
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
from backend.rag.retrieval.retriever import RAGRetriever


# ==================================================
# Fake
# ==================================================

class FakeEmbedder:
    """确定性 Query Vector。"""

    def embed(self, query):
        return np.array([1.0, 0.0])


class FakeRerankItem:
    """
    模拟 backend.reranking.reranker.RerankResult 的字段契约。
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score


class ContentScoringReranker:
    """
    按 content -> score 表打分。

    与 Fusion 的顺序刻意不同（相当于一次重排），
    用于验证 Fusion → Rerank 之间身份没有串位。
    """

    def __init__(self, score_by_content):
        self._score_by_content = score_by_content

    def rerank(self, query, texts):
        items = [
            FakeRerankItem(
                index=index,
                score=self._score_by_content[text],
            )
            for index, text in enumerate(texts)
        ]

        items.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return items


# ==================================================
# Corpus
# ==================================================

CHUNK_STORE = {
    "chunk_dense": Chunk(
        chunk_id="chunk_dense",
        document_id="doc_a",
        content="alpha beta gamma",
    ),
    "chunk_other": Chunk(
        chunk_id="chunk_other",
        document_id="doc_a",
        content="delta epsilon",
    ),
    "chunk_sparse": Chunk(
        chunk_id="chunk_sparse",
        document_id="doc_b",
        content="unique_keyword_xyz",
    ),
}

DOCUMENT_STORE = {
    "doc_a": Document(
        document_id="doc_a",
        file_name="dense.md",
        file_type="markdown",
    ),
    "doc_b": Document(
        document_id="doc_b",
        file_name="sparse.md",
        file_type="markdown",
    ),
}

# chunk_sparse 刻意没有向量：
# Dense 在物理上不可能召回它，只能靠 BM25。
CHUNK_VECTORS = {
    "chunk_dense": np.array([1.0, 0.0]),
    "chunk_other": np.array([0.6, 0.8]),
}

RERANK_SCORES = {
    "alpha beta gamma": 0.5,
    "delta epsilon": 0.2,
    "unique_keyword_xyz": 0.9,
}


def build_retriever() -> RAGRetriever:
    return RAGRetriever(
        embedder=FakeEmbedder(),
        exact_search=ExactDenseSearch(),
        bm25_retriever=BM25Retriever(),
        fusion=RRFFusion(),
        rerank_document_builder=RerankDocumentBuilder(),
        reranker=RerankerAdapter(
            ContentScoringReranker(RERANK_SCORES)
        ),
        hydrator=RAGHydrator(),
    )


# ==================================================
# Hybrid E2E
# ==================================================

def test_bm25_only_recall_survives_until_hydration():
    """
    chunk_sparse 没有任何向量：

    Dense 召不到它，只有 BM25 能召回。

    验证完整链路：

    BM25 单独召回
        ↓
    Fusion 保留
        ↓
    Rerank 把它推到第 1（身份不能串位）
        ↓
    Hydration 正确恢复 document_id / content / source
    """

    results = build_retriever().retrieve(
        query="unique_keyword_xyz",
        chunk_vectors=CHUNK_VECTORS,
        chunk_store=CHUNK_STORE,
        document_store=DOCUMENT_STORE,
        top_k=3,
    )

    # BM25 单独召回的 Chunk 必须存在
    assert "chunk_sparse" in [
        result.chunk_id
        for result in results
    ]

    # Fusion + Rerank 之后顺序与身份正确
    assert [
        result.chunk_id
        for result in results
    ] == [
        "chunk_sparse",
        "chunk_dense",
        "chunk_other",
    ]

    assert [
        result.retrieval_rank
        for result in results
    ] == [1, 2, 3]

    # Hydration 正确
    assert all(
        isinstance(result, RagRetrievedChunk)
        for result in results
    )

    sparse = results[0]

    assert sparse.document_id == "doc_b"
    assert sparse.content == "unique_keyword_xyz"
    assert sparse.source == "sparse.md"

    assert results[1].source == "dense.md"


def test_top_k_truncates_after_reranking():
    """
    top_k 在 Rerank 之后裁剪：
    先完成 Fusion + Rerank，再取前 top_k 条。
    """

    results = build_retriever().retrieve(
        query="unique_keyword_xyz",
        chunk_vectors=CHUNK_VECTORS,
        chunk_store=CHUNK_STORE,
        document_store=DOCUMENT_STORE,
        top_k=1,
    )

    assert [
        result.chunk_id
        for result in results
    ] == ["chunk_sparse"]

    assert results[0].retrieval_rank == 1


def test_empty_corpus_returns_empty_list():
    results = build_retriever().retrieve(
        query="unique_keyword_xyz",
        chunk_vectors={},
        chunk_store={},
        document_store={},
        top_k=5,
    )

    assert results == []
