import numpy as np
import pytest

from backend.rag.contracts import (
    Chunk,
    Document,
)
from backend.rag.retrieval.bm25_retriever import BM25Retriever
from backend.rag.retrieval.exact_search import ExactDenseSearch
from backend.rag.retrieval.fusion import RRFFusion
from backend.rag.retrieval.rerank_document_builder import (
    RerankDocumentBuilder,
)
from backend.rag.retrieval.reranker_adapter import (
    RerankerAdapter,
)
from backend.rag.retrieval.retriever import RAGRetriever

from inmemory_repositories import build_inmemory_hydrator


class FakeEmbedder:
    """
    测试专用 Embedder。

    不加载真实模型，
    固定返回一个确定的 Query Vector。
    """

    def embed(self, query: str):
        return np.array([1.0, 0.0])


class FailingEmbedder:
    """
    用于验证 Embedding Failure 是否正常向上传播。
    """

    def embed(self, query: str):
        raise RuntimeError("embedding failed")


class FakeRerankItem:
    """
    模拟 backend.reranking.reranker.RerankResult 的字段契约。
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score


class PreservingReranker:
    """
    测试专用底层 Reranker：保持输入顺序。

    真实 Cross Encoder Reranker 需要本地模型文件，
    因此编排测试注入 Fake，
    让测试只关注 Pipeline 结构。
    """

    def __init__(self):
        self.received_query = None
        self.received_texts = None

    def rerank(self, query, texts):
        self.received_query = query
        self.received_texts = list(texts)

        return [
            FakeRerankItem(
                index=index,
                score=1.0 / (index + 1),
            )
            for index, text in enumerate(texts)
        ]


def build_retriever(
    embedder,
    chunk_vectors,
    chunk_store,
    document_store,
) -> RAGRetriever:
    """
    Dense 路径注入 DenseRetriever Contract 的内存实现。

    Hydrator 只依赖 Repository Contract：

    测试注入 InMemory Repository，
    生产注入 PostgreSQL Repository。
    """

    return RAGRetriever(
        embedder=embedder,
        dense_retriever=ExactDenseSearch(
            chunk_vectors
        ),
        bm25_retriever=BM25Retriever(),
        fusion=RRFFusion(),
        rerank_document_builder=RerankDocumentBuilder(),
        reranker=RerankerAdapter(PreservingReranker()),
        hydrator=build_inmemory_hydrator(
            chunk_store,
            document_store,
        ),
    )


def test_retrieve_runs_complete_retrieval_pipeline():
    """
    Query
        ↓
    Embedding
        ↓
    Dense Retrieval + BM25 Retrieval
        ↓
    Fusion(RRF)
        ↓
    Rerank
        ↓
    Hydration
        ↓
    RagRetrievedChunk[]
    """

    chunk_store = {
        "chunk_1": Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Low relevance content",
        ),
        "chunk_2": Chunk(
            chunk_id="chunk_2",
            document_id="doc_2",
            content="High relevance content",
        ),
    }

    document_store = {
        "doc_1": Document(
            document_id="doc_1",
            file_name="a.md",
            file_type="markdown",
        ),
        "doc_2": Document(
            document_id="doc_2",
            file_name="b.pdf",
            file_type="pdf",
        ),
    }

    chunk_vectors = {
        "chunk_1": np.array([0.0, 1.0]),
        "chunk_2": np.array([1.0, 0.0]),
    }

    retriever = build_retriever(
        FakeEmbedder(),
        chunk_vectors,
        chunk_store,
        document_store,
    )

    results = retriever.retrieve(
        query="test query",
        chunk_store=chunk_store,
        top_k=1,
    )

    assert len(results) == 1

    assert results[0].chunk_id == "chunk_2"
    assert results[0].content == "High relevance content"
    assert results[0].source == "b.pdf"


def test_retrieve_empty_corpus_returns_empty_list():
    """
    Empty Corpus 最终应该正常得到 Empty Retrieval。
    """

    retriever = build_retriever(
        FakeEmbedder(),
        {},
        {},
        {},
    )

    results = retriever.retrieve(
        query="test query",
        chunk_store={},
        top_k=5,
    )

    assert results == []


def test_retrieve_propagates_embedding_failure():
    """
    RAGRetriever 不吞掉 Embedding Failure。
    """

    retriever = build_retriever(
        FailingEmbedder(),
        {},
        {},
        {},
    )

    with pytest.raises(
        RuntimeError,
        match="embedding failed",
    ):
        retriever.retrieve(
            query="test query",
            chunk_store={},
            top_k=5,
        )


def test_retrieve_propagates_consistency_failure():
    """
    候选指向不存在的 Chunk：

    属于 Retrieval / Storage 数据一致性错误，
    必须继续向上传播。

    Pipeline 重排后，该错误在进入 Hydration 之前
    就会在 RerankDocumentBuilder 边界暴露。
    """

    chunk_vectors = {
        "missing_chunk": np.array([1.0, 0.0]),
    }

    retriever = build_retriever(
        FakeEmbedder(),
        chunk_vectors,
        {},
        {},
    )

    with pytest.raises(KeyError):
        retriever.retrieve(
            query="test query",
            chunk_store={},
            top_k=1,
        )
