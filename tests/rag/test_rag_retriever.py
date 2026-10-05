import numpy as np
import pytest

from backend.rag.contracts import (
    Chunk,
    Document,
)
from backend.rag.retrieval.exact_search import ExactDenseSearch
from backend.rag.retrieval.hydrator import RAGHydrator
from backend.rag.retrieval.retriever import RAGRetriever


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


def test_retrieve_runs_complete_retrieval_pipeline():
    """
    Query
        ↓
    Embedding
        ↓
    Exact Search
        ↓
    Hydration
        ↓
    RetrievedChunk[]
    """

    retriever = RAGRetriever(
        embedder=FakeEmbedder(),
        exact_search=ExactDenseSearch(),
        hydrator=RAGHydrator(),
    )

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

    results = retriever.retrieve(
        query="test query",
        chunk_vectors=chunk_vectors,
        chunk_store=chunk_store,
        document_store=document_store,
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

    retriever = RAGRetriever(
        embedder=FakeEmbedder(),
        exact_search=ExactDenseSearch(),
        hydrator=RAGHydrator(),
    )

    results = retriever.retrieve(
        query="test query",
        chunk_vectors={},
        chunk_store={},
        document_store={},
        top_k=5,
    )

    assert results == []


def test_retrieve_propagates_embedding_failure():
    """
    RAGRetriever 不吞掉 Embedding Failure。
    """

    retriever = RAGRetriever(
        embedder=FailingEmbedder(),
        exact_search=ExactDenseSearch(),
        hydrator=RAGHydrator(),
    )

    with pytest.raises(
        RuntimeError,
        match="embedding failed",
    ):
        retriever.retrieve(
            query="test query",
            chunk_vectors={},
            chunk_store={},
            document_store={},
            top_k=5,
        )


def test_retrieve_propagates_hydration_failure():
    """
    Hydration 的数据一致性错误必须继续向上传播。
    """

    retriever = RAGRetriever(
        embedder=FakeEmbedder(),
        exact_search=ExactDenseSearch(),
        hydrator=RAGHydrator(),
    )

    chunk_vectors = {
        "missing_chunk": np.array([1.0, 0.0]),
    }

    with pytest.raises(KeyError):
        retriever.retrieve(
            query="test query",
            chunk_vectors=chunk_vectors,
            chunk_store={},
            document_store={},
            top_k=1,
        )