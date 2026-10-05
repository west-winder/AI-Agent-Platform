import pytest

from backend.rag.contracts import (
    Chunk,
    Document,
    RetrievedChunk,
    SearchResult,
)
from backend.rag.retrieval.hydrator import RAGHydrator


def test_hydrate_returns_complete_retrieved_chunks_in_order():
    """
    正常路径：

    SearchResult
        ↓
    Chunk
        ↓
    Document
        ↓
    RetrievedChunk

    同时必须保持 SearchResult 原有 Ranking 顺序。
    """

    hydrator = RAGHydrator()

    search_results = [
        SearchResult(
            chunk_id="chunk_2",
            score=0.95,
        ),
        SearchResult(
            chunk_id="chunk_1",
            score=0.80,
        ),
    ]

    chunk_store = {
        "chunk_1": Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content A",
        ),
        "chunk_2": Chunk(
            chunk_id="chunk_2",
            document_id="doc_2",
            content="Content B",
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

    results = hydrator.hydrate(
        search_results=search_results,
        chunk_store=chunk_store,
        document_store=document_store,
    )

    assert len(results) == 2

    assert all(
        isinstance(result, RetrievedChunk)
        for result in results
    )

    # Hydrator 不得改变 Retriever 的 Ranking
    assert results[0].chunk_id == "chunk_2"
    assert results[1].chunk_id == "chunk_1"

    # 检查 Hydration 后的数据是否完整
    assert results[0].content == "Content B"
    assert results[0].source == "b.pdf"
    assert results[0].score == 0.95

    assert results[1].content == "Content A"
    assert results[1].source == "a.md"
    assert results[1].score == 0.80


def test_hydrate_empty_results_returns_empty_list():
    """
    Empty Retrieval 是合法业务状态。
    """

    hydrator = RAGHydrator()

    results = hydrator.hydrate(
        search_results=[],
        chunk_store={},
        document_store={},
    )

    assert results == []


def test_hydrate_raises_when_chunk_is_missing():
    """
    SearchResult 指向不存在的 Chunk：

    属于 Retrieval / Storage Consistency Failure，
    不能静默 skip。
    """

    hydrator = RAGHydrator()

    search_results = [
        SearchResult(
            chunk_id="missing_chunk",
            score=0.90,
        )
    ]

    with pytest.raises(KeyError):
        hydrator.hydrate(
            search_results=search_results,
            chunk_store={},
            document_store={},
        )


def test_hydrate_raises_when_document_is_missing():
    """
    Chunk 指向不存在的 Document：

    属于数据一致性错误，
    不能生成不完整 RetrievedChunk。
    """

    hydrator = RAGHydrator()

    search_results = [
        SearchResult(
            chunk_id="chunk_1",
            score=0.90,
        )
    ]

    chunk_store = {
        "chunk_1": Chunk(
            chunk_id="chunk_1",
            document_id="missing_doc",
            content="Some content",
        )
    }

    with pytest.raises(KeyError):
        hydrator.hydrate(
            search_results=search_results,
            chunk_store=chunk_store,
            document_store={},
        )