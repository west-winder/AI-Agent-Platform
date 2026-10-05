import numpy as np
import pytest

from backend.rag.contracts import SearchResult
from backend.rag.retrieval.exact_search import ExactDenseSearch


def test_search_returns_top_k_results():
    """
    正常路径：
    ExactDenseSearch 应该只返回相关度最高的 Top-K。
    """

    searcher = ExactDenseSearch()

    query_vector = np.array([1.0, 0.0])

    chunk_vectors = {
        "chunk_1": np.array([1.0, 0.0]),
        "chunk_2": np.array([0.0, 1.0]),
        "chunk_3": np.array([0.8, 0.2]),
    }

    results = searcher.search(
        query_vector=query_vector,
        chunk_vectors=chunk_vectors,
        top_k=2,
    )

    assert len(results) == 2

    assert all(
        isinstance(result, SearchResult)
        for result in results
    )

    assert results[0].chunk_id == "chunk_1"
    assert results[1].chunk_id == "chunk_3"


def test_search_empty_corpus_returns_empty_list():
    """
    Empty Corpus 是合法业务状态，不是 Failure。
    """

    searcher = ExactDenseSearch()

    query_vector = np.array([1.0, 0.0])

    results = searcher.search(
        query_vector=query_vector,
        chunk_vectors={},
        top_k=5,
    )

    assert results == []


def test_search_sorts_results_by_score_descending():
    """
    SearchResult 必须按照 score 从高到低排列。
    """

    searcher = ExactDenseSearch()

    query_vector = np.array([1.0, 0.0])

    chunk_vectors = {
        "chunk_low": np.array([0.0, 1.0]),
        "chunk_high": np.array([1.0, 0.0]),
        "chunk_middle": np.array([0.8, 0.2]),
    }

    results = searcher.search(
        query_vector=query_vector,
        chunk_vectors=chunk_vectors,
        top_k=3,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        "chunk_high",
        "chunk_middle",
        "chunk_low",
    ]

    assert (
        results[0].score
        >= results[1].score
        >= results[2].score
    )


def test_search_raises_when_vector_dimensions_do_not_match():
    """
    Query Vector 与 Chunk Vector 维度不同，
    属于数据 / Embedding Contract Failure，
    不能静默跳过。
    """

    searcher = ExactDenseSearch()

    query_vector = np.array(
        [1.0, 0.0, 0.0]
    )

    chunk_vectors = {
        "chunk_1": np.array(
            [1.0, 0.0]
        )
    }

    with pytest.raises(ValueError):
        searcher.search(
            query_vector=query_vector,
            chunk_vectors=chunk_vectors,
            top_k=1,
        )


def test_search_raises_when_vector_is_zero_vector():
    """
    Zero Vector 无法定义 Cosine Similarity。

    这是 Invalid Vector，
    不是 similarity = 0。
    """

    searcher = ExactDenseSearch()

    query_vector = np.array(
        [1.0, 0.0]
    )

    chunk_vectors = {
        "chunk_1": np.array(
            [0.0, 0.0]
        )
    }

    with pytest.raises(ValueError):
        searcher.search(
            query_vector=query_vector,
            chunk_vectors=chunk_vectors,
            top_k=1,
        )