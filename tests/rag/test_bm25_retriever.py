from backend.rag.contracts import BM25RetrieverResult
from backend.rag.retrieval.bm25_retriever import BM25Retriever
from backend.rag.retrieval.tokenizer import JiebaTokenizer


CHUNK_TEXTS = {
    "chunk_1": "FastAPI returns validation error",
    "chunk_2": "PostgreSQL vector database",
    "chunk_3": "FastAPI deployment",
}


def search(query: str, top_k: int = 5) -> list[BM25RetrieverResult]:
    return BM25Retriever().search(
        query=query,
        chunk_texts=CHUNK_TEXTS,
        top_k=top_k,
    )


# ==================================================
# 1. Basic Retrieval
# ==================================================

def test_retrieves_most_relevant_chunk_first():
    """
    Query "FastAPI validation"

    chunk_1 同时命中两个词，必须排第一。
    """

    results = search("FastAPI validation")

    assert results[0].chunk_id == "chunk_1"


def test_returns_bm25_retriever_result_contract():
    """
    输出必须是 BM25RetrieverResult，
    而不是任何 Hydration 之后的形态。
    """

    results = search("FastAPI validation")

    assert len(results) > 0

    assert all(
        isinstance(result, BM25RetrieverResult)
        for result in results
    )


def test_does_not_leak_hydration_fields():
    """
    BM25Retriever 是 Retrieval Engine，
    不允许越界产出 Hydration 阶段的字段。
    """

    results = search("FastAPI validation")

    for result in results:
        assert not hasattr(result, "content")
        assert not hasattr(result, "retrieval_rank")
        assert not hasattr(result, "source")


# ==================================================
# 2. Rank Correctness
# ==================================================

def test_first_result_rank_is_one():
    """
    rank 是 1-based 业务排名。
    """

    results = search("FastAPI validation")

    assert results[0].bm25_rank == 1


def test_ranks_are_consecutive():
    """
    rank 必须是连续整数。
    """

    results = search("FastAPI validation")

    assert [
        result.bm25_rank
        for result in results
    ] == list(
        range(1, len(results) + 1)
    )


# ==================================================
# 3. Score Ordering
# ==================================================

def test_scores_are_ordered_descending():
    """
    只验证顺序，不绑定具体浮点值。
    """

    results = search("FastAPI validation")

    scores = [
        result.bm25_score
        for result in results
    ]

    assert all(
        previous >= current
        for previous, current in zip(
            scores,
            scores[1:],
        )
    )


# ==================================================
# Retrieval Engine 行为边界
# ==================================================

def test_respects_top_k():
    results = search("FastAPI validation", top_k=2)

    assert len(results) == 2

    assert [
        result.bm25_rank
        for result in results
    ] == [1, 2]


def test_empty_corpus_returns_empty_list():
    """
    Empty Corpus 是合法业务状态，不是 Failure。
    """

    results = BM25Retriever().search(
        query="FastAPI",
        chunk_texts={},
    )

    assert results == []


def test_no_token_overlap_is_excluded():
    """
    与 Query 没有任何词重合的 Chunk 不是检索结果，
    不允许进入候选池（否则会污染下游 Fusion）。
    """

    results = search("FastAPI validation")

    returned = [
        result.chunk_id
        for result in results
    ]

    assert "chunk_2" not in returned


def test_query_without_any_match_returns_empty_list():
    """
    Query 与整个语料都没有词重合时，
    Empty Retrieval 是合法状态。
    """

    results = search("kubernetes helm")

    assert results == []


def test_jieba_punctuation_does_not_produce_recall():
    """
    纯标点不是有效 token：

    - Query 只有标点时，不应召回任何 Chunk；
    - 某 Chunk 只有标点时，也不应被真实 Query 召回。
    """

    chinese_texts = {
        "chunk_1": "用户 学习 AI",
        "chunk_2": "。，！？——",
    }

    retriever = BM25Retriever(
        tokenizer=JiebaTokenizer()
    )

    assert retriever.search(
        query="。，",
        chunk_texts=chinese_texts,
    ) == []

    results = retriever.search(
        query="用户。，",
        chunk_texts=chinese_texts,
    )

    assert [
        result.chunk_id
        for result in results
    ] == ["chunk_1"]


def test_jieba_tokenizer_supports_chinese():
    """
    BM25Retriever 的 Tokenizer 可替换：
    注入 JiebaTokenizer 后可正确切分中文。
    """

    chinese_texts = {
        "chunk_1": "用户正在学习 AI Agent",
        "chunk_2": "用户偏好深色主题",
        "chunk_3": "数据库使用 PostgreSQL",
    }

    results = BM25Retriever(
        tokenizer=JiebaTokenizer()
    ).search(
        query="用户 学习 Agent",
        chunk_texts=chinese_texts,
    )

    assert results[0].chunk_id == "chunk_1"

    assert all(
        isinstance(result, BM25RetrieverResult)
        for result in results
    )
