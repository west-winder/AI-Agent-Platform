import pytest

from backend.rag.contracts import (
    RerankDocument,
    RerankedResult,
)
from backend.rag.retrieval.reranker_adapter import (
    RerankerAdapter,
)


# ==================================================
# Fake：模拟 backend.reranking.reranker 的对外契约
# ==================================================

class FakeRerankItem:
    """
    模拟 backend.reranking.reranker.RerankResult 的字段契约：
    index / score。
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score


class FakeCrossEncoder:
    """
    模拟 CrossEncoderReranker.rerank(query, texts)：

    返回 item(index, score)，
    其中 index 是文本在 texts 中的位置。
    """

    def __init__(self, score_by_content):
        self._score_by_content = score_by_content
        self.received_query = None
        self.received_texts = None

    def rerank(self, query, texts):
        self.received_query = query
        self.received_texts = list(texts)

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


class OutOfRangeReranker:
    """
    返回越界 index，用于验证 Adapter 的防御边界。
    """

    def rerank(self, query, texts):
        return [FakeRerankItem(index=99, score=1.0)]


def build_documents(*pairs):
    return [
        RerankDocument(
            chunk_id=chunk_id,
            content=content,
        )
        for chunk_id, content in pairs
    ]


# ==================================================
# 1. rerank_rank 正确性
# ==================================================

def test_rerank_rank_is_one_based_and_consecutive():
    adapter = RerankerAdapter(
        FakeCrossEncoder({
            "A": 0.1,
            "B": 0.9,
            "C": 0.5,
        })
    )

    results = adapter.rerank(
        "query",
        build_documents(
            ("c1", "A"),
            ("c2", "B"),
            ("c3", "C"),
        ),
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        "c2",
        "c3",
        "c1",
    ]

    assert [
        result.rerank_rank
        for result in results
    ] == [1, 2, 3]


def test_rerank_scores_are_ordered_descending():
    adapter = RerankerAdapter(
        FakeCrossEncoder({
            "A": 0.1,
            "B": 0.9,
            "C": 0.5,
        })
    )

    results = adapter.rerank(
        "query",
        build_documents(
            ("c1", "A"),
            ("c2", "B"),
            ("c3", "C"),
        ),
    )

    scores = [
        result.rerank_score
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
# 2. previous_rank：入参顺序
# ==================================================

def test_previous_rank_reflects_input_order():
    """
    previous_rank 是 RerankDocument 的入参顺序（1-based），
    刻意不叫 fusion_rank：
    Reranker 不假设前一个组件一定是 Fusion。
    """

    adapter = RerankerAdapter(
        FakeCrossEncoder({
            "A": 0.1,
            "B": 0.9,
            "C": 0.5,
        })
    )

    results = adapter.rerank(
        "query",
        build_documents(
            ("c1", "A"),
            ("c2", "B"),
            ("c3", "C"),
        ),
    )

    previous_rank_by_chunk = {
        result.chunk_id: result.previous_rank
        for result in results
    }

    assert previous_rank_by_chunk == {
        "c1": 1,
        "c2": 2,
        "c3": 3,
    }


# ==================================================
# 3. 契约
# ==================================================

def test_returns_reranked_result_contract():
    adapter = RerankerAdapter(
        FakeCrossEncoder({"A": 0.5})
    )

    results = adapter.rerank(
        "query",
        build_documents(("c1", "A")),
    )

    assert all(
        isinstance(result, RerankedResult)
        for result in results
    )


def test_passes_query_and_texts_to_underlying_reranker():
    """
    Reranker 只接收 query + content，
    不接触 chunk_store / document store。
    """

    underlying = FakeCrossEncoder({
        "A": 0.5,
        "B": 0.1,
    })

    RerankerAdapter(underlying).rerank(
        "my query",
        build_documents(
            ("c1", "A"),
            ("c2", "B"),
        ),
    )

    assert underlying.received_query == "my query"
    assert underlying.received_texts == ["A", "B"]


# ==================================================
# 4. 边界情况
# ==================================================

def test_empty_documents_returns_empty_list():
    adapter = RerankerAdapter(
        FakeCrossEncoder({})
    )

    assert adapter.rerank("query", []) == []


def test_raises_when_underlying_index_is_out_of_range():
    adapter = RerankerAdapter(
        OutOfRangeReranker()
    )

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        adapter.rerank(
            "query",
            build_documents(("c1", "A")),
        )


# ==================================================
# 5. 契约校验：异常 Reranker 实现不得污染 Pipeline
# ==================================================

class DuplicateIndexReranker:
    """
    同一个 index 返回两次。
    index 必须唯一，否则身份映射会串位。
    """

    def rerank(self, query, texts):
        return [
            FakeRerankItem(index=0, score=0.9),
            FakeRerankItem(index=0, score=0.1),
        ]


class ConstantScoreReranker:
    """
    返回指定的 score，用于验证 finite / 数值校验。
    """

    def __init__(self, score):
        self._score = score

    def rerank(self, query, texts):
        return [
            FakeRerankItem(
                index=0,
                score=self._score,
            )
        ]


def test_raises_when_index_is_duplicated():
    adapter = RerankerAdapter(
        DuplicateIndexReranker()
    )

    with pytest.raises(
        ValueError,
        match="duplicate index",
    ):
        adapter.rerank(
            "query",
            build_documents(
                ("c1", "A"),
                ("c2", "B"),
            ),
        )


@pytest.mark.parametrize(
    "score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_raises_when_score_is_not_finite(score):
    adapter = RerankerAdapter(
        ConstantScoreReranker(score)
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        adapter.rerank(
            "query",
            build_documents(("c1", "A")),
        )


@pytest.mark.parametrize(
    "score",
    [
        None,
        "1.0",
        [1.0],
    ],
)
def test_raises_when_score_is_not_numeric(score):
    adapter = RerankerAdapter(
        ConstantScoreReranker(score)
    )

    with pytest.raises(
        ValueError,
        match="non-numeric",
    ):
        adapter.rerank(
            "query",
            build_documents(("c1", "A")),
        )


def test_accepts_numpy_numeric_score():
    """
    真实 Cross Encoder 的输出可能是 numpy 标量，
    校验不能把合法数值误判为非数值。
    """

    import numpy as np

    adapter = RerankerAdapter(
        ConstantScoreReranker(np.float32(0.7))
    )

    results = adapter.rerank(
        "query",
        build_documents(("c1", "A")),
    )

    assert results[0].rerank_score == pytest.approx(
        0.7,
        abs=1e-6,
    )




class CountMismatchReranker:
    """模拟底层 Reranker 返回数量不符合输入数量。"""

    def __init__(self, returned_count):
        self.returned_count = returned_count

    def rerank(self, query, texts):
        return [
            FakeRerankItem(
                index=index,
                score=float(self.returned_count - index),
            )
            for index in range(self.returned_count)
        ]


@pytest.mark.parametrize("returned_count", [0, 1, 3])
def test_raises_when_result_count_mismatches(
    returned_count,
):
    adapter = RerankerAdapter(
        CountMismatchReranker(returned_count)
    )

    documents = build_documents(
        ("chunk_1", "Content A"),
        ("chunk_2", "Content B"),
    )

    with pytest.raises(
        ValueError,
        match="reranker result count mismatch",
    ):
        adapter.rerank("query", documents)
