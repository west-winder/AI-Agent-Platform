import pytest

from backend.rag.contracts import (
    BM25RetrieverResult,
    DenseRetrieverResult,
    FusedRetrieverResult,
)
from backend.rag.retrieval.fusion import RRFFusion


def dense(chunk_id, rank, score=None):
    return DenseRetrieverResult(
        chunk_id=chunk_id,
        dense_score=(
            score if score is not None else 1.0 / rank
        ),
        dense_rank=rank,
    )


def bm25(chunk_id, rank, score=None):
    return BM25RetrieverResult(
        chunk_id=chunk_id,
        bm25_score=(
            score if score is not None else 1.0 / rank
        ),
        bm25_rank=rank,
    )


# ==================================================
# 1. 融合基本行为
# ==================================================

def test_chunk_hit_by_both_retrievers_ranks_first():
    """
    B 同时被 Dense 与 BM25 命中：
    即使它在 Dense 里只排第 2，
    RRF 也应该把它推到第 1。
    """

    fused = RRFFusion().fuse(
        dense_results=[
            dense("A", 1),
            dense("B", 2),
        ],
        bm25_results=[
            bm25("B", 1),
            bm25("D", 2),
        ],
    )

    assert [
        result.chunk_id
        for result in fused
    ] == [
        "B",
        "A",
        "D",
    ]


def test_returns_fused_retriever_result_contract():
    fused = RRFFusion().fuse(
        dense_results=[dense("A", 1)],
        bm25_results=[bm25("B", 1)],
    )

    assert all(
        isinstance(result, FusedRetrieverResult)
        for result in fused
    )


# ==================================================
# 2. 单边命中：缺失字段必须为 None
# ==================================================

def test_dense_only_chunk_has_no_bm25_fields():
    fused = RRFFusion().fuse(
        dense_results=[dense("A", 1, score=0.9)],
        bm25_results=[],
    )

    assert fused[0].chunk_id == "A"
    assert fused[0].dense_score == 0.9
    assert fused[0].dense_rank == 1

    assert fused[0].bm25_score is None
    assert fused[0].bm25_rank is None


def test_bm25_only_chunk_has_no_dense_fields():
    fused = RRFFusion().fuse(
        dense_results=[],
        bm25_results=[bm25("D", 1, score=7.5)],
    )

    assert fused[0].chunk_id == "D"
    assert fused[0].bm25_score == 7.5
    assert fused[0].bm25_rank == 1

    assert fused[0].dense_score is None
    assert fused[0].dense_rank is None


def test_does_not_expose_content():
    fused = RRFFusion().fuse(
        dense_results=[dense("A", 1)],
        bm25_results=[bm25("A", 1)],
    )

    assert not hasattr(fused[0], "content")


# ==================================================
# 3. Rank 正确性
# ==================================================

def test_rrf_rank_is_one_based_and_consecutive():
    fused = RRFFusion().fuse(
        dense_results=[
            dense("A", 1),
            dense("B", 2),
        ],
        bm25_results=[
            bm25("B", 1),
            bm25("D", 2),
        ],
    )

    assert [
        result.rrf_rank
        for result in fused
    ] == [1, 2, 3]


def test_rrf_scores_are_ordered_descending():
    fused = RRFFusion().fuse(
        dense_results=[
            dense("A", 1),
            dense("B", 2),
        ],
        bm25_results=[
            bm25("B", 1),
            bm25("D", 2),
        ],
    )

    scores = [
        result.rrf_score
        for result in fused
    ]

    assert all(
        previous >= current
        for previous, current in zip(
            scores,
            scores[1:],
        )
    )


# ==================================================
# 4. RRF 只依赖 rank，不依赖 score 量级
# ==================================================

def test_fusion_depends_on_rank_not_score_magnitude():
    """
    X 的 dense_score 比 Y 大 7 个数量级，
    但 Y 在 BM25 里排第 1、X 没有被 BM25 命中，
    因此 Y 必须排在 X 前面。
    """

    fused = RRFFusion().fuse(
        dense_results=[
            dense("X", 1, score=999.0),
            dense("Y", 2, score=0.0001),
        ],
        bm25_results=[
            bm25("Y", 1, score=0.0001),
        ],
    )

    assert fused[0].chunk_id == "Y"
    assert fused[1].chunk_id == "X"


# ==================================================
# 5. 边界情况
# ==================================================

def test_both_empty_returns_empty_list():
    assert RRFFusion().fuse([], []) == []


def test_top_k_limits_result_size():
    fused = RRFFusion().fuse(
        dense_results=[
            dense("A", 1),
            dense("B", 2),
        ],
        bm25_results=[
            bm25("B", 1),
        ],
        top_k=2,
    )

    assert len(fused) == 2

    assert [
        result.rrf_rank
        for result in fused
    ] == [1, 2]


def test_duplicate_chunk_id_within_one_retriever_raises():
    with pytest.raises(
        ValueError,
        match="重复",
    ):
        RRFFusion().fuse(
            dense_results=[
                dense("A", 1),
                dense("A", 2),
            ],
            bm25_results=[],
        )


def test_invalid_rank_raises():
    with pytest.raises(
        ValueError,
        match="必须是正整数",
    ):
        RRFFusion().fuse(
            dense_results=[dense("A", 0, score=0.5)],
            bm25_results=[],
        )


# ==================================================
# 6. rank 必须是连续 1..N
# ==================================================

def test_rank_with_a_gap_raises():
    """
    [1, 3] 不是合法排名：
    不能静默当作 [1, 2] 重新编号。
    """

    with pytest.raises(
        ValueError,
        match="连续的 1..N",
    ):
        RRFFusion().fuse(
            dense_results=[
                dense("A", 1, score=0.9),
                dense("B", 3, score=0.5),
            ],
            bm25_results=[],
        )


def test_rank_not_starting_at_one_raises():
    with pytest.raises(
        ValueError,
        match="连续的 1..N",
    ):
        RRFFusion().fuse(
            dense_results=[
                dense("A", 2, score=0.9),
                dense("B", 3, score=0.5),
            ],
            bm25_results=[],
        )


def test_duplicate_rank_raises():
    with pytest.raises(
        ValueError,
        match="连续的 1..N",
    ):
        RRFFusion().fuse(
            dense_results=[
                dense("A", 1, score=0.9),
                dense("B", 1, score=0.5),
            ],
            bm25_results=[],
        )


def test_bm25_rank_gap_raises():
    with pytest.raises(
        ValueError,
        match="bm25_rank",
    ):
        RRFFusion().fuse(
            dense_results=[],
            bm25_results=[
                bm25("A", 1, score=0.9),
                bm25("B", 3, score=0.5),
            ],
        )
