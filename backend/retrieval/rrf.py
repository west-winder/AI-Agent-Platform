from dataclasses import dataclass


@dataclass
class RRFResult:
    index: int
    score: float


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    rank_constant: int = 60,
    top_n: int | None = None,
) -> list[RRFResult]:
    """
    使用 Reciprocal Rank Fusion (RRF)
    融合多组 ranking。

    参数：
        rankings:
            多组排序结果。

            例如：
            [
                [0, 1, 2],
                [1, 2, 0],
            ]

            第一组可以来自 Dense，
            第二组可以来自 BM25。

            每个数字表示原始 Corpus 中的 index。

        rank_constant:
            RRF 公式中的常数。
            默认使用 60。

        top_n:
            最终最多返回多少个结果。
            None 表示全部返回。

    返回：
        按 RRF score 从高到低排列的 RRFResult。
    """

    if rank_constant <= 0:
        raise ValueError(
            "rank_constant 必须大于 0"
        )

    if top_n is not None and top_n <= 0:
        raise ValueError(
            "top_n 必须大于 0"
        )

    scores: dict[int, float] = {}

    for ranking in rankings:
        seen_indexes: set[int] = set()

        for rank, index in enumerate(
            ranking,
            start=1
        ):
            if not isinstance(index, int):
                raise TypeError(
                    "ranking 中的 index 必须是 int"
                )

            if index < 0:
                raise ValueError(
                    "ranking 中的 index 不能小于 0"
                )

            if index in seen_indexes:
                raise ValueError(
                    f"同一组 ranking 中出现重复 index: {index}"
                )

            seen_indexes.add(index)

            rrf_score = 1.0 / (
                rank_constant + rank
            )

            scores[index] = (
                scores.get(index, 0.0)
                + rrf_score
            )

    results = [
        RRFResult(
            index=index,
            score=score
        )
        for index, score in scores.items()
    ]

    results.sort(
        key=lambda item: item.score,
        reverse=True
    )

    if top_n is not None:
        results = results[:top_n]

    return results