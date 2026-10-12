from backend.rag.contracts import (
    BM25RetrieverResult,
    DenseRetrieverResult,
    FusedRetrieverResult,
)
from backend.retrieval.rrf import reciprocal_rank_fusion


# RRF 的 rank 常数属于 Fusion 内部配置，不进入 Contract。
RRF_RANK_CONSTANT = 60


class RRFFusion:
    """
    Dense + BM25 的 RRF（Reciprocal Rank Fusion）Adapter。

    职责：

    DenseRetrieverResult[]  +  BM25RetrieverResult[]
        ↓
    映射成 chunk_id 排名序列
        ↓
    调用通用 RRF 算法（backend.retrieval.rrf）
        ↓
    翻译回 RAG 契约 FusedRetrieverResult[]

    RRF 只依赖 rank，不依赖 score 归一化。

    本组件不负责：

    1. Retrieval
    2. Hydration
    3. Reranking
    4. Context Assembly
    5. LLM Generation

    它不产出 index-based RRFResult，
    对外只暴露 RAG 自己的 FusedRetrieverResult。
    """

    def fuse(
        self,
        dense_results: list[DenseRetrieverResult],
        bm25_results: list[BM25RetrieverResult],
        top_k: int | None = None,
    ) -> list[FusedRetrieverResult]:
        """
        融合 Dense 与 BM25 的检索结果。

        参数：
            dense_results:
                Dense Retriever 输出。

            bm25_results:
                BM25 Retriever 输出。

            top_k:
                最多返回多少个 FusedRetrieverResult。
                None 表示返回全部融合候选。

        返回：
            按 rrf_score 从高到低排列的 FusedRetrieverResult
            （含 1-based rrf_rank）。
        """

        # --------------------------------------------------
        # 1. 参数校验
        # --------------------------------------------------

        if not isinstance(dense_results, list):
            raise TypeError(
                "dense_results must be "
                "list[DenseRetrieverResult]"
            )

        if not isinstance(bm25_results, list):
            raise TypeError(
                "bm25_results must be "
                "list[BM25RetrieverResult]"
            )

        if top_k is not None:

            if isinstance(top_k, bool) or not isinstance(
                top_k, int
            ):
                raise TypeError("top_k must be int or None")

            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than 0"
                )

        # --------------------------------------------------
        # 2. 两边都为空是合法状态
        # --------------------------------------------------

        if not dense_results and not bm25_results:
            return []

        # --------------------------------------------------
        # 3. 各自按业务排名升序，并校验
        # --------------------------------------------------

        dense_ordered = self._ordered_by_rank(
            dense_results,
            rank_attribute="dense_rank",
            source="dense_results",
        )

        bm25_ordered = self._ordered_by_rank(
            bm25_results,
            rank_attribute="bm25_rank",
            source="bm25_results",
        )

        # --------------------------------------------------
        # 4. chunk_id -> 连续 index
        # --------------------------------------------------
        #
        # 通用 RRF 算法是 index-based，
        # 这里做一次纯映射，不改变任何排序语义。

        indexes: dict[str, int] = {}

        def index_of(chunk_id: str) -> int:

            if chunk_id not in indexes:
                indexes[chunk_id] = len(indexes)

            return indexes[chunk_id]

        rankings = [
            [
                index_of(result.chunk_id)
                for result in dense_ordered
            ],
            [
                index_of(result.chunk_id)
                for result in bm25_ordered
            ],
        ]

        # --------------------------------------------------
        # 5. 调用通用 RRF 算法
        # --------------------------------------------------

        rrf_results = reciprocal_rank_fusion(
            rankings,
            rank_constant=RRF_RANK_CONSTANT,
            top_n=top_k,
        )

        # --------------------------------------------------
        # 6. 翻译回 RAG Contract
        # --------------------------------------------------

        dense_by_id = {
            result.chunk_id: result
            for result in dense_ordered
        }

        bm25_by_id = {
            result.chunk_id: result
            for result in bm25_ordered
        }

        chunk_by_index = {
            index: chunk_id
            for chunk_id, index in indexes.items()
        }

        fused: list[FusedRetrieverResult] = []

        for rank, item in enumerate(rrf_results, start=1):

            chunk_id = chunk_by_index[item.index]

            dense_result = dense_by_id.get(chunk_id)
            bm25_result = bm25_by_id.get(chunk_id)

            fused.append(
                FusedRetrieverResult(
                    chunk_id=chunk_id,
                    dense_score=(
                        dense_result.dense_score
                        if dense_result is not None
                        else None
                    ),
                    dense_rank=(
                        dense_result.dense_rank
                        if dense_result is not None
                        else None
                    ),
                    bm25_score=(
                        bm25_result.bm25_score
                        if bm25_result is not None
                        else None
                    ),
                    bm25_rank=(
                        bm25_result.bm25_rank
                        if bm25_result is not None
                        else None
                    ),
                    rrf_score=item.score,
                    rrf_rank=rank,
                )
            )

        return fused

    @staticmethod
    def _ordered_by_rank(
        results: list,
        *,
        rank_attribute: str,
        source: str,
    ) -> list:
        """
        校验并返回按业务排名升序排列的结果。

        校验：
        - chunk_id 必须是非空 str
        - 同一组内 chunk_id 不允许重复
        - 排名必须是正整数
        - 排名必须连续：1..N

        排名不连续说明上游排名的语义已经破损，
        这里直接失败，不做静默重编号。
        """

        seen: set[str] = set()
        ranks: list[int] = []

        for result in results:

            chunk_id = result.chunk_id

            if (
                not isinstance(chunk_id, str)
                or not chunk_id.strip()
            ):
                raise ValueError(
                    f"{source} 中的 chunk_id "
                    "必须是非空 str"
                )

            if chunk_id in seen:
                raise ValueError(
                    f"{source} 中 chunk_id 重复: {chunk_id}"
                )

            seen.add(chunk_id)

            rank = getattr(result, rank_attribute)

            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank <= 0
            ):
                raise ValueError(
                    f"{source} 中的 {rank_attribute} "
                    "必须是正整数"
                )

            ranks.append(rank)

        expected_ranks = list(
            range(1, len(results) + 1)
        )

        if sorted(ranks) != expected_ranks:
            raise ValueError(
                f"{source} 的 {rank_attribute} "
                "必须是连续的 1..N，"
                f"实际为 {sorted(ranks)}"
            )

        return sorted(
            results,
            key=lambda result: getattr(
                result, rank_attribute
            ),
        )
