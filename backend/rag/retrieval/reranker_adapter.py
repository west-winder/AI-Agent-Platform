import math
import numbers

from backend.rag.contracts import (
    RerankDocument,
    RerankedResult,
)


class RerankerAdapter:
    """
    RAG 层 Reranker Adapter。

    职责：

    query + RerankDocument[]
        ↓
    调用通用文本排序能力（backend.reranking.reranker）
        ↓
    RerankedResult[]

    RerankDocument 的入参顺序即 previous_rank（1-based）。

    previous_rank 刻意不叫 fusion_rank：
    Reranker 不假设前一个组件一定是 Fusion。

    本组件不负责：

    1. Retrieval
    2. Fusion
    3. Hydration
    4. 文本存储访问
    5. 模型加载

    底层排序能力通过依赖注入提供，
    因此测试可以注入 Fake Reranker，
    不需要加载真实 Cross Encoder 模型。
    """

    def __init__(
        self,
        reranker,
    ):
        """
        注入底层通用 Reranker。

        reranker 需要提供：

            rerank(
                query: str,
                texts: list[str],
            ) -> list[RerankResult]

        其中 RerankResult 提供 index / score。
        """

        self._reranker = reranker

    def rerank(
        self,
        query: str,
        documents: list[RerankDocument],
    ) -> list[RerankedResult]:
        """
        对 RerankDocument[] 做二次排序。

        返回：
            按 rerank_score 从高到低排列的 RerankedResult
            （含 1-based rerank_rank 与 previous_rank）。
        """

        # --------------------------------------------------
        # 1. 参数校验
        # --------------------------------------------------

        if not isinstance(query, str):
            raise TypeError("query must be str")

        if not query.strip():
            raise ValueError("query must not be blank")

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be list[RerankDocument]"
            )

        # --------------------------------------------------
        # 2. Empty Candidate 是合法状态
        # --------------------------------------------------

        if not documents:
            return []

        # --------------------------------------------------
        # 3. RerankDocument[] -> texts
        # --------------------------------------------------

        texts = [
            document.content
            for document in documents
        ]

        # --------------------------------------------------
        # 4. 调用通用文本排序能力
        # --------------------------------------------------

        raw_results = self._reranker.rerank(
            query,
            texts,
        )

        # --------------------------------------------------
        # 5. 校验底层 Reranker 输出
        # --------------------------------------------------
        #
        # 注入的实现可能是异常实现，
        # 这里做契约校验，避免污染整条 Pipeline。

        self._validate_results(
            raw_results,
            document_count=len(documents),
        )

        # --------------------------------------------------
        # 6. 保证切分契约：score 降序
        # --------------------------------------------------
        #
        # 稳定排序：score 相同时保留底层返回顺序。

        ordered = sorted(
            raw_results,
            key=lambda item: item.score,
            reverse=True,
        )

        # --------------------------------------------------
        # 7. 翻译回 RAG Contract
        # --------------------------------------------------

        return [
            RerankedResult(
                chunk_id=documents[item.index].chunk_id,
                previous_rank=item.index + 1,
                rerank_score=float(item.score),
                rerank_rank=rank,
            )
            for rank, item in enumerate(ordered, start=1)
        ]

    @staticmethod
    def _validate_results(
        raw_results: list,
        *,
        document_count: int,
    ) -> None:
        """
        校验底层 Reranker 的输出契约。

        index：
        - 必须是 int
        - 必须唯一
        - 必须在 [0, document_count) 范围内

        score：
        - 必须是数值
        - 必须 finite（拒绝 NaN / inf）
        """

        # 1. 首先检查结果数量完整性
        if len(raw_results) != document_count:
            raise ValueError(
                "reranker result count mismatch: "
                f"expected {document_count}, "
                f"got {len(raw_results)}"
            )

         # 2. 再逐个检查 index 和 score
        seen_indexes: set[int] = set()

        for item in raw_results:


            index = item.index

            if isinstance(index, bool) or not isinstance(
                index, int
            ):
                raise ValueError(
                    "reranker returned non-int index: "
                    f"{index!r}"
                )

            if not 0 <= index < document_count:
                raise ValueError(
                    "reranker returned out-of-range index: "
                    f"{index}"
                )

            if index in seen_indexes:
                raise ValueError(
                    "reranker returned duplicate index: "
                    f"{index}"
                )

            seen_indexes.add(index)

            raw_score = item.score

            if isinstance(raw_score, bool) or not isinstance(
                raw_score, numbers.Real
            ):
                raise ValueError(
                    "reranker returned non-numeric score: "
                    f"{raw_score!r}"
                )

            if not math.isfinite(float(raw_score)):
                raise ValueError(
                    "reranker returned non-finite score: "
                    f"{raw_score!r}"
                )
