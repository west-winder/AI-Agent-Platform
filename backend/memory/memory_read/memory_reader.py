from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from backend.models.memory import Memory

from backend.memory.memory_read.memory_repository import (
    get_memories_for_read
)

from backend.memory.memory_read.memory_query_scope_judge import (
    MemoryQueryScopeJudge
)

from backend.memory.memory_read.memory_query_scope import (
    MemoryQueryScopeDecision
)

from backend.memory.memory_read.dense_retriever import (
    DenseMemoryRetriever,
    DenseRetrievedMemory
)

from backend.retrieval.bm25 import (
    BM25Retriever,
    BM25Result
)

from backend.retrieval.rrf import (
    reciprocal_rank_fusion,
    RRFResult
)

from backend.memory.memory_read.memory_relevance_judge import (
    MemoryRelevanceCandidate,
    MemoryRelevanceJudge,
    JudgeDecision
)

from backend.reranking.reranker import (
    CrossEncoderReranker,
    RerankResult
)

from backend.memory.memory_read.memory_injector import (
    MemoryInjectionItem,
    MemoryInjector
)


# ==================================================
# Runtime Data Structures
# ==================================================


@dataclass
class HybridMemoryCandidate:
    """
    一条经过 Hybrid Retrieval 后的 Memory Candidate。

    index:
        Memory 在原始 memories 列表中的位置。

    memory:
        原始 Memory ORM 对象。

    dense_score:
        Dense Retrieval similarity。

        None 表示：
        该 Memory 没有进入 Dense Top-N。

    bm25_score:
        BM25 Retrieval score。

        None 表示：
        该 Memory 没有进入 BM25 Top-N。

    rrf_score:
        RRF 融合后的最终 Retrieval score。
    """

    index: int

    memory: Memory

    dense_score: float | None

    bm25_score: float | None

    rrf_score: float


@dataclass
class MemoryReadItem:
    """
    单条 Memory 在一次 Memory Read Pipeline
    中的完整运行时状态。

    保存：

    Dense Retrieval
    +
    BM25 Retrieval
    +
    RRF
    +
    Reranker
    +
    Judge

    的完整 Runtime Trace。
    """

    memory: Memory

    dense_score: float | None

    bm25_score: float | None

    rrf_score: float

    rerank_score: float

    judge_selected: bool | None = None

    judge_reason: str | None = None


@dataclass
class MemoryReadResult:
    """
    一次完整 Memory Read 的运行结果。

    items:
        按照 Reranker 最终排序保存。

    memory_context:
        Judge 最终选中的 Memory，
        经过 MemoryInjector 后生成的上下文。

    scope_decision:
        本次 Memory Read 的 Retrieval Scope
        判断结果。

        保存：

        current / historical / both
        +
        reason
        +
        source

        包括 Early Return 场景。
    """

    query: str

    items: list[MemoryReadItem]

    memory_context: str = ""

    scope_decision: MemoryQueryScopeDecision | None = None


# ==================================================
# Memory Reader
# ==================================================


class MemoryReader:
    """
    Memory Read Pipeline Orchestrator。

    负责组织：

    Query
        ↓
    Memory Query Scope Judge
        ↓
    current / historical / both
        ↓
    Repository
        ↓
    Memory[]
        ↓
    ┌─────────────────┐
    ↓                 ↓
    Dense           BM25
    Retrieval       Retrieval
    ↓                 ↓
    Dense Ranking   BM25 Ranking
    └────────┬────────┘
             ↓
            RRF
             ↓
    Hybrid Candidate Pool
             ↓
         Reranker
             ↓
          Top-K
             ↓
        LLM Judge
             ↓
      MemoryInjector
             ↓
      MemoryReadResult

    本模块本身不实现底层算法。
    """

    def __init__(
        self,
        repository_callable: Callable | None = None,
        dense_retriever=None,
        bm25_retriever=None,
        rrf_callable: Callable | None = None,
        reranker=None,
        judge=None,
        injector=None,
        scope_judge=None
    ):
        """
        初始化 MemoryReader。

        所有核心组件均支持依赖注入，
        方便 Controlled Test 使用 Fake Component。

        注意：

        scope_judge 放在参数列表最后，
        避免破坏旧的 positional argument 调用。
        """

        if scope_judge is None:
            scope_judge = (
                MemoryQueryScopeJudge()
            )

        if repository_callable is None:
            repository_callable = (
                get_memories_for_read
            )

        if dense_retriever is None:
            dense_retriever = (
                DenseMemoryRetriever()
            )

        if bm25_retriever is None:
            bm25_retriever = (
                BM25Retriever()
            )

        if rrf_callable is None:
            rrf_callable = (
                reciprocal_rank_fusion
            )

        if judge is None:
            judge = (
                MemoryRelevanceJudge()
            )

        if injector is None:
            injector = (
                MemoryInjector()
            )

        self._repository_callable = (
            repository_callable
        )

        self._scope_judge = (
            scope_judge
        )

        self._dense_retriever = (
            dense_retriever
        )

        self._bm25_retriever = (
            bm25_retriever
        )

        self._rrf_callable = (
            rrf_callable
        )

        self._reranker = (
            reranker
        )

        self._judge = (
            judge
        )

        self._injector = (
            injector
        )

    # ==================================================
    # Component Initialization
    # ==================================================

    def _ensure_reranker(self):
        """
        确保 Reranker 已初始化。

        Cross Encoder 模型加载成本较高，
        所以继续采用 Lazy Initialization。
        """

        if self._reranker is None:
            self._reranker = (
                CrossEncoderReranker()
            )

    # ==================================================
    # Input Validation
    # ==================================================

    def _validate_input(
        self,
        user_id: int,
        query: str,
        top_n: int,
        top_k: int
    ):
        """
        校验 Memory Read Pipeline 输入参数。
        """

        # --------------------------------------------------
        # user_id
        # --------------------------------------------------

        if type(user_id) is not int:
            raise TypeError(
                "user_id 必须是 int 类型"
            )

        if user_id <= 0:
            raise ValueError(
                "user_id 必须大于 0"
            )

        # --------------------------------------------------
        # query
        # --------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "query 必须是 str 类型"
            )

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        # --------------------------------------------------
        # top_n
        # --------------------------------------------------

        if type(top_n) is not int:
            raise TypeError(
                "top_n 必须是 int 类型"
            )

        if top_n <= 0:
            raise ValueError(
                "top_n 必须大于 0"
            )

        # --------------------------------------------------
        # top_k
        # --------------------------------------------------

        if type(top_k) is not int:
            raise TypeError(
                "top_k 必须是 int 类型"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        # --------------------------------------------------
        # Pipeline Configuration
        # --------------------------------------------------

        if top_k > top_n:
            raise ValueError(
                "top_k 不能大于 top_n"
            )

    # ==================================================
    # Hybrid Candidate Mapping
    # ==================================================

    def _build_hybrid_candidates(
        self,
        memories: list[Memory],
        dense_results: list[DenseRetrievedMemory],
        bm25_results: list[BM25Result],
        rrf_results: list[RRFResult]
    ) -> list[HybridMemoryCandidate]:
        """
        将 Dense / BM25 / RRF 的结果
        汇总成统一 HybridMemoryCandidate。

        index 坐标系：

        DenseRetrievedMemory.index
        BM25Result.index
        RRFResult.index

        都必须指向：

            原始 memories[]

        即：

            memories[index]
        """

        # --------------------------------------------------
        # Dense：
        #
        # 原始 Memory Index
        #       ↓
        # Dense Similarity
        # --------------------------------------------------

        dense_scores = {
            item.index: item.similarity
            for item in dense_results
        }

        # --------------------------------------------------
        # BM25：
        #
        # 原始 Memory Index
        #       ↓
        # BM25 Score
        # --------------------------------------------------

        bm25_scores = {
            item.index: item.score
            for item in bm25_results
        }

        hybrid_candidates = []

        seen_indexes = set()

        memory_count = len(
            memories
        )

        # --------------------------------------------------
        # RRF 已经给出了最终 Hybrid Ranking。
        #
        # 所以这里按照 rrf_results 的顺序
        # 构建 Hybrid Candidate。
        # --------------------------------------------------

        for rrf_result in rrf_results:

            index = rrf_result.index

            # ------------------------------------------
            # index 类型
            # ------------------------------------------

            if type(index) is not int:
                raise TypeError(
                    "RRFResult.index "
                    "必须是 int"
                )

            # ------------------------------------------
            # index 范围
            # ------------------------------------------

            if not (
                0 <= index < memory_count
            ):
                raise ValueError(
                    "RRFResult.index "
                    f"超出 memories 范围：{index}"
                )

            # ------------------------------------------
            # index 重复
            # ------------------------------------------

            if index in seen_indexes:
                raise ValueError(
                    "RRFResult.index "
                    f"重复：{index}"
                )

            seen_indexes.add(
                index
            )

            # ------------------------------------------
            # 根据原始 index 找回 Memory
            # ------------------------------------------

            memory = memories[index]

            # ------------------------------------------
            # 构造 Hybrid Candidate
            # ------------------------------------------

            candidate = (
                HybridMemoryCandidate(
                    index=index,
                    memory=memory,
                    dense_score=(
                        dense_scores.get(index)
                    ),
                    bm25_score=(
                        bm25_scores.get(index)
                    ),
                    rrf_score=(
                        rrf_result.score
                    )
                )
            )

            hybrid_candidates.append(
                candidate
            )

        return hybrid_candidates

    # ==================================================
    # Rerank Mapping
    # ==================================================

    def _build_memory_read_items(
        self,
        hybrid_candidates: list[
            HybridMemoryCandidate
        ],
        rerank_results: list[RerankResult]
    ) -> list[MemoryReadItem]:
        """
        根据 Reranker 的 index，
        将通用 RerankResult
        映射回 HybridMemoryCandidate。

        注意：

        RerankResult.index

        指向的是：

            hybrid_candidates[]

        而不是：

            原始 memories[]

        这是一个新的局部 index 坐标系。
        """

        if (
            len(rerank_results)
            != len(hybrid_candidates)
        ):
            raise ValueError(
                "Reranker 返回结果数量"
                "与 Hybrid Candidate 数量不一致"
            )

        items = []

        seen_indexes = set()

        candidate_count = len(
            hybrid_candidates
        )

        for rerank_result in rerank_results:

            index = rerank_result.index

            # ------------------------------------------
            # index 类型
            # ------------------------------------------

            if type(index) is not int:
                raise TypeError(
                    "RerankResult.index "
                    "必须是 int"
                )

            # ------------------------------------------
            # index 范围
            # ------------------------------------------

            if not (
                0 <= index < candidate_count
            ):
                raise ValueError(
                    "RerankResult.index "
                    f"超出范围：{index}"
                )

            # ------------------------------------------
            # index 重复
            # ------------------------------------------

            if index in seen_indexes:
                raise ValueError(
                    "RerankResult.index "
                    f"重复：{index}"
                )

            seen_indexes.add(
                index
            )

            # ------------------------------------------
            # 根据 Reranker 的局部 index
            # 找回 Hybrid Candidate
            # ------------------------------------------

            hybrid_candidate = (
                hybrid_candidates[index]
            )

            # ------------------------------------------
            # 构造 Runtime Trace
            # ------------------------------------------

            item = MemoryReadItem(
                memory=(
                    hybrid_candidate.memory
                ),
                dense_score=(
                    hybrid_candidate.dense_score
                ),
                bm25_score=(
                    hybrid_candidate.bm25_score
                ),
                rrf_score=(
                    hybrid_candidate.rrf_score
                ),
                rerank_score=(
                    rerank_result.score
                )
            )

            items.append(
                item
            )

        # ----------------------------------------------
        # 确保 Reranker 完整覆盖输入 Candidate
        # ----------------------------------------------

        expected_indexes = set(
            range(candidate_count)
        )

        if seen_indexes != expected_indexes:
            raise ValueError(
                "Reranker 没有完整覆盖"
                "所有 Hybrid Candidate"
            )

        return items

    # ==================================================
    # Judge Mapping
    # ==================================================

    def _apply_judge_decisions(
        self,
        judge_items: list[MemoryReadItem],
        decisions: list[JudgeDecision]
    ):
        """
        根据 JudgeDecision.index，
        将 Judge 判断结果映射回
        Top-K MemoryReadItem。

        JudgeDecision.index 指向：

            judge_items[]
        """

        if len(decisions) != len(
            judge_items
        ):
            raise ValueError(
                "Judge Decision 数量"
                "与 Top-K Candidate 数量不一致"
            )

        seen_indexes = set()

        candidate_count = len(
            judge_items
        )

        for decision in decisions:

            index = decision.index

            # ------------------------------------------
            # index 类型
            # ------------------------------------------

            if type(index) is not int:
                raise TypeError(
                    "JudgeDecision.index "
                    "必须是 int"
                )

            # ------------------------------------------
            # index 范围
            # ------------------------------------------

            if not (
                0 <= index < candidate_count
            ):
                raise ValueError(
                    "JudgeDecision.index "
                    f"超出范围：{index}"
                )

            # ------------------------------------------
            # index 重复
            # ------------------------------------------

            if index in seen_indexes:
                raise ValueError(
                    "JudgeDecision.index "
                    f"重复：{index}"
                )

            seen_indexes.add(
                index
            )

            # ------------------------------------------
            # 找到对应 Top-K MemoryReadItem
            # ------------------------------------------

            item = judge_items[index]

            # ------------------------------------------
            # 写入 Judge Runtime State
            # ------------------------------------------

            item.judge_selected = (
                decision.selected
            )

            item.judge_reason = (
                decision.reason
            )

        # ----------------------------------------------
        # 完整覆盖检查
        # ----------------------------------------------

        expected_indexes = set(
            range(candidate_count)
        )

        if seen_indexes != expected_indexes:
            raise ValueError(
                "Judge 没有完整覆盖"
                "所有 Top-K Candidate"
            )

    # ==================================================
    # Public API
    # ==================================================

    def read(
        self,
        db: Session,
        user_id: int,
        query: str,
        top_n: int = 20,
        top_k: int = 5
    ) -> MemoryReadResult:
        """
        执行一次完整的 Memory Read V2 Pipeline。

        流程：

        Query
            ↓
        Query Scope Judge
            ↓
        current / historical / both
            ↓
        Memory Repository
            ↓
        Memory[]
            ↓
        ┌──────────────────────┐
        ↓                      ↓
        Dense Retrieval     BM25 Retrieval
        ↓                      ↓
        Dense Top-N         BM25 Top-N
        ↓                      ↓
        Dense Ranking       BM25 Ranking
        └──────────┬───────────┘
                   ↓
                  RRF
                   ↓
        HybridMemoryCandidate[]
                   ↓
                 Text[]
                   ↓
               Reranker
                   ↓
             MemoryReadItem[]
                   ↓
                 Top-K
                   ↓
                Judge
                   ↓
               Injector
                   ↓
          MemoryReadResult
        """

        # --------------------------------------------------
        # 1. 参数校验
        # --------------------------------------------------

        self._validate_input(
            user_id=user_id,
            query=query,
            top_n=top_n,
            top_k=top_k
        )

        # --------------------------------------------------
        # 2. Query Scope Judge
        #
        # Query
        #   ↓
        # current / historical / both
        #
        # MemoryReader 不解释 scope，
        # 只把 decision.scope 传给 Repository。
        # --------------------------------------------------

        scope_decision = (
            self._scope_judge.judge(
                query
            )
        )

        if not isinstance(
            scope_decision,
            MemoryQueryScopeDecision
        ):
            raise TypeError(
                "Memory Query Scope Judge "
                "必须返回 MemoryQueryScopeDecision"
            )

        # --------------------------------------------------
        # 3. Repository
        #
        # user_id
        # +
        # scope
        #   ↓
        # Retrieval Corpus
        # --------------------------------------------------

        memories = self._repository_callable(
            db,
            user_id,
            scope=scope_decision.scope
        )

        if not isinstance(memories, list):
            raise TypeError(
                "Memory Repository "
                "必须返回 list"
            )

        # --------------------------------------------------
        # 4. 用户没有任何 Memory
        # --------------------------------------------------

        if not memories:
            return MemoryReadResult(
                query=query,
                items=[],
                scope_decision=scope_decision
            )


        effective_top_n = min(
            top_n,
            len(memories)
        )

        # --------------------------------------------------
        # 5. Dense Retrieval
        #
        # Memory[]
        #    ↓
        # Dense Top-N
        # --------------------------------------------------

        dense_results = (
            self._dense_retriever.search(
                query=query,
                memories=memories,
                top_n=effective_top_n
            )
        )

        # --------------------------------------------------
        # 6. Memory ORM
        #      ↓
        # Generic Text[]
        #
        # BM25Retriever 不依赖 Memory。
        # --------------------------------------------------

        memory_texts = [
            memory.content
            for memory in memories
        ]

        # --------------------------------------------------
        # 7. BM25 Index
        #
        # 当前 Memory 数据规模很小，
        # 每次 Read 临时建立 BM25 Index。
        # --------------------------------------------------

        self._bm25_retriever.index(
            memory_texts
        )

        # --------------------------------------------------
        # 8. BM25 Retrieval
        # --------------------------------------------------

        bm25_results = (
            self._bm25_retriever.search(
                query=query,
                top_n=effective_top_n
            )
        )

        # --------------------------------------------------
        # 9. Dense / BM25
        #        ↓
        # 原始 memories[] index ranking
        # --------------------------------------------------

        dense_ranking = [
            item.index
            for item in dense_results
        ]

        bm25_ranking = [
            item.index
            for item in bm25_results
            if item.score > 0
        ]

        # --------------------------------------------------
        # 10. RRF
        #
        # RRF 不关心：
        #
        # Dense similarity
        # BM25 score
        #
        # 只关心两个 Retriever 的 ranking。
        # --------------------------------------------------

        rrf_results = (
            self._rrf_callable(
                rankings=[
                    dense_ranking,
                    bm25_ranking
                ],
                top_n=effective_top_n
            )
        )

        if not rrf_results:
            return MemoryReadResult(
                query=query,
                items=[],
                scope_decision=scope_decision
            )

        # --------------------------------------------------
        # 11. 构建 Hybrid Candidate
        #
        # Dense Score
        # BM25 Score
        # RRF Score
        # Memory ORM
        #
        # 汇总到统一 Runtime Candidate。
        # --------------------------------------------------

        hybrid_candidates = (
            self._build_hybrid_candidates(
                memories=memories,
                dense_results=dense_results,
                bm25_results=bm25_results,
                rrf_results=rrf_results
            )
        )

        if not hybrid_candidates:
            return MemoryReadResult(
                query=query,
                items=[],
                scope_decision=scope_decision
            )

        # --------------------------------------------------
        # 12. Hybrid Business Object
        #          ↓
        #      Generic Text[]
        #
        # Reranker 只认识文本，
        # 不认识 Memory / BM25 / RRF。
        # --------------------------------------------------

        texts = [
            item.memory.content
            for item in hybrid_candidates
        ]

        # --------------------------------------------------
        # 13. Reranker
        # --------------------------------------------------

        self._ensure_reranker()

        rerank_results = (
            self._reranker.rerank(
                query=query,
                texts=texts
            )
        )

        # --------------------------------------------------
        # 14. RerankResult.index
        #          ↓
        # HybridMemoryCandidate[]
        #          ↓
        # MemoryReadItem[]
        #
        # items 按 Reranker 排名保存。
        # --------------------------------------------------

        items = (
            self._build_memory_read_items(
                hybrid_candidates=(
                    hybrid_candidates
                ),
                rerank_results=(
                    rerank_results
                )
            )
        )

        # --------------------------------------------------
        # 15. Top-K
        # --------------------------------------------------

        judge_items = items[:top_k]

        # --------------------------------------------------
        # 16. MemoryReadItem[]
        #          ↓
        # MemoryRelevanceCandidate[]
        #
        # MemoryReader 只做 Data Adaptation。
        #
        # 它不解释 memory_status，
        # 只把 content + memory_status
        # 转换为 Judge 的输入 Contract。
        # --------------------------------------------------

        judge_candidates = [
            MemoryRelevanceCandidate(
                content=(
                    item.memory.content
                ),
                memory_status=(
                    item.memory.memory_status
                )
            )
            for item in judge_items
        ]

        # --------------------------------------------------
        # 17. Memory Relevance Judge
        #
        # Query
        # +
        # MemoryRelevanceCandidate[]
        #   ↓
        # JudgeDecision[]
        # --------------------------------------------------

        decisions = (
            self._judge.judge(
                query=query,
                candidates=(
                    judge_candidates
                )
            )
        )

        # --------------------------------------------------
        # 18. JudgeDecision.index
        #          ↓
        # Top-K MemoryReadItem
        # --------------------------------------------------

        self._apply_judge_decisions(
            judge_items=judge_items,
            decisions=decisions
        )

        # --------------------------------------------------
        # 19. Selected Memory
        # --------------------------------------------------

        selected_injection_items = [
            MemoryInjectionItem(
                content=item.memory.content,
                memory_status=(
                    item.memory.memory_status
                )
            )
            for item in items
            if item.judge_selected is True
        ]

        # --------------------------------------------------
        # 20. Memory Injection
        # --------------------------------------------------

        memory_context = (
            self._injector.build_context(
                selected_injection_items
            )
        )

        # --------------------------------------------------
        # 21. 返回完整 Memory Read Result
        # --------------------------------------------------

        return MemoryReadResult(
            query=query,
            items=items,
            memory_context=memory_context,
            scope_decision=scope_decision
        )