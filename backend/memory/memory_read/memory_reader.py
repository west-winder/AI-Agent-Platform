from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from backend.models.memory import Memory

from backend.memory.memory_read.memory_repository import (
    get_memories_for_read
)

from backend.memory.memory_read.retriever import (
    MemoryRetriever,
    RetrievedMemory
)

from backend.memory.memory_read.memory_relevance_judge import (
    MemoryRelevanceJudge,
    JudgeDecision
)

from backend.reranking.reranker import (
    CrossEncoderReranker,
    RerankResult
)

from backend.memory.memory_read.memory_injector import (
    MemoryInjector
)

# ==================================================
# Runtime Data Structures
# ==================================================


@dataclass
class MemoryReadItem:
    """
    单条 Memory 在一次 Memory Read Pipeline
    中的完整运行时状态。

    属性：
        memory:
            原始 Memory ORM 对象。

        retrieval_score:
            Retriever 阶段计算得到的
            Query-Memory 向量相似度。

        rerank_score:
            Reranker 阶段计算得到的
            Query-Memory 相关性分数。

        judge_selected:
            LLM Judge 的最终判断。

            True:
                Judge 判断该 Memory
                应该用于当前 Query。

            False:
                Judge 已经判断，
                但认为该 Memory
                不应该用于当前 Query。

            None:
                该 Memory 没有进入 Top-K，
                因此没有进入 Judge。

        judge_reason:
            Judge 的判断原因。

            如果没有进入 Judge，
            则保持 None。
    """

    memory: Memory

    retrieval_score: float

    rerank_score: float

    judge_selected: bool | None = None

    judge_reason: str | None = None


@dataclass
class MemoryReadResult:
    """
    一次完整 Memory Read 的运行结果。

    当前 V1 只保存：

    Query
    +
    本次进入 Candidate Pipeline 的完整 Trace。

    items 按照 Reranker 最终排序保存。
    """

    query: str

    items: list[MemoryReadItem]

    memory_context: str = ""


# ==================================================
# Memory Reader
# ==================================================


class MemoryReader:
    """
    Memory Read V1 Pipeline Orchestrator。

    负责组织：

    Repository
        ↓
    Retriever
        ↓
    Top-N
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

    不负责：

    1. SQL 查询细节
    2. Embedding
    3. Cosine Similarity
    4. Cross Encoder 推理
    5. Judge Prompt
    6. LLM API 调用细节
    7. Memory Context 格式化细节
    8. Chat Integration
    """

    def __init__(
        self,
        repository_callable: Callable | None = None,
        retriever=None,
        reranker=None,
        judge=None,
        injector=None
    ):
        """
        初始化 MemoryReader。

        所有组件都支持依赖注入，
        方便单元测试时使用 Fake Component。

        repository_callable:
            默认使用：
            get_memories_for_read

        retriever:
            默认使用：
            MemoryRetriever

        reranker:
            默认使用：
            CrossEncoderReranker

            Reranker 模型较大，
            当前采用 Lazy Initialization，
            第一次真正执行 read() 时再加载。

        judge:
            默认使用：
            MemoryRelevanceJudge

        injector:
            默认使用：
            MemoryInjector
        """

        if repository_callable is None:
            repository_callable = (
                get_memories_for_read
            )

        if retriever is None:
            retriever = MemoryRetriever()

        if judge is None:
            judge = MemoryRelevanceJudge()

        if injector is None:
            injector = MemoryInjector()

        self._repository_callable = (
            repository_callable
        )

        self._retriever = retriever

        self._reranker = reranker

        self._judge = judge

        self._injector = injector

    # ==================================================
    # Component Initialization
    # ==================================================

    def _ensure_reranker(self):
        """
        确保 Reranker 已初始化。

        Cross Encoder 模型加载成本较高，
        所以当前采用 Lazy Initialization。
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
    # Rerank Mapping
    # ==================================================

    def _build_memory_read_items(
        self,
        retrieved_memories: list[RetrievedMemory],
        rerank_results: list[RerankResult]
    ) -> list[MemoryReadItem]:
        """
        根据 Reranker 的 index，
        将通用 RerankResult
        映射回 Memory 业务对象。

        输入：

        RetrievedMemory[]
            +
        RerankResult[]

        输出：

        MemoryReadItem[]

        返回顺序与 Reranker 排序顺序一致。
        """

        # 当前 Reranker 的契约是：
        #
        # 输入多少条 texts，
        # 就应该返回多少条 RerankResult。
        #
        # 如果数量不一致，
        # 说明组件之间的契约被破坏。

        if (
            len(rerank_results)
            != len(retrieved_memories)
        ):
            raise ValueError(
                "Reranker 返回结果数量"
                "与 Retriever Candidate 数量不一致"
            )

        items = []

        seen_indexes = set()

        candidate_count = len(
            retrieved_memories
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
            # 根据 index 找回 Retriever Candidate
            # ------------------------------------------

            retrieved_memory = (
                retrieved_memories[index]
            )

            # ------------------------------------------
            # 构造统一 Runtime Item
            # ------------------------------------------

            item = MemoryReadItem(
                memory=(
                    retrieved_memory.memory
                ),
                retrieval_score=(
                    retrieved_memory.similarity
                ),
                rerank_score=(
                    rerank_result.score
                )
            )

            items.append(
                item
            )

        # ----------------------------------------------
        # 确保所有 Candidate 均被覆盖
        # ----------------------------------------------

        expected_indexes = set(
            range(candidate_count)
        )

        if seen_indexes != expected_indexes:
            raise ValueError(
                "Reranker 没有完整覆盖"
                "所有 Retriever Candidate"
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

        Judge 的 index 属于：

        judge_items

        这个输入列表自己的坐标系。

        注意：

        judge_items 中保存的是
        MemoryReadItem 对象引用。

        因此修改这里的对象，
        MemoryReadResult.items 中对应对象
        也会同步更新。
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
        执行一次完整的 Memory Read V1 Pipeline。

        流程：

        Query
            ↓
        Memory Repository
            ↓
        Memory[]
            ↓
        Retriever
            ↓
        Top-N RetrievedMemory[]
            ↓
        提取 Memory.content
            ↓
        Reranker
            ↓
        RerankResult[]
            ↓
        index 映射回 RetrievedMemory
            ↓
        MemoryReadItem[]
            ↓
        Top-K
            ↓
        Memory Relevance Judge
            ↓
        JudgeDecision[]
            ↓
        写入 Judge Runtime State
            ↓
        MemoryReadResult

        注意：

        返回的 items 保留完整 Top-N Trace。

        没有进入 Top-K 的 Candidate：

            judge_selected = None
            judge_reason = None

        进入 Top-K 并被 Judge 拒绝：

            judge_selected = False

        进入 Top-K 并被 Judge 采用：

            judge_selected = True
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
        # 2. Repository
        #
        # SQLite
        #   ↓
        # User Memories
        # --------------------------------------------------

        memories = self._repository_callable(
            db,
            user_id
        )

        if not isinstance(memories, list):
            raise TypeError(
                "Memory Repository "
                "必须返回 list"
            )

        # --------------------------------------------------
        # 3. 用户没有任何 Memory
        # --------------------------------------------------

        if not memories:
            return MemoryReadResult(
                query=query,
                items=[]
            )

        # --------------------------------------------------
        # 4. Candidate Retrieval
        #
        # Memory[]
        #   ↓
        # Vector Retrieval
        #   ↓
        # Top-N
        # --------------------------------------------------

        retrieved_memories = (
            self._retriever.search(
                query=query,
                memories=memories,
                top_n=top_n
            )
        )

        # 理论上：
        #
        # memories 非空时 Retriever 应该返回 Candidate。
        #
        # 但这里仍然允许 Retriever 合法返回 []，
        # 方便未来 Retrieval Strategy 演进。

        if not retrieved_memories:
            return MemoryReadResult(
                query=query,
                items=[]
            )

        # --------------------------------------------------
        # 5. Memory Business Object
        #        ↓
        #    Generic Text[]
        #
        # 这是业务层 → 通用基础设施的边界。
        # --------------------------------------------------

        texts = [
            item.memory.content
            for item in retrieved_memories
        ]

        # --------------------------------------------------
        # 6. Reranker
        # --------------------------------------------------

        self._ensure_reranker()

        rerank_results = (
            self._reranker.rerank(
                query=query,
                texts=texts
            )
        )

        # --------------------------------------------------
        # 7. RerankResult.index
        #        ↓
        # RetrievedMemory
        #        ↓
        # MemoryReadItem
        #
        # 同时按照 Reranker 排序顺序
        # 构建完整 Top-N Trace。
        # --------------------------------------------------

        items = (
            self._build_memory_read_items(
                retrieved_memories=(
                    retrieved_memories
                ),
                rerank_results=(
                    rerank_results
                )
            )
        )

        # --------------------------------------------------
        # 8. Top-K
        #
        # 注意：
        # Top-K 是 Memory Read 的业务策略，
        # 不属于通用 Reranker。
        # --------------------------------------------------

        judge_items = items[:top_k]

        # --------------------------------------------------
        # 9. Top-K Memory
        #        ↓
        # Generic Text[]
        # --------------------------------------------------

        judge_texts = [
            item.memory.content
            for item in judge_items
        ]

        # --------------------------------------------------
        # 10. LLM Judge
        # --------------------------------------------------

        decisions = self._judge.judge(
            query=query,
            texts=judge_texts
        )

        # --------------------------------------------------
        # 11. JudgeDecision.index
        #        ↓
        # Top-K MemoryReadItem
        #
        # 写入：
        #
        # judge_selected
        # judge_reason
        # --------------------------------------------------

        self._apply_judge_decisions(
            judge_items=judge_items,
            decisions=decisions
        )

        # --------------------------------------------------
        # 12. Selected Memory
        #
        # 只提取 Judge selected=True 的 Memory。
        # ----------------------------------------------------

        selected_texts = [
            item.memory.content
            for item in items
            if item.judge_selected is True
        ]

        # --------------------------------------------------
        # 13. Memory Injection
        #
        # Selected Memory Texts
        #       ↓
        # Memory Context
        # --------------------------------------------------

        memory_context = (
            self._injector.build_context(
                selected_texts
            )
        )

        # --------------------------------------------------
        # 14. 返回完整 Memory Read Result
        # --------------------------------------------------

        return MemoryReadResult(
            query=query,
            items=items,
            memory_context=memory_context
        )