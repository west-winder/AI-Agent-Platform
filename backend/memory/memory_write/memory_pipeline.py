from sqlalchemy.orm import Session

from backend.memory.memory_write.memory_extractor import MemoryExtractor
from backend.memory.memory_write.memory_validator import MemoryValidator
from backend.memory.memory_write.memory_deduplicator import check_duplicate
from backend.memory.memory_write.memory_similarity import (
    MemorySimilarity,
    MemorySimilarityError,
)
from backend.memory.memory_write.memory_relationship_judge import (
    MemoryRelationshipJudge,
    RelationshipContractError,
    validate_relationship_coverage,
)

from backend.memory.memory_write.memory_lifecycle import (
    decide_memory_lifecycle,
)

from backend.schemas.memory_lifecycle import (
    MemoryLifecycleDecision,
)

from backend.services.memory_service import (
    add_memory,
    get_current_memories,
    mark_memories_historical,
)

from backend.schemas.memory import MemoryCreate


class MemoryPipeline:
    """
    Memory Write Pipeline。

    完整流程：

    User Message
        ↓
    Memory Extraction
        ↓
    Candidates
        ↓
    Memory Validation
        ↓
        ├── invalid → 丢弃
        ↓
    Exact Deduplication（只针对 current）
        ↓
        ├── duplicate → 跳过
        ↓
    Embedding
        ↓
    Similarity Search（只针对 current）
        ↓
        ├── Retrieval Failure → Fail Closed → 跳过
        ↓
    Top-K
        ↓
    Memory Relationship Judge
        ↓
        ├── Judge Contract Failure → Fail Closed → 跳过
        ↓
    Relationship
        ↓
    Lifecycle Decision（纯确定性）
        ↓
    单个事务：
        historical_memory_ids → historical
        +
        save_candidate → insert current
        ↓
    ONE commit（失败 rollback）
        ↓
    commit 成功后才同步 Runtime current corpus

    重要：

    save_candidate 与 historical_memory_ids
    是两个独立维度。

    save_candidate = False
    只表示 Candidate 不需要 Insert，
    不表示 historical_memory_ids 可以被忽略。

    四种分支：

    Case A：save=False, historical=[]  → no-op
    Case B：save=False, historical=[..] → mark + ONE commit
    Case C：save=True,  historical=[..] → mark + insert + ONE commit
    Case D：save=True,  historical=[]  → insert + ONE commit

    当前版本暂不处理：

    1. Memory Update
    2. Memory Delete
    3. Memory Versioning
    4. superseded_by / parent_id
    5. Merge
    6. Historical Retrieval

    Memory Lifecycle V1 只处理：

    current
    historical

    注意：

    当前 SQLite 不保存 Embedding。

    Embedding 只在 Pipeline 运行过程中使用。

    Embedding 的具体实现由 MemorySimilarity 负责，
    Pipeline 本身不直接操作 Embedding Vector。
    """

    def __init__(
        self,
        extractor=None,
        validator=None,
        similarity=None,
        relationship_judge=None,
        top_k: int = 5,
        threshold: float = 0.70,
    ):
        """
        初始化 Memory Pipeline。

        支持依赖注入，方便：

        1. Pipeline 单元测试
        2. Mock Embedding
        3. Mock Similarity Search
        4. Mock LLM Judge
        """

        # ==================================================
        # 参数校验
        # ==================================================

        if top_k <= 0:
            raise ValueError(
                "top_k必须大于0"
            )

        if not 0 <= threshold <= 1:
            raise ValueError(
                "threshold必须位于0到1之间"
            )

        self.top_k = top_k
        self.threshold = threshold

        # ==================================================
        # Memory Extraction
        # ==================================================

        self.extractor = (
            extractor
            if extractor is not None
            else MemoryExtractor()
        )

        # ==================================================
        # Memory Validation
        # ==================================================

        self.validator = (
            validator
            if validator is not None
            else MemoryValidator()
        )

        # ==================================================
        # Memory Similarity
        # ==================================================

        self.similarity = (
            similarity
            if similarity is not None
            else MemorySimilarity()
        )

        # ==================================================
        # Memory Relationship Judge
        # ==================================================

        self.relationship_judge = (
            relationship_judge
            if relationship_judge is not None
            else MemoryRelationshipJudge()
        )

    # ==================================================
    # Pipeline
    # ==================================================

    async def process(
        self,
        db: Session,
        user_id: int,
        user_message: str,
    ):
        """
        执行完整 Memory Write Pipeline。

        返回：

            本次最终保存的 Memory 列表。
        """

        # ==================================================
        # Step 1：Memory Extraction
        # ==================================================

        candidates = await self.extractor.extract(
            user_message
        )

        if not candidates:
            return []

        # ==================================================
        # Step 2：获取 Write Corpus
        #
        # Memory Write V1/V2 的：
        #
        # Exact Dedup
        # Related Retrieval
        # Relationship Judge
        #
        # 一律只针对 current Memory。
        #
        # historical Memory 不参与
        # 普通 Write Lifecycle 判断。
        # ==================================================

        current_memories = get_current_memories(
            db,
            user_id,
        )

        # ==================================================
        # Step 3：逐个处理 Candidate
        # ==================================================

        saved_memories = []

        for candidate in candidates:

            # ==================================================
            # 3.1 Validation
            # ==================================================

            validation_result = await self.validator.validate(
                candidate
            )

            if not validation_result.valid:
                continue

            # ==================================================
            # 3.2 Exact Deduplication
            # ==================================================

            duplicate_result = check_duplicate(
                candidate,
                current_memories,
            )

            if duplicate_result.duplicate:
                continue

            # ==================================================
            # 3.3 Embedding + Similarity Search
            #
            # 必须区分：
            #
            # 检索成功但没有 match
            #     → matches=[]
            #     → 继续
            #
            # Retrieval 本身失败
            #     → MemorySimilarityError
            #     → Fail Closed
            #     → 本 Candidate 不写数据库
            # ==================================================

            try:

                similarity_result = (
                    self.similarity.search(
                        candidate=candidate,
                        memories=current_memories,
                        top_k=self.top_k,
                        threshold=self.threshold,
                    )
                )

            except MemorySimilarityError as e:

                print(
                    "[Memory Lifecycle] "
                    "Related Retrieval failed, "
                    "candidate skipped: "
                    f"{e}"
                )

                continue

            similar_memories = (
                similarity_result.matches
            )

            # ==================================================
            # 3.4 Relationship Judge
            # ==================================================

            # --------------------------------------------------
            # 没有相似 Memory：
            #
            # 不需要 Relationship Judge。
            #
            # Candidate 直接进入 Lifecycle Decision。
            # --------------------------------------------------

            if not similar_memories:

                decision = MemoryLifecycleDecision(
                    save_candidate=True,
                    historical_memory_ids=[],
                )

            # --------------------------------------------------
            # 存在相似 Memory：
            #
            # 调用 Relationship Judge。
            # --------------------------------------------------

            else:

                relationship_result = await self.relationship_judge.judge(
                    candidate,
                    similar_memories,
                )

                # ----------------------------------------------
                # Judge Contract 校验
                #
                # 所有送入 Judge 的 Existing Memory
                # 必须得到且只得到一个 Relationship。
                #
                # 任何一项不满足：
                #
                # Fail Closed
                # 本 Candidate 不写数据库
                # ----------------------------------------------

                try:

                    validate_relationship_coverage(
                        similar_memories,
                        relationship_result,
                    )

                except RelationshipContractError as e:

                    print(
                        "[Memory Lifecycle] "
                        "Judge Contract failure, "
                        "candidate skipped: "
                        f"{e}"
                    )

                    continue

                # ==========================================
                # 3.5 Lifecycle Decision
                #
                # 纯确定性组件。
                #
                # 不调用 LLM
                # 不访问数据库
                # 不执行 persistence
                # ==========================================

                decision = decide_memory_lifecycle(
                    relationship_result
                )

            # ==================================================
            # 3.6 Lifecycle Action
            #
            # save_candidate 与 historical_memory_ids
            # 是两个独立维度：
            #
            # duplicate → save_candidate = False
            #     只表示 Candidate 不需要 Insert
            #
            # conflict → historical_memory_ids
            #     独立决定对应 current Memory 转 historical
            #
            # 因此不能因为
            # save_candidate = False
            # 就忽略 historical_memory_ids。
            #
            # 四种分支：
            #
            # Case A：save=False, historical=[]  → no-op
            # Case B：save=False, historical=[..] → mark + commit
            # Case C：save=True,  historical=[..] → mark + insert + commit
            # Case D：save=True,  historical=[]  → insert + commit
            # ==================================================

            # --------------------------------------------------
            # Case A：纯 no-op
            #
            # 不产生任何数据库 mutation，
            # 也不需要开事务。
            # --------------------------------------------------

            if (
                not decision.save_candidate
                and not decision.historical_memory_ids
            ):
                continue

            # ==================================================
            # 3.7 预计算 Runtime corpus（commit 之前）
            #
            # 目的：
            #
            # 让 commit 之后的 corpus 同步
            # 成为真正的纯内存操作。
            #
            # SQLAlchemy Session 默认：
            #
            # expire_on_commit = True
            #
            # 因此 commit 之后访问
            #
            # m.id
            # m.content
            # m.memory_status
            #
            # 都可能触发 lazy reload（数据库 IO）。
            #
            # 不要假设 commit 之后
            # ORM attribute access 一定是纯内存操作。
            #
            # 所以这里在
            # DB mutation / commit 之前
            # 就把下一步的 corpus 预计算好。
            #
            # 注意：
            #
            # 这里只预计算，
            # 不修改 current_memories。
            #
            # current_memories 的替换
            # 只发生在 commit 成功之后。
            #
            # 本 Candidate 之前的
            # Exact Dedup / Similarity
            # 已经访问过这些 ORM 对象，
            # 因此这里的 m.id 读取
            # 发生在对象仍在内存中时。
            # ==================================================

            historical_ids = set(
                decision.historical_memory_ids
            )

            next_current_memories = [
                current_memory
                for current_memory
                in current_memories
                if current_memory.id
                not in historical_ids
            ]

            # ==================================================
            # 3.8 Persistence（单个事务）
            #
            # 一个 Candidate 的全部 Lifecycle Mutation
            # 属于一个原子事务：
            #
            # historical_memory_ids → historical
            # +
            # Candidate → insert current
            #
            # 然后 ONE commit。
            #
            # 任何一步失败 → rollback。
            #
            # 不允许出现：
            #
            # 旧 Memory 已 historical
            # 但新 Candidate 保存失败
            #
            # --------------------------------------------------
            # Transaction Boundary（重要）
            # --------------------------------------------------
            #
            # try / except 只覆盖
            # "仍然可以 rollback" 的阶段：
            #
            # DB mutation
            #     ↓
            # commit
            #
            # db.commit() 一旦成功返回，
            # 这次事务已经正式提交。
            #
            # 此后不得再进入
            # 声称可以撤销该事务的 rollback 分支。
            #
            # 因此：
            #
            # db.refresh()
            # Runtime corpus 同步
            #
            # 都必须放在 try / except 之后。
            #
            # 否则会产生语义错位：
            #
            # DB 已成功 commit
            # 但程序把这一轮当成 rollback failure
            # ==================================================

            memory = None

            try:

                # ----------------------------------------------
                # conflict → historical
                #
                # 这里只需要它的 mutation 副作用。
                #
                # 后续 Runtime corpus 同步直接使用
                # decision.historical_memory_ids，
                # 不依赖返回值，
                # 因此不需要 refresh。
                # ----------------------------------------------

                mark_memories_historical(
                    db=db,
                    user_id=user_id,
                    memory_ids=(
                        decision.historical_memory_ids
                    ),
                )

                # ----------------------------------------------
                # Candidate → insert current
                # ----------------------------------------------

                if decision.save_candidate:

                    memory_data = MemoryCreate(
                        content=candidate.content,
                        memory_type=candidate.memory_type,
                    )

                    memory = add_memory(
                        db=db,
                        user_id=user_id,
                        memory_data=memory_data,
                    )

                # ----------------------------------------------
                # ONE commit
                #
                # 这是 try 块的最后一句。
                #
                # 它成功返回之后，
                # 事务不可再回滚。
                # ----------------------------------------------

                db.commit()

            except Exception as e:

                db.rollback()

                print(
                    "[Memory Lifecycle] "
                    "Lifecycle write failed, "
                    "transaction rolled back: "
                    f"{e}"
                )

                # ----------------------------------------------
                # Rollback Contract：
                #
                # 1. current_memories 保持不变
                #
                # 2. next_current_memories 被丢弃，
                #    不应用
                #
                # 这里绝对不能提前修改
                # current_memories。
                # --------------------------------------------------

                continue

            # ==================================================
            # 3.9 应用 Runtime corpus（commit 成功之后）
            #
            # 这一整段位于 try / except 之后。
            #
            # 从这里开始：
            #
            # 1. 事务已提交，不可回滚
            # 2. 不再进入 rollback 分支
            #
            # 顺序：
            #
            # 预计算 next_current_memories
            #     ↓
            # DB mutation
            #     ↓
            # commit 成功
            #     ↓
            # 应用 next_current_memories
            #
            # 这一段是纯内存操作：
            #
            # 不读取任何 ORM 数据库字段，
            # 不触发 lazy reload。
            #
            # 规则：
            #
            # 1. 移除被置为 historical 的 Memory
            #
            # 2. save_candidate = True
            #    → append 新创建的 current Memory
            #
            # 3. save_candidate = False
            #    → 不 append
            #
            # 原因：
            #
            # 同一次 extract 可能返回多个 Candidate。
            #
            # 如果已经被转为 historical 的 Memory
            # 仍然留在 Runtime corpus 里，
            # 后续 Candidate 的
            # Exact Dedup / Similarity / Judge
            # 会错误地把它当作 current。
            # ==================================================

            current_memories = (
                next_current_memories
            )

            if memory is not None:

                saved_memories.append(
                    memory
                )

                current_memories.append(
                    memory
                )

        # ==================================================
        # Pipeline 完成
        # ==================================================

        return saved_memories