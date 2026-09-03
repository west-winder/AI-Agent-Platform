from sqlalchemy.orm import Session

from backend.memory.memory_write.memory_extractor import MemoryExtractor
from backend.memory.memory_write.memory_validator import MemoryValidator
from backend.memory.memory_write.memory_deduplicator import check_duplicate
from backend.memory.memory_write.memory_similarity import MemorySimilarity
from backend.memory.memory_write.memory_relationship_judge import (
    MemoryRelationshipJudge,
)

from backend.services.memory_service import (
    create_memory,
    get_memories,
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
    Exact Deduplication
        ↓
        ├── duplicate → 跳过
        ↓
    Embedding
        ↓
    Similarity Search
        ↓
    Top-K
        ↓
    Memory Relationship Judge
        ↓
    Relationship
        ↓
    Final Decision
        ↓
    SQLite Persistence

    当前版本暂不处理：

    1. Conflict Resolution
    2. Memory Update
    3. Memory Delete
    4. Memory Versioning

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

    def process(
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

        candidates = self.extractor.extract(
            user_message
        )

        if not candidates:
            return []

        # ==================================================
        # Step 2：获取已有 Memory
        # ==================================================

        memories = get_memories(
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

            validation_result = self.validator.validate(
                candidate
            )

            if not validation_result.valid:
                continue

            # ==================================================
            # 3.2 Exact Deduplication
            # ==================================================

            duplicate_result = check_duplicate(
                candidate,
                memories,
            )

            if duplicate_result.duplicate:
                continue

            # ==================================================
            # 3.3 Embedding + Similarity Search
            # ==================================================

            similarity_result = self.similarity.search(
                candidate=candidate,
                memories=memories,
                top_k=self.top_k,
                threshold=self.threshold,
            )

            similar_memories = (
                similarity_result.matches
            )

            # ==================================================
            # 3.4 Relationship Judge
            # ==================================================

            relationship_result = None

            # --------------------------------------------------
            # 没有相似 Memory：
            #
            # 不需要 Relationship Judge。
            #
            # Candidate 可以直接进入 Final Decision。
            # --------------------------------------------------

            # --------------------------------------------------
            # 存在相似 Memory：
            #
            # 调用 Relationship Judge。
            # --------------------------------------------------

            if similar_memories:

                relationship_result = (
                    self.relationship_judge.judge(
                        candidate,
                        similar_memories,
                    )
                )

            # ==================================================
            # 3.5 Final Decision
            # ==================================================

            should_save = True

            # --------------------------------------------------
            # 没有 Relationship Result：
            #
            # 说明：
            #
            # 1. 没有相似 Memory
            #
            # 此时默认 Candidate 是新的 Memory。
            # --------------------------------------------------

            # --------------------------------------------------
            # 存在 Relationship Result：
            # --------------------------------------------------

            if relationship_result is not None:

                for relationship in (
                    relationship_result.relationships
                ):

                    # ------------------------------------------
                    # Duplicate
                    # ------------------------------------------

                    if (
                        relationship.relationship
                        == "duplicate"
                    ):

                        should_save = False

                        break

                    # ------------------------------------------
                    # Conflict
                    # ------------------------------------------
                    #
                    # 当前阶段：
                    #
                    # Conflict 只识别，
                    # 不处理。
                    #
                    # Candidate 仍然保存。
                    #
                    # ------------------------------------------

                    if (
                        relationship.relationship
                        == "conflict"
                    ):

                        continue

                    # ------------------------------------------
                    # Related
                    # ------------------------------------------

                    if (
                        relationship.relationship
                        == "related"
                    ):

                        continue

                    # ------------------------------------------
                    # New
                    # ------------------------------------------

                    if (
                        relationship.relationship
                        == "new"
                    ):

                        continue

            # ==================================================
            # Duplicate → 不保存
            # ==================================================

            if not should_save:
                continue

            # ==================================================
            # 3.6 Persistence
            # ==================================================

            memory_data = MemoryCreate(
                content=candidate.content,
                memory_type=candidate.memory_type,
            )

            memory = create_memory(
                db=db,
                user_id=user_id,
                memory_data=memory_data,
            )

            saved_memories.append(
                memory
            )

            # ==================================================
            # 3.7 更新当前 Memory 集合
            # ==================================================
            #
            # 后续 Candidate 必须能够看到刚刚创建的 Memory。
            #
            # ==================================================

            memories.append(
                memory
            )

        # ==================================================
        # Pipeline 完成
        # ==================================================

        return saved_memories