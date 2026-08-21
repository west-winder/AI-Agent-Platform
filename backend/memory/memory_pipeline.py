from sqlalchemy.orm import Session

from backend.memory.memory_extractor import MemoryExtractor
from backend.memory.memory_validator import MemoryValidator
from backend.memory.memory_deduplicator import check_duplicate
from backend.memory.memory_similarity import MemorySimilarity
from backend.memory.memory_relationship_judge import (
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
            print(
                "[Memory Pipeline] No candidates extracted."
            )
            return []

        print(
            f"[Memory Pipeline] Extracted {len(candidates)} candidate(s)."
        )

        # ==================================================
        # Step 2：获取已有 Memory
        # ==================================================

        memories = get_memories(
            db,
            user_id,
        )

        print(
            f"[Memory Pipeline] Existing memories: {len(memories)}"
        )

        # ==================================================
        # Step 3：逐个处理 Candidate
        # ==================================================

        saved_memories = []

        for candidate in candidates:

            print("\n========================================")
            print("[Memory Pipeline] Processing Candidate")
            print(f"Content: {candidate.content}")
            print(f"Type: {candidate.memory_type}")
            print("========================================")

            # ==================================================
            # 3.1 Validation
            # ==================================================

            validation_result = self.validator.validate(
                candidate
            )

            print(
                "[Validation]",
                validation_result.valid,
                validation_result.reason,
            )

            if not validation_result.valid:
                print(
                    "[Decision] Invalid candidate -> SKIP"
                )
                continue

            # ==================================================
            # 3.2 Exact Deduplication
            # ==================================================

            duplicate_result = check_duplicate(
                candidate,
                memories,
            )

            print(
                "[Exact Dedup]",
                duplicate_result.duplicate,
                duplicate_result.reason,
            )

            if duplicate_result.duplicate:
                print(
                    "[Decision] Exact duplicate -> SKIP"
                )
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

            print(
                f"[Similarity] "
                f"Found {len(similar_memories)} match(es)"
            )

            if similar_memories:

                for memory in similar_memories:
                    print(
                        f"  - id={memory.memory_id} | "
                        f"similarity={memory.similarity:.6f} | "
                        f"type={memory.memory_type} | "
                        f"content={memory.content}"
                    )

            else:
                print(
                    "[Similarity] No similar memories."
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

            if not similar_memories:

                print(
                    "[Relationship Judge] SKIP "
                    "(no similar memories)"
                )

            # --------------------------------------------------
            # 存在相似 Memory：
            #
            # 调用 Relationship Judge。
            # --------------------------------------------------

            else:

                relationship_result = (
                    self.relationship_judge.judge(
                        candidate,
                        similar_memories,
                    )
                )

                print(
                    "[Relationship Judge] Result:"
                )

                for relationship in (
                    relationship_result.relationships
                ):
                    print(
                        f"  - memory_id={relationship.memory_id} | "
                        f"relationship={relationship.relationship} | "
                        f"reason={relationship.reason}"
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

            if relationship_result is None:

                print(
                    "[Final Decision] "
                    "No relationship result -> SAVE"
                )

            # --------------------------------------------------
            # 存在 Relationship Result：
            # --------------------------------------------------

            else:

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

                        print(
                            "[Final Decision] "
                            "Duplicate -> SKIP"
                        )

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

                        print(
                            "[Final Decision] "
                            "Conflict -> SAVE "
                            "(current version does not resolve conflict)"
                        )

                        continue

                    # ------------------------------------------
                    # Related
                    # ------------------------------------------

                    if (
                        relationship.relationship
                        == "related"
                    ):

                        print(
                            "[Final Decision] "
                            "Related -> SAVE"
                        )

                        continue

                    # ------------------------------------------
                    # New
                    # ------------------------------------------

                    if (
                        relationship.relationship
                        == "new"
                    ):

                        print(
                            "[Final Decision] "
                            "New -> SAVE"
                        )

                        continue

            # ==================================================
            # Duplicate → 不保存
            # ==================================================

            if not should_save:
                continue

            # ==================================================
            # 3.6 Persistence
            # ==================================================

            print(
                "[Persistence] Saving memory..."
            )

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

            print(
                f"[Persistence] Saved successfully: "
                f"id={memory.id}"
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

        print(
            f"\n[Memory Pipeline] Completed. "
            f"Saved {len(saved_memories)} memory(s)."
        )

        return saved_memories