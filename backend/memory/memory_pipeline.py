from sqlalchemy.orm import Session

from backend.memory.memory_extractor import MemoryExtractor
from backend.memory.memory_validator import MemoryValidator
from backend.memory.memory_deduplicator import check_duplicate
from backend.services.memory_service import (
    create_memory,
    get_memories,
)
from backend.schemas.memory import MemoryCreate


class MemoryPipeline:
    """
    Memory Write Pipeline。

    负责协调：

    User Message
        ↓
    Memory Extraction
        ↓
    Memory Validation
        ↓
    Memory Deduplication
        ↓
    Memory Persistence

    注意：

    Pipeline本身不负责：

    1. Memory提取逻辑
    2. Memory验证逻辑
    3. Memory去重逻辑
    4. 数据库底层操作

    这些职责分别由对应模块负责。
    """

    def __init__(self):

        self.extractor = MemoryExtractor()

        self.validator = MemoryValidator()

    # ==================================================
    # Memory Pipeline
    # ==================================================

    def process(
        self,
        db: Session,
        user_id: int,
        user_message: str
    ):
        """
        执行完整的Memory写入流程。

        流程：

        User Message
                ↓
        Extractor
                ↓
        Validator
                ↓
        Deduplicator
                ↓
        Persistence

        返回本次最终保存的Memory列表。
        """

        # ==================================================
        # Step 1：Memory Extraction
        # ==================================================

        candidates = self.extractor.extract(
            user_message
        )

        # 如果没有提取到Candidate
        if not candidates:
            return []

        # ==================================================
        # Step 2：获取用户已有Memory
        # ==================================================

        memories = get_memories(
            db,
            user_id
        )

        # ==================================================
        # Step 3：逐个处理Candidate
        # ==================================================

        saved_memories = []

        for candidate in candidates:

            # --------------------------------------------------
            # 3.1 Validation
            # --------------------------------------------------

            validation_result = self.validator.validate(
                candidate
            )

            # Validation失败
            if not validation_result.valid:
                continue

            # --------------------------------------------------
            # 3.2 Deduplication
            # --------------------------------------------------

            duplicate_result = check_duplicate(
                candidate,
                memories
            )

            # 已经存在
            if duplicate_result.duplicate:
                continue

            # --------------------------------------------------
            # 3.3 Persistence
            # --------------------------------------------------

            memory_data = MemoryCreate(
                content=candidate.content,
                memory_type=candidate.memory_type
            )

            memory = create_memory(
                db=db,
                user_id=user_id,
                memory_data=memory_data
            )

            # 保存结果
            saved_memories.append(
                memory
            )

            # --------------------------------------------------
            # 非常重要：
            #
            # 把刚刚创建的Memory加入当前Memory列表。
            #
            # 防止同一次Pipeline中出现两个完全相同的Candidate。
            # --------------------------------------------------

            memories.append(
                memory
            )

        return saved_memories