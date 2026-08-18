from backend.memory.memory_deduplicator import check_duplicate
from backend.schemas.memory_candidate import MemoryCandidate


# ============================================================
# 模拟数据库中已经存在的 Memory
# ============================================================

class FakeMemory:
    """
    模拟数据库中的 Memory ORM 对象。

    这里暂时不连接真实数据库，
    因为我们现在测试的是 Deduplicator 本身的判断逻辑。
    """

    def __init__(self, content: str, memory_type: str):
        self.content = content
        self.memory_type = memory_type


existing_memories = [
    FakeMemory(
        content="用户喜欢Python",
        memory_type="preference"
    ),
    FakeMemory(
        content="用户是测绘工程硕士",
        memory_type="profile"
    ),
    FakeMemory(
        content="用户希望成为AI Agent工程师",
        memory_type="goal"
    )
]


# ============================================================
# 测试案例
# ============================================================

test_cases = [
    # 完全相同
    MemoryCandidate(
        content="用户喜欢Python",
        memory_type="preference"
    ),

    # 语义相似，但当前版本无法识别
    MemoryCandidate(
        content="用户喜欢使用Python进行后端开发",
        memory_type="preference"
    ),

    # 完全相同
    MemoryCandidate(
        content="用户是测绘工程硕士",
        memory_type="profile"
    ),

    # 新Memory
    MemoryCandidate(
        content="用户正在学习FastAPI",
        memory_type="fact"
    ),

    # 只有句号不同
    MemoryCandidate(
        content="用户喜欢Python。",
        memory_type="preference"
    ),

    # 只有首尾空格不同
    MemoryCandidate(
        content=" 用户喜欢Python ",
        memory_type="preference"
    ),

    # 只有感叹号不同
    MemoryCandidate(
        content="用户喜欢Python！",
        memory_type="preference"
    ),
]


# ============================================================
# 执行测试
# ============================================================

for candidate in test_cases:

    result = check_duplicate(
        candidate=candidate,
        memories=existing_memories
    )

    print("=" * 60)

    print("Candidate:")
    print(candidate)

    print("Deduplication Result:")
    print(result)