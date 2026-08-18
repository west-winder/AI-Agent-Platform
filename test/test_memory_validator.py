from backend.memory.memory_validator import MemoryValidator
from backend.schemas.memory_candidate import MemoryCandidate


validator = MemoryValidator()


# ==================================================
# 测试案例
# ==================================================

test_cases = [
    MemoryCandidate(
        content="好的",
        memory_type="fact",
    ),

    MemoryCandidate(
        content="用户今晚想吃火锅",
        memory_type="preference",
    ),

    MemoryCandidate(
        content="用户喜欢使用Python进行后端开发",
        memory_type="preference",
    ),

    MemoryCandidate(
        content="用户希望未来从事AI Agent开发",
        memory_type="goal",
    ),

    MemoryCandidate(
        content="用户是测绘工程硕士研究生",
        memory_type="profile",
    ),

    MemoryCandidate(
        content="今天深圳下雨了",
        memory_type="fact",
    ),
]


for candidate in test_cases:

    print("=" * 60)

    print("Candidate:")
    print(candidate)

    result = validator.validate(candidate)

    print("Validation Result:")
    print(result)