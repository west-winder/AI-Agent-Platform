from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.memory.memory_pipeline import MemoryPipeline
from backend.schemas.memory_candidate import MemoryCandidate
from backend.schemas.memory_validation import MemoryValidationResult


# ============================================================
# 测试辅助工具
# ============================================================

def create_fake_memory(
    content: str,
    memory_type: str,
    user_id: int = 1,
    memory_id: int = 1,
):
    """
    创建一个假的 Memory 对象。

    注意：
    这里不是 SQLAlchemy ORM 对象。
    只是为了让 Pipeline 测试可以模拟
    create_memory() 返回的对象。
    """

    return SimpleNamespace(
        id=memory_id,
        user_id=user_id,
        content=content,
        memory_type=memory_type,
    )


def print_separator():
    print("\n" + "=" * 70)


def print_test_header(test_number: int, title: str):
    print_separator()
    print(f"TEST {test_number}: {title}")
    print_separator()


# ============================================================
# Test 1
# 正常提取并保存 Memory
# ============================================================

def test_normal_memory_save():

    print_test_header(
        1,
        "正常提取并保存 Memory"
    )

    # --------------------------------------------------------
    # Arrange
    # --------------------------------------------------------

    pipeline = MemoryPipeline()

    candidate = MemoryCandidate(
        content="用户喜欢使用Python进行后端开发",
        memory_type="preference"
    )

    fake_memory = create_fake_memory(
        content=candidate.content,
        memory_type=candidate.memory_type,
        user_id=1,
        memory_id=1
    )

    # Mock Extractor
    pipeline.extractor.extract = MagicMock(
        return_value=[candidate]
    )

    # Mock Validator
    pipeline.validator.validate = MagicMock(
        return_value=MemoryValidationResult(
            valid=True,
            reason="长期稳定的技术偏好"
        )
    )

    # Mock Persistence
    create_memory_mock = MagicMock(
        return_value=fake_memory
    )

    get_memories_mock = MagicMock(
        return_value=[]
    )

    # 替换 Pipeline 内部使用的函数
    import backend.memory.memory_pipeline as pipeline_module

    original_create_memory = pipeline_module.create_memory
    original_get_memories = pipeline_module.get_memories

    pipeline_module.create_memory = create_memory_mock
    pipeline_module.get_memories = get_memories_mock

    try:

        print("【Input】")
        print("user_id      =", 1)
        print("user_message =", "我很喜欢使用Python进行后端开发")

        print("\n【Extractor Mock】")
        print("Candidate:")
        print(candidate)

        print("\n【Validator Mock】")
        print("valid  = True")
        print("reason = 长期稳定的技术偏好")

        print("\n【Existing Memories】")
        print("[]")

        # ----------------------------------------------------
        # Act
        # ----------------------------------------------------

        result = pipeline.process(
            db=MagicMock(),
            user_id=1,
            user_message="我很喜欢使用Python进行后端开发"
        )

        # ----------------------------------------------------
        # Assert
        # ----------------------------------------------------

        assert len(result) == 1
        assert result[0].content == candidate.content
        assert result[0].memory_type == candidate.memory_type

        pipeline.extractor.extract.assert_called_once()

        pipeline.validator.validate.assert_called_once_with(
            candidate
        )

        create_memory_mock.assert_called_once()

        print("\n【Persistence】")
        print("create_memory() 调用次数 =", create_memory_mock.call_count)

        print("\n【Result】")
        print("saved_memories =", result)

        print("\n[PASS] 正常 Memory 成功保存")

    finally:

        pipeline_module.create_memory = original_create_memory
        pipeline_module.get_memories = original_get_memories


# ============================================================
# Test 2
# Validation 失败
# ============================================================

def test_validation_failure():

    print_test_header(
        2,
        "Validation 失败时不保存 Memory"
    )

    pipeline = MemoryPipeline()

    candidate = MemoryCandidate(
        content="用户今晚想吃火锅",
        memory_type="preference"
    )

    pipeline.extractor.extract = MagicMock(
        return_value=[candidate]
    )

    pipeline.validator.validate = MagicMock(
        return_value=MemoryValidationResult(
            valid=False,
            reason="一次性临时需求，不具备长期记忆价值"
        )
    )

    create_memory_mock = MagicMock()
    get_memories_mock = MagicMock(
        return_value=[]
    )

    import backend.memory.memory_pipeline as pipeline_module

    original_create_memory = pipeline_module.create_memory
    original_get_memories = pipeline_module.get_memories

    pipeline_module.create_memory = create_memory_mock
    pipeline_module.get_memories = get_memories_mock

    try:

        print("【Input】")
        print("user_message =", "我今晚想吃火锅")

        print("\n【Extractor Mock】")
        print(candidate)

        print("\n【Validator Mock】")
        print("valid  = False")
        print("reason = 一次性临时需求，不具备长期记忆价值")

        result = pipeline.process(
            db=MagicMock(),
            user_id=1,
            user_message="我今晚想吃火锅"
        )

        assert result == []

        pipeline.extractor.extract.assert_called_once()
        pipeline.validator.validate.assert_called_once_with(
            candidate
        )

        # Validation失败后不应该继续执行 Persistence
        create_memory_mock.assert_not_called()

        print("\n【Result】")
        print("saved_memories =", result)
        print("create_memory() 调用次数 =", create_memory_mock.call_count)

        print("\n[PASS] Validation 失败后没有保存 Memory")

    finally:

        pipeline_module.create_memory = original_create_memory
        pipeline_module.get_memories = original_get_memories


# ============================================================
# Test 3
# Exact Match Duplicate
# ============================================================

def test_exact_duplicate():

    print_test_header(
        3,
        "Exact Match Duplicate 不重复保存"
    )

    pipeline = MemoryPipeline()

    candidate = MemoryCandidate(
        content="用户喜欢Python",
        memory_type="preference"
    )

    existing_memory = create_fake_memory(
        content="用户喜欢Python",
        memory_type="preference",
        user_id=1,
        memory_id=1
    )

    pipeline.extractor.extract = MagicMock(
        return_value=[candidate]
    )

    pipeline.validator.validate = MagicMock(
        return_value=MemoryValidationResult(
            valid=True,
            reason="长期稳定的偏好"
        )
    )

    create_memory_mock = MagicMock()

    get_memories_mock = MagicMock(
        return_value=[existing_memory]
    )

    import backend.memory.memory_pipeline as pipeline_module

    original_create_memory = pipeline_module.create_memory
    original_get_memories = pipeline_module.get_memories

    pipeline_module.create_memory = create_memory_mock
    pipeline_module.get_memories = get_memories_mock

    try:

        print("【Candidate】")
        print(candidate)

        print("\n【Existing Memory】")
        print(existing_memory)

        print("\n【Expected Deduplication】")
        print("candidate.content == existing_memory.content")
        print("=> duplicate = True")

        result = pipeline.process(
            db=MagicMock(),
            user_id=1,
            user_message="我喜欢Python"
        )

        assert result == []

        pipeline.extractor.extract.assert_called_once()
        pipeline.validator.validate.assert_called_once_with(
            candidate
        )

        create_memory_mock.assert_not_called()

        print("\n【Result】")
        print("saved_memories =", result)
        print("create_memory() 调用次数 =", create_memory_mock.call_count)

        print("\n[PASS] Exact Match 正确阻止重复 Memory")

    finally:

        pipeline_module.create_memory = original_create_memory
        pipeline_module.get_memories = original_get_memories


# ============================================================
# Test 4
# 一条消息产生多个 Memory
# ============================================================

def test_multiple_memories():

    print_test_header(
        4,
        "一次 Extraction 产生多个 Memory"
    )

    pipeline = MemoryPipeline()

    candidate_1 = MemoryCandidate(
        content="用户正在学习FastAPI",
        memory_type="fact"
    )

    candidate_2 = MemoryCandidate(
        content="用户希望未来从事AI Agent开发",
        memory_type="goal"
    )

    fake_memory_1 = create_fake_memory(
        content=candidate_1.content,
        memory_type=candidate_1.memory_type,
        user_id=1,
        memory_id=1
    )

    fake_memory_2 = create_fake_memory(
        content=candidate_2.content,
        memory_type=candidate_2.memory_type,
        user_id=1,
        memory_id=2
    )

    pipeline.extractor.extract = MagicMock(
        return_value=[
            candidate_1,
            candidate_2
        ]
    )

    pipeline.validator.validate = MagicMock(
        side_effect=[
            MemoryValidationResult(
                valid=True,
                reason="当前长期学习方向"
            ),
            MemoryValidationResult(
                valid=True,
                reason="长期职业目标"
            )
        ]
    )

    create_memory_mock = MagicMock(
        side_effect=[
            fake_memory_1,
            fake_memory_2
        ]
    )

    get_memories_mock = MagicMock(
        return_value=[]
    )

    import backend.memory.memory_pipeline as pipeline_module

    original_create_memory = pipeline_module.create_memory
    original_get_memories = pipeline_module.get_memories

    pipeline_module.create_memory = create_memory_mock
    pipeline_module.get_memories = get_memories_mock

    try:

        print("【Candidates】")

        print("Candidate 1:")
        print(candidate_1)

        print("\nCandidate 2:")
        print(candidate_2)

        result = pipeline.process(
            db=MagicMock(),
            user_id=1,
            user_message=(
                "我正在学习FastAPI，"
                "未来希望从事AI Agent开发"
            )
        )

        assert len(result) == 2

        assert result[0].content == candidate_1.content
        assert result[1].content == candidate_2.content

        assert pipeline.validator.validate.call_count == 2
        assert create_memory_mock.call_count == 2

        print("\n【Persistence】")
        print(
            "create_memory() 调用次数 =",
            create_memory_mock.call_count
        )

        print("\n【Result】")
        for index, memory in enumerate(result, start=1):
            print(f"Memory {index}: {memory}")

        print("\n[PASS] 一条消息成功保存多个 Memory")

    finally:

        pipeline_module.create_memory = original_create_memory
        pipeline_module.get_memories = original_get_memories


# ============================================================
# Test 5
# 同一次 Extraction 内部出现重复 Candidate
# ============================================================

def test_duplicate_candidates_in_same_pipeline():

    print_test_header(
        5,
        "同一次 Pipeline 中出现重复 Candidate"
    )

    pipeline = MemoryPipeline()

    candidate_1 = MemoryCandidate(
        content="用户喜欢Python",
        memory_type="preference"
    )

    candidate_2 = MemoryCandidate(
        content="用户喜欢Python",
        memory_type="preference"
    )

    fake_memory = create_fake_memory(
        content="用户喜欢Python",
        memory_type="preference",
        user_id=1,
        memory_id=1
    )

    pipeline.extractor.extract = MagicMock(
        return_value=[
            candidate_1,
            candidate_2
        ]
    )

    pipeline.validator.validate = MagicMock(
        return_value=MemoryValidationResult(
            valid=True,
            reason="长期稳定的偏好"
        )
    )

    create_memory_mock = MagicMock(
        return_value=fake_memory
    )

    get_memories_mock = MagicMock(
        return_value=[]
    )

    import backend.memory.memory_pipeline as pipeline_module

    original_create_memory = pipeline_module.create_memory
    original_get_memories = pipeline_module.get_memories

    pipeline_module.create_memory = create_memory_mock
    pipeline_module.get_memories = get_memories_mock

    try:

        print("【Candidates】")

        print("Candidate 1:")
        print(candidate_1)

        print("\nCandidate 2:")
        print(candidate_2)

        print("\n【Expected Behavior】")
        print("Candidate 1 → Save")
        print("Candidate 2 → Exact Match → Skip")

        result = pipeline.process(
            db=MagicMock(),
            user_id=1,
            user_message="我喜欢Python"
        )

        assert len(result) == 1

        assert create_memory_mock.call_count == 1

        assert pipeline.validator.validate.call_count == 2

        print("\n【Persistence】")
        print(
            "create_memory() 调用次数 =",
            create_memory_mock.call_count
        )

        print("\n【Result】")
        print("saved_memories =", result)

        print(
            "\n[PASS] 同一次 Pipeline 内部重复 Candidate "
            "没有重复保存"
        )

    finally:

        pipeline_module.create_memory = original_create_memory
        pipeline_module.get_memories = original_get_memories


# ============================================================
# Test 6
# 一个 Candidate Validation 失败，
# 不影响其他 Candidate
# ============================================================

def test_validation_failure_does_not_stop_pipeline():

    print_test_header(
        6,
        "单个 Candidate Validation 失败不影响其他 Candidate"
    )

    pipeline = MemoryPipeline()

    candidate_1 = MemoryCandidate(
        content="用户正在学习FastAPI",
        memory_type="fact"
    )

    candidate_2 = MemoryCandidate(
        content="用户今晚想吃火锅",
        memory_type="preference"
    )

    candidate_3 = MemoryCandidate(
        content="用户希望未来从事AI Agent开发",
        memory_type="goal"
    )

    fake_memory_1 = create_fake_memory(
        content=candidate_1.content,
        memory_type=candidate_1.memory_type,
        user_id=1,
        memory_id=1
    )

    fake_memory_3 = create_fake_memory(
        content=candidate_3.content,
        memory_type=candidate_3.memory_type,
        user_id=1,
        memory_id=2
    )

    pipeline.extractor.extract = MagicMock(
        return_value=[
            candidate_1,
            candidate_2,
            candidate_3
        ]
    )

    pipeline.validator.validate = MagicMock(
        side_effect=[
            MemoryValidationResult(
                valid=True,
                reason="长期学习状态"
            ),
            MemoryValidationResult(
                valid=False,
                reason="一次性临时需求"
            ),
            MemoryValidationResult(
                valid=True,
                reason="长期职业目标"
            )
        ]
    )

    create_memory_mock = MagicMock(
        side_effect=[
            fake_memory_1,
            fake_memory_3
        ]
    )

    get_memories_mock = MagicMock(
        return_value=[]
    )

    import backend.memory.memory_pipeline as pipeline_module

    original_create_memory = pipeline_module.create_memory
    original_get_memories = pipeline_module.get_memories

    pipeline_module.create_memory = create_memory_mock
    pipeline_module.get_memories = get_memories_mock

    try:

        print("【Candidates】")

        print("Candidate 1:")
        print(candidate_1)

        print("\nCandidate 2:")
        print(candidate_2)

        print("\nCandidate 3:")
        print(candidate_3)

        print("\n【Validation Results】")
        print("Candidate 1 → True")
        print("Candidate 2 → False")
        print("Candidate 3 → True")

        result = pipeline.process(
            db=MagicMock(),
            user_id=1,
            user_message=(
                "我正在学习FastAPI，"
                "今晚想吃火锅，"
                "未来希望从事AI Agent开发"
            )
        )

        assert len(result) == 2

        assert result[0].content == candidate_1.content
        assert result[1].content == candidate_3.content

        assert pipeline.validator.validate.call_count == 3

        assert create_memory_mock.call_count == 2

        print("\n【Persistence】")
        print(
            "create_memory() 调用次数 =",
            create_memory_mock.call_count
        )

        print("\n【Result】")

        for index, memory in enumerate(result, start=1):
            print(f"Memory {index}: {memory}")

        print(
            "\n[PASS] 一个 Candidate Validation 失败，"
            "没有影响其他 Candidate"
        )

    finally:

        pipeline_module.create_memory = original_create_memory
        pipeline_module.get_memories = original_get_memories


# ============================================================
# Main
# ============================================================

def main():

    print_separator()
    print("Memory Pipeline V1 Test")
    print("测试范围：")
    print("Extractor → Validation → Exact Match → Persistence")
    print()
    print("数据库：不连接 SQLite")
    print("LLM：不调用真实 LLM")
    print("Deduplicator：使用真实 Exact Match 实现")
    print_separator()

    tests = [
        test_normal_memory_save,
        test_validation_failure,
        test_exact_duplicate,
        test_multiple_memories,
        test_duplicate_candidates_in_same_pipeline,
        test_validation_failure_does_not_stop_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:

        try:

            test()

            passed += 1

        except AssertionError as e:

            failed += 1

            print("\n[FAIL]")
            print("测试：", test.__name__)
            print("AssertionError：", e)

        except Exception as e:

            failed += 1

            print("\n[ERROR]")
            print("测试：", test.__name__)
            print("异常类型：", type(e).__name__)
            print("异常信息：", e)

    print_separator()
    print("TEST SUMMARY")
    print_separator()

    print("Total :", len(tests))
    print("Passed:", passed)
    print("Failed:", failed)

    if failed == 0:

        print("\n[SUCCESS] Memory Pipeline V1 全部测试通过")

    else:

        print("\n[FAILED] Memory Pipeline V1 存在测试失败")

    print_separator()


if __name__ == "__main__":
    main()