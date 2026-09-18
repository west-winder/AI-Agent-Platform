import asyncio
from dataclasses import dataclass

from backend.memory.memory_write.memory_relationship_judge import (
    ALLOWED_RELATIONSHIPS,
    MemoryRelationshipJudge,
    validate_relationship_coverage,
)
from backend.schemas.memory_candidate import MemoryCandidate


# ==================================================
# MANUAL KEEP
#
# Real DeepSeek Structured Output Integration
#
# 验证：
#
# MemoryRelationshipJudge
#     ↓
# call_llm_structured()
#     ↓
# DeepSeek
#     ↓
# MemoryRelationshipLLMOutput
#     ↓
# Runtime Business Validation
#     ↓
# MemoryRelationshipResult
#     ↓
# validate_relationship_coverage()
#
# 本脚本：
# - 调用真实 DeepSeek
# - 不属于 deterministic regression
# - 长期保留为 [MANUAL KEEP]
# ==================================================


@dataclass
class ExistingMemoryStub:
    """
    只提供 MemoryRelationshipJudge
    本次测试真正需要的字段。

    不依赖数据库 ORM。
    """

    memory_id: int
    content: str
    memory_type: str
    similarity: float


async def run_case(
    judge: MemoryRelationshipJudge,
    *,
    title: str,
    candidate: MemoryCandidate,
    existing_memories: list[ExistingMemoryStub],
    expected_relationships: dict[int, str],
):
    print("=" * 80)
    print(title)
    print()

    # ==================================================
    # Input
    # ==================================================

    print("Candidate:")
    print("content     :", candidate.content)
    print("memory_type :", candidate.memory_type)
    print()

    print("Existing Memories:")

    for memory in existing_memories:
        print(
            f"[id={memory.memory_id}] "
            f"type={memory.memory_type} "
            f"similarity={memory.similarity} "
            f"content={memory.content}"
        )

    print()

    # ==================================================
    # Real Relationship Judge
    # ==================================================

    result = await judge.judge(
        candidate=candidate,
        existing_memories=existing_memories,
    )

    print("Relationships:")

    if not result.relationships:
        print("[]")

    for relationship in result.relationships:
        print(
            f"[memory_id={relationship.memory_id}] "
            f"relationship={relationship.relationship} "
            f"reason={relationship.reason}"
        )

    print()

    # ==================================================
    # Hard Contract 1
    #
    # Judge 不得生成输入中不存在的 ID。
    # ==================================================

    expected_ids = {
        memory.memory_id
        for memory in existing_memories
    }

    actual_ids = {
        relationship.memory_id
        for relationship in result.relationships
    }

    assert actual_ids.issubset(
        expected_ids
    ), (
        "Judge 返回了不存在的 memory_id："
        f"{actual_ids - expected_ids}"
    )

    # ==================================================
    # Hard Contract 2
    #
    # 每条 Relationship 本身必须合法。
    # ==================================================

    for relationship in result.relationships:

        assert relationship.relationship in (
            ALLOWED_RELATIONSHIPS
        )

        assert isinstance(
            relationship.reason,
            str,
        )

        assert relationship.reason.strip()

    # ==================================================
    # Hard Contract 3
    #
    # 最关键：
    # 必须完整覆盖每一个 Existing Memory。
    #
    # 如果真实 Provider 调用失败，
    # Judge 可能由于 fail-closed 返回 []。
    #
    # 这里只看 result.relationships 不够，
    # 必须继续经过 Coverage Contract。
    # ==================================================

    validate_relationship_coverage(
        existing_memories=existing_memories,
        relationship_result=result,
    )

    print("Coverage Contract: PASS")

    # ==================================================
    # Semantic Check
    #
    # LLM 判断有一定非确定性，
    # 所以这里使用 PASS / CHECK，
    # 不把具体 relationship 写成硬 assert。
    # ==================================================

    actual_relationships = {
        relationship.memory_id:
            relationship.relationship
        for relationship in result.relationships
    }

    print()
    print(
        "Expected relationships:",
        expected_relationships,
    )

    print(
        "Actual relationships  :",
        actual_relationships,
    )

    if (
        actual_relationships
        == expected_relationships
    ):
        print("Semantic Result: PASS")
    else:
        print("Semantic Result: CHECK")

    print()


async def main():
    judge = MemoryRelationshipJudge()

    # ==================================================
    # Case 1
    # Duplicate
    #
    # 同一长期偏好 + 同 memory_type
    # 应判断 duplicate。
    # ==================================================

    await run_case(
        judge,
        title="Case 1 - Duplicate",
        candidate=MemoryCandidate(
            content="用户喜欢使用 Python",
            memory_type="preference",
        ),
        existing_memories=[
            ExistingMemoryStub(
                memory_id=101,
                content="用户喜欢 Python",
                memory_type="preference",
                similarity=0.96,
            ),
        ],
        expected_relationships={
            101: "duplicate",
        },
    )

    # ==================================================
    # Case 2
    # Conflict
    #
    # 同一主题，但状态明显矛盾。
    # ==================================================

    await run_case(
        judge,
        title="Case 2 - Conflict",
        candidate=MemoryCandidate(
            content="用户已经停止学习 FastAPI",
            memory_type="fact",
        ),
        existing_memories=[
            ExistingMemoryStub(
                memory_id=201,
                content="用户正在学习 FastAPI",
                memory_type="fact",
                similarity=0.94,
            ),
        ],
        expected_relationships={
            201: "conflict",
        },
    )

    # ==================================================
    # Case 3
    # Related + New
    #
    # 同时验证：
    #
    # 一个 Existing Memory 与 Candidate 有关联，
    # 另一个完全属于不同主题。
    #
    # 也顺便验证一次调用返回多条 Relationship。
    # ==================================================

    await run_case(
        judge,
        title="Case 3 - Related And New",
        candidate=MemoryCandidate(
            content="用户正在学习 LangGraph",
            memory_type="fact",
        ),
        existing_memories=[
            ExistingMemoryStub(
                memory_id=301,
                content="用户正在学习 FastAPI",
                memory_type="fact",
                similarity=0.72,
            ),
            ExistingMemoryStub(
                memory_id=302,
                content="用户喜欢吃火锅",
                memory_type="preference",
                similarity=0.12,
            ),
        ],
        expected_relationships={
            301: "related",
            302: "new",
        },
    )

    # ==================================================
    # Case 4
    # High Similarity But Conflict
    #
    # 专门验证 Prompt 的重要规则：
    #
    # similarity 高
    # !=
    # duplicate
    #
    # 语义相反时仍然应该是 conflict。
    # ==================================================

    await run_case(
        judge,
        title="Case 4 - High Similarity But Conflict",
        candidate=MemoryCandidate(
            content="用户不喜欢 Python",
            memory_type="preference",
        ),
        existing_memories=[
            ExistingMemoryStub(
                memory_id=401,
                content="用户喜欢 Python",
                memory_type="preference",
                similarity=0.98,
            ),
        ],
        expected_relationships={
            401: "conflict",
        },
    )


if __name__ == "__main__":
    asyncio.run(main())