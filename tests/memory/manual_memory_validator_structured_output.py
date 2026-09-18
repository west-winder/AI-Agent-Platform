import asyncio

from backend.memory.memory_write.memory_validator import (
    MemoryValidator,
)
from backend.schemas.memory_candidate import (
    MemoryCandidate,
)


async def run_case(
    validator: MemoryValidator,
    *,
    title: str,
    candidate: MemoryCandidate,
    expected_valid: bool,
):
    print("=" * 80)
    print(title)
    print()

    print("Candidate:")
    print("content     :", candidate.content)
    print("memory_type :", candidate.memory_type)
    print()

    result = await validator.validate(
        candidate
    )

    print("Result:")
    print("valid :", result.valid)
    print("reason:", result.reason)
    print()

    print(
        "Expected valid:",
        expected_valid,
    )

    if result.valid is expected_valid:
        print("Result: PASS")
    else:
        print("Result: CHECK")

    print()


async def main():
    validator = MemoryValidator()

    # ==================================================
    # Case 1
    # Current State
    # ==================================================

    await run_case(
        validator,
        title="Case 1 - Current State",
        candidate=MemoryCandidate(
            content="用户目前正在学习 AI Agent 开发",
            memory_type="fact",
        ),
        expected_valid=True,
    )

    # ==================================================
    # Case 2
    # State Termination
    # ==================================================

    await run_case(
        validator,
        title="Case 2 - State Termination",
        candidate=MemoryCandidate(
            content="用户已经停止学习 C++",
            memory_type="fact",
        ),
        expected_valid=True,
    )

    # ==================================================
    # Case 3
    # State Transition
    # ==================================================

    await run_case(
        validator,
        title="Case 3 - State Transition",
        candidate=MemoryCandidate(
            content="用户正在从 Python 后端转向 Java 后端",
            memory_type="goal",
        ),
        expected_valid=True,
    )

    # ==================================================
    # Case 4
    # One-off Event
    # ==================================================

    await run_case(
        validator,
        title="Case 4 - One-off Event",
        candidate=MemoryCandidate(
            content="用户今天下午学习了两个小时 C++",
            memory_type="fact",
        ),
        expected_valid=False,
    )


if __name__ == "__main__":
    asyncio.run(main())