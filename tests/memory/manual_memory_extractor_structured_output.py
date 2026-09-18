import asyncio

from backend.memory.memory_write.memory_extractor import (
    MemoryExtractor,
)


async def run_case(
    extractor: MemoryExtractor,
    *,
    title: str,
    user_message: str,
    should_extract: bool,
):
    print("=" * 80)
    print(title)
    print()

    print("User Message:")
    print(user_message)
    print()

    candidates = await extractor.extract(
        user_message
    )

    print("Extracted Candidates:")

    if not candidates:
        print("[]")
    else:
        for index, candidate in enumerate(
            candidates
        ):
            print(
                f"[{index}] "
                f"type={candidate.memory_type} "
                f"content={candidate.content}"
            )

    # ==================================================
    # Hard Contract
    # ==================================================

    allowed_types = {
        "fact",
        "preference",
        "goal",
        "profile",
    }

    for candidate in candidates:

        assert isinstance(
            candidate.content,
            str,
        )

        assert candidate.content

        # Extractor 应该已经完成 strip
        assert (
            candidate.content
            == candidate.content.strip()
        )

        assert (
            candidate.memory_type
            in allowed_types
        )

    # ==================================================
    # Semantic Check
    # ==================================================

    actually_extracted = bool(
        candidates
    )

    print()
    print(
        "Expected extract:",
        should_extract,
    )

    print(
        "Actual extract  :",
        actually_extracted,
    )

    if actually_extracted is should_extract:
        print("Result: PASS")
    else:
        print("Result: CHECK")

    print()


async def main():
    extractor = MemoryExtractor()

    # ==================================================
    # Case 1
    # Current State
    # ==================================================

    await run_case(
        extractor,
        title="Case 1 - Current State",
        user_message=(
            "我目前正在学习 AI Agent 开发。"
        ),
        should_extract=True,
    )

    # ==================================================
    # Case 2
    # State Termination
    # ==================================================

    await run_case(
        extractor,
        title="Case 2 - State Termination",
        user_message=(
            "我已经停止学习 C++ 了。"
        ),
        should_extract=True,
    )

    # ==================================================
    # Case 3
    # State Transition
    # ==================================================

    await run_case(
        extractor,
        title="Case 3 - State Transition",
        user_message=(
            "我准备从 Python 后端转向 Java 后端。"
        ),
        should_extract=True,
    )

    # ==================================================
    # Case 4
    # State Restart
    # ==================================================

    await run_case(
        extractor,
        title="Case 4 - State Restart",
        user_message=(
            "我最近重新开始学习 C++ 了。"
        ),
        should_extract=True,
    )

    # ==================================================
    # Case 5
    # One-off Event
    # ==================================================

    await run_case(
        extractor,
        title="Case 5 - One-off Event",
        user_message=(
            "我今天下午学了两个小时 C++。"
        ),
        should_extract=False,
    )

    # ==================================================
    # Case 6
    # Instant Action
    # ==================================================

    await run_case(
        extractor,
        title="Case 6 - Instant Action",
        user_message=(
            "我现在正在点外卖。"
        ),
        should_extract=False,
    )


if __name__ == "__main__":
    asyncio.run(main())