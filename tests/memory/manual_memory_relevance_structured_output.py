import asyncio

from backend.memory.memory_read.memory_relevance_judge import (
    MemoryRelevanceCandidate,
    MemoryRelevanceJudge,
)


async def run_case(
    judge: MemoryRelevanceJudge,
    *,
    title: str,
    query: str,
    candidates: list[MemoryRelevanceCandidate],
    expected_selected: set[int],
):
    print("=" * 80)
    print(title)
    print()
    print("Query:")
    print(query)
    print()

    print("Candidates:")

    for index, candidate in enumerate(candidates):
        print(
            f"[{index}] "
            f"status={candidate.memory_status} "
            f"content={candidate.content}"
        )

    decisions = await judge.judge(
        query=query,
        candidates=candidates,
    )

    print()
    print("Decisions:")

    for decision in decisions:
        print(
            f"[{decision.index}] "
            f"selected={decision.selected} "
            f"reason={decision.reason}"
        )

    # ==================================================
    # Structured / Runtime Contract
    # ==================================================

    assert len(decisions) == len(candidates)

    actual_indexes = {
        decision.index
        for decision in decisions
    }

    expected_indexes = set(
        range(len(candidates))
    )

    assert actual_indexes == expected_indexes

    for decision in decisions:
        assert isinstance(
            decision.selected,
            bool,
        )

        assert isinstance(
            decision.reason,
            str,
        )

        assert decision.reason.strip()

    # ==================================================
    # Semantic Check
    # ==================================================

    actual_selected = {
        decision.index
        for decision in decisions
        if decision.selected
    }

    print()
    print(
        "Expected selected:",
        expected_selected,
    )

    print(
        "Actual selected  :",
        actual_selected,
    )

    if actual_selected == expected_selected:
        print("Result: PASS")
    else:
        print("Result: CHECK")

    print()


async def main():
    judge = MemoryRelevanceJudge()

    # ==================================================
    # Case 1
    # Current Technical State
    # ==================================================

    await run_case(
        judge,
        title="Case 1 - Current Technical State",
        query="我现在主要在学习什么技术？",
        candidates=[
            MemoryRelevanceCandidate(
                content="用户当前正在学习 AI Agent 开发",
                memory_status="current",
            ),
            MemoryRelevanceCandidate(
                content="用户以前学习过 C++",
                memory_status="historical",
            ),
            MemoryRelevanceCandidate(
                content="用户喜欢吃火锅",
                memory_status="current",
            ),
        ],
        expected_selected={0},
    )

    # ==================================================
    # Case 2
    # Historical Technical State
    # ==================================================

    await run_case(
        judge,
        title="Case 2 - Historical Technical State",
        query="我以前学过什么技术？",
        candidates=[
            MemoryRelevanceCandidate(
                content="用户以前学习过 C++",
                memory_status="historical",
            ),
            MemoryRelevanceCandidate(
                content="用户以前喜欢吃火锅",
                memory_status="historical",
            ),
        ],
        expected_selected={0},
    )

    # ==================================================
    # Case 3
    # State Transition
    # ==================================================

    await run_case(
        judge,
        title="Case 3 - State Transition",
        query="我从 Python 后端转向 Java 后端的经历是什么？",
        candidates=[
            MemoryRelevanceCandidate(
                content="用户曾计划继续学习 Python 后端",
                memory_status="historical",
            ),
            MemoryRelevanceCandidate(
                content="用户当前准备转向 Java 后端",
                memory_status="current",
            ),
            MemoryRelevanceCandidate(
                content="用户喜欢吃火锅",
                memory_status="current",
            ),
        ],
        expected_selected={0, 1},
    )


if __name__ == "__main__":
    asyncio.run(main())