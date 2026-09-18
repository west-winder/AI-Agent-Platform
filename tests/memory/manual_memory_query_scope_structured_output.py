import asyncio

from backend.memory.memory_read.memory_query_scope_judge import (
    MemoryQueryScopeJudge,
)


async def main():
    judge = MemoryQueryScopeJudge()

    cases = [
        # 同时出现 current + historical marker，
        # Layer 1 会拿不准，因此进入真实 LLM
        (
            "我现在想知道我以前主要学什么？",
            "historical",
        ),

        # 没有明确时间 marker，
        # 但属于转变过程，应由 LLM 判断 both
        (
            "我为什么从 Python 转向 Java？",
            "both",
        ),

        # 没有明确时间 marker，
        # Prompt 规则要求时间范围不明确时优先 current
        (
            "我的主要技术方向是什么？",
            "current",
        ),
    ]

    for query, expected_scope in cases:
        print("=" * 70)
        print("Query:")
        print(query)

        result = await judge.judge(query)

        print()
        print("Result:")
        print("scope :", result.scope)
        print("reason:", result.reason)
        print("source:", result.source)

        print()
        print("Expected scope:", expected_scope)

        if result.scope == expected_scope:
            print("Result: PASS")
        else:
            print("Result: CHECK")

        print()


if __name__ == "__main__":
    asyncio.run(main())