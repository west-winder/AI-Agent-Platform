"""
[MANUAL KEEP]

真实 DeepSeek Timeout 集成验证。

目的：
1. 确认真正的 DeepSeek 调用在正常 Timeout 下可以成功。
2. 人为把 Application Overall Timeout 压得极低，
   确认真实调用链会触发 TimeoutError。

依赖：
- 真实 DeepSeek API
- 网络
- .env
"""

import asyncio
import time

from backend.services import llm_service


MESSAGES = [
    {
        "role": "user",
        "content": "只回复：OK",
    }
]


async def case_normal():
    print("=" * 70)
    print("Case 1 - Normal Real DeepSeek Call")
    print("=" * 70)

    start = time.perf_counter()

    result = await llm_service.call_llm(
        messages=MESSAGES
    )

    elapsed = time.perf_counter() - start

    print(f"result  : {result}")
    print(f"elapsed : {elapsed:.2f}s")
    print("Normal Call: PASS")


async def case_forced_overall_timeout():
    print()
    print("=" * 70)
    print("Case 2 - Forced Application Overall Timeout")
    print("=" * 70)

    original_timeout = (
        llm_service.LLM_CHAT_OVERALL_TIMEOUT_SECONDS
    )

    try:
        # 故意压到极低。
        # 目的不是模拟真实生产参数，
        # 而是快速证明 Overall Timeout Contract。
        llm_service.LLM_CHAT_OVERALL_TIMEOUT_SECONDS = 0.01

        start = time.perf_counter()

        try:
            await llm_service.call_llm(
                messages=MESSAGES
            )

        except TimeoutError as exc:
            elapsed = time.perf_counter() - start

            print(
                f"exception: {type(exc).__name__}"
            )
            print(
                f"elapsed  : {elapsed:.3f}s"
            )
            print(
                "Forced Overall Timeout: PASS"
            )

            return

        raise AssertionError(
            "Expected TimeoutError, but call completed."
        )

    finally:
        # 避免测试改变后续运行环境。
        llm_service.LLM_CHAT_OVERALL_TIMEOUT_SECONDS = (
            original_timeout
        )


async def main():
    await case_normal()
    await case_forced_overall_timeout()


if __name__ == "__main__":
    asyncio.run(main())