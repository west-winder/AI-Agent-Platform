r"""真实 DeepSeek + Weather 手动联调（不由 pytest 收集）。

在仓库根目录执行：
    .venv\Scripts\python.exe -m tests.agent_runtime.manual_deepseek_weather_smoke

复用项目已有 .env / Client；会调用真实外部服务。不打印凭据、请求头或原始异常。
"""

import argparse
import asyncio
import json

from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import AgentRunStatus, LLMMessage
from backend.agent_runtime.errors import LLMProtocolError
from backend.exceptions.external_exceptions import ExternalServiceError
from backend.tools.infrastructure.contracts import ToolContext


async def main(question: str, max_iterations: int) -> None:
    from backend.services.llm_service import client

    try:
        runtime = build_agent_runtime(max_iterations=max_iterations)
        result = await runtime.run([LLMMessage("user", question)], ToolContext())
        print("status:", result.status.value)
        print("iterations:", result.iterations)
        successful_weather = False
        calls = {
            call.id: call.name
            for message in result.messages
            for call in message.tool_calls
        }
        for message in result.messages:
            if message.role == "tool":
                payload = json.loads(message.content)
                name = calls[message.tool_call_id]
                print("tool:", name, "tool_call_id:", message.tool_call_id, "status:", payload["status"])
                if payload["error"] is not None:
                    print("tool_error_code:", payload["error"]["code"])
                if name == "get_weather" and payload["status"] == "success":
                    successful_weather = True
        print("final_answer:", result.final_answer)
        if result.status is not AgentRunStatus.COMPLETED or not successful_weather:
            raise SystemExit("Smoke 未通过：需要成功的 Weather 调用和最终回答。")
        print("Smoke PASSED")
    except (ExternalServiceError, LLMProtocolError) as exc:
        raise SystemExit(f"Smoke 未通过：{type(exc).__name__}；请检查外部服务与配置。") from None
    finally:
        # 此脚本独占进程；关闭复用的 Client，应用中的生命周期仍由上游管理。
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实 DeepSeek + Weather Runtime smoke")
    parser.add_argument("--question", default="请查询武汉现在的天气，并告诉我气温和湿度。")
    parser.add_argument("--max-iterations", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.question, args.max_iterations))
