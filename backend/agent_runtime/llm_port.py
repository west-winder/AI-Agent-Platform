"""一次模型调用的接口；不依赖任何 Provider 或工具执行组件。"""

from typing import Protocol

from backend.agent_runtime.contracts import LLMRequest, LLMResponse

# 任何满足 LLMPort 的模型组件，都应该提供：
# - 一个叫 complete() 的异步方法。
# - 接收 LLMRequest。
# - 返回 LLMResponse。
class LLMPort(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """完成一次非流式模型请求，或向上抛出不可恢复的异常。"""
        ...
