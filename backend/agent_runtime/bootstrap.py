"""只组装依赖；注入完整依赖时不加载旧服务或读取真实凭据。"""

import os

from openai import AsyncOpenAI

from backend.agent_runtime.deepseek_adapter import DeepSeekAdapter, ErrorTranslator
from backend.agent_runtime.llm_port import LLMPort
from backend.agent_runtime.runtime import AgentRuntime
from backend.tools.bootstrap import build_tool_registry
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.registry import ToolRegistry


def build_agent_runtime(
    *,
    llm: LLMPort | None = None,
    registry: ToolRegistry | None = None,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    overall_timeout_seconds: float | None = None,
    translate_error: ErrorTranslator | None = None,
    max_iterations: int = 5,
) -> AgentRuntime:
    """默认接入已有 DeepSeek Client；测试可注入 LLMPort 或完整 Adapter 依赖。

    Client 归调用者所有，本函数与 Adapter 都不创建或关闭它。
    """
    if llm is not None:
        if any(value is not None for value in (client, model, overall_timeout_seconds, translate_error)):
            raise ValueError("llm cannot be combined with DeepSeek adapter configuration")
    else:
        if any(value is None for value in (client, model, overall_timeout_seconds, translate_error)):
            # 仅默认生产接线需要旧模块；Runtime 和 Adapter 不依赖该模块。
            from backend.services import llm_service

            if client is None:
                client = llm_service.client
            if model is None:
                model = os.getenv("DEFAULT_MODEL", "deepseek-flash")
            if overall_timeout_seconds is None:
                overall_timeout_seconds = llm_service.LLM_CHAT_OVERALL_TIMEOUT_SECONDS
            if translate_error is None:
                translate_error = llm_service._translate_external_error
        llm = DeepSeekAdapter(
            client=client,
            model=model,
            overall_timeout_seconds=overall_timeout_seconds,
            translate_error=translate_error,
        )

    actual_registry = registry if registry is not None else build_tool_registry()
    return AgentRuntime(
        llm=llm,
        registry=actual_registry,
        dispatcher=ToolDispatcher(actual_registry),
        max_iterations=max_iterations,
    )
