"""最小顺序 Agent Loop；不读取业务历史、Memory 或数据库。"""

import asyncio
from copy import deepcopy
import json
import logging
from uuid import uuid4

from backend.agent_runtime.context_builder import ContextBuilder, find_current_turn_start
from backend.agent_runtime.contracts import (
    AgentRunResult,
    AgentRunStatus,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolDefinition,
    validate_llm_response,
)
from backend.agent_runtime.llm_port import LLMPort
from backend.agent_runtime.deepseek_adapter import classify_llm_error
from backend.exceptions.external_exceptions import ExternalTimeoutError
from backend.tools.infrastructure.contracts import ToolContext, ToolErrorInfo, ToolFailure
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.registry import ToolRegistry

logger = logging.getLogger(__name__)
LLM_CALL_BUDGET_SECONDS = 180.0


class AgentRuntime:
    def __init__(
        self,
        llm: LLMPort,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        max_iterations: int = 5,
        *,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        """注入模型、工具与上下文构造器，保留原有位置参数兼容性。"""
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self._llm = llm
        self._registry = registry
        self._dispatcher = dispatcher
        self._max_iterations = max_iterations
        self._context_builder = (
            context_builder if context_builder is not None else ContextBuilder()
        )

    async def _complete(self, request: LLMRequest, iteration: int) -> LLMResponse:
        """一个逻辑调用最多两次尝试；工具执行不在恢复边界内。"""
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + LLM_CALL_BUDGET_SECONDS
        call_id = uuid4().hex
        budget = asyncio.timeout_at(deadline)
        waiting_error: Exception | None = None
        try:
            async with budget:
                for attempt in (1, 2):
                    try:
                        response = await self._llm.complete(deepcopy(request))
                        if loop.time() >= deadline:
                            raise ExternalTimeoutError("外部服务调用超时")
                        return response
                    except Exception as exc:
                        decision = classify_llm_error(exc)
                        remaining = deadline - loop.time()
                        delay = decision.delay_seconds
                        retry = attempt == 1 and delay is not None and remaining > delay
                        logger.warning(
                            "agent_runtime.llm_failure",
                            extra={
                                "llm_call_id": call_id,
                                "iteration": iteration,
                                "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "category": decision.category,
                                "status_code": decision.status_code,
                                "elapsed_seconds": loop.time() - started,
                                "remaining_seconds": max(0.0, remaining),
                                "retry_delay_seconds": delay,
                                "will_retry": retry,
                            },
                        )
                        if not retry:
                            raise
                        if delay:
                            waiting_error = exc
                            await asyncio.sleep(delay)
                        # 等待后再次核对；不足时保留第一次异常。
                        if loop.time() >= deadline:
                            raise
                        waiting_error = None
        except TimeoutError as exc:
            if not budget.expired():
                raise
            logger.warning(
                "agent_runtime.llm_budget_exhausted",
                extra={"llm_call_id": call_id, "iteration": iteration, "attempt": attempt},
            )
            if waiting_error is not None:
                # 等待耗尽预算，尚未开始第二次调用：保留首次失败。
                raise waiting_error
            raise ExternalTimeoutError("外部服务调用超时") from exc
        raise AssertionError("Unreachable: LLM attempts must return or raise")

    async def run(
        self,
        messages: list[LLMMessage],
        tool_context: ToolContext,
    ) -> AgentRunResult:
        """执行一次标准用户请求；维护完整 History，每次调用构造独立 Context。"""
        if not messages:
            raise ValueError("messages must not be empty")
        # list 的浅拷贝不足以隔离嵌套 arguments 和 tool_calls。
        history = deepcopy(messages)
        # 起点仅在 Run 开始时确定，后续工具调用不会改变它。
        current_turn_start = find_current_turn_start(history)
        tools = [
            LLMToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.args_schema.model_json_schema(),
            )
            for tool in self._registry.list_tools()
        ]

        for iteration in range(1, self._max_iterations + 1):
            context = self._context_builder.build(history, tools, current_turn_start)
            # 每次请求是快照；自定义 LLMPort 不会意外修改 Runtime 历史。
            response = await self._complete(
                LLMRequest(messages=deepcopy(context), tools=deepcopy(tools)), iteration
            )
            validate_llm_response(response)
            logger.info(
                "agent_runtime.iteration",
                extra={
                    "iteration": iteration,
                    "finish_reason": response.finish_reason,
                    "tool_count": len(response.tool_calls),
                },
            )

            if not response.tool_calls:
                history.append(LLMMessage(role="assistant", content=response.content))
                return AgentRunResult(
                    status=AgentRunStatus.COMPLETED,
                    final_answer=response.content,
                    iterations=iteration,
                    messages=history,
                )

            if iteration == self._max_iterations:
                # 没有下一次模型请求可以接收结果，因此不执行或追加本批调用。
                logger.info("agent_runtime.max_iterations", extra={"iteration": iteration})
                return AgentRunResult(
                    status=AgentRunStatus.MAX_ITERATIONS_REACHED,
                    final_answer=None,
                    iterations=iteration,
                    messages=history,
                )

            assistant = LLMMessage(
                role="assistant",
                content=response.content,
                tool_calls=deepcopy(response.tool_calls),
            )
            history.append(assistant)
            for call in assistant.tool_calls:
                if call.parse_error is not None:
                    result = ToolFailure(
                        error=ToolErrorInfo(
                            code="INVALID_ARGUMENTS",
                            message="工具参数不是合法的 JSON 对象",
                            retryable=False,
                        )
                    )
                else:
                    # JSON 解析在 Adapter，Schema 校验和 Handler 调用仅在 Dispatcher。
                    result = await self._dispatcher.dispatch(
                        tool_name=call.name,
                        arguments=deepcopy(call.arguments),
                        context=tool_context,
                    )
                result_json = json.dumps(
                    result.model_dump(mode="json"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                history.append(
                    LLMMessage(role="tool", content=result_json, tool_call_id=call.id)
                )
                logger.info(
                    "agent_runtime.tool_result",
                    extra={
                        "iteration": iteration,
                        "tool_name": call.name,
                        "tool_status": result.status,
                        "tool_error_code": result.error.code if isinstance(result, ToolFailure) else None,
                    },
                )

        raise AssertionError("Unreachable: iteration limit must return a result")
