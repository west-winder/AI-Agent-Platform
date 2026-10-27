"""最小顺序 Agent Loop；不读取业务历史、Memory 或数据库。"""

from copy import deepcopy
import json
import logging

from backend.agent_runtime.contracts import (
    AgentRunResult,
    AgentRunStatus,
    LLMMessage,
    LLMRequest,
    LLMToolDefinition,
    validate_llm_response,
)
from backend.agent_runtime.llm_port import LLMPort
from backend.tools.infrastructure.contracts import ToolContext, ToolErrorInfo, ToolFailure
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(
        self,
        llm: LLMPort,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        max_iterations: int = 5,
    ) -> None:
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self._llm = llm
        self._registry = registry
        self._dispatcher = dispatcher
        self._max_iterations = max_iterations

    async def run(
        self,
        messages: list[LLMMessage],
        tool_context: ToolContext,
    ) -> AgentRunResult:
        if not messages:
            raise ValueError("messages must not be empty")
        # list 的浅拷贝不足以隔离嵌套 arguments 和 tool_calls。
        history = deepcopy(messages)
        tools = [
            LLMToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.args_schema.model_json_schema(),
            )
            for tool in self._registry.list_tools()
        ]

        for iteration in range(1, self._max_iterations + 1):
            # 每次请求是快照；自定义 LLMPort 不会意外修改 Runtime 历史。
            response = await self._llm.complete(
                LLMRequest(messages=deepcopy(history), tools=deepcopy(tools))
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
