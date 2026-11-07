"""最小顺序 Agent Loop；不读取业务历史、Memory 或数据库。"""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
import json
import logging
from time import monotonic
from uuid import uuid4

from backend.agent_runtime.context_builder import ContextBuilder, find_current_turn_start
from backend.agent_runtime.contracts import (
    AgentRunResult,
    AgentRunStatus,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    validate_llm_response,
)
from backend.agent_runtime.llm_port import LLMPort
from backend.agent_runtime.deepseek_adapter import classify_llm_error
from backend.agent_runtime.observability import (
    ObservationEvent,
    ObservationValue,
    Observer,
    RunContext,
    get_default_observer,
    safe_emit_factory,
)
from backend.exceptions.external_exceptions import ExternalTimeoutError
from backend.tools.infrastructure.contracts import ToolContext, ToolErrorInfo, ToolFailure, ToolResult
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.registry import ToolRegistry

logger = logging.getLogger(__name__)
LLM_CALL_BUDGET_SECONDS = 180.0


def _log(level: int, message: str, extra: dict[str, ObservationValue]) -> None:
    """共享日志出口故障时，既有诊断同样不能掩盖业务结果或原始异常。"""
    try:
        logger.log(level, message, extra=extra)
    except Exception:
        pass


class AgentRuntime:
    def __init__(
        self,
        llm: LLMPort,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        max_iterations: int = 5,
        *,
        context_builder: ContextBuilder | None = None,
        observer: Observer | None = None,
    ) -> None:
        """注入模型、工具、上下文构造器和 Observer，保留位置参数兼容性。"""
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self._llm = llm
        self._registry = registry
        self._dispatcher = dispatcher
        self._max_iterations = max_iterations
        self._observer = observer if observer is not None else get_default_observer()
        self._context_builder = (
            context_builder if context_builder is not None else ContextBuilder()
        )

    def _emit(self, run_id: str, event_type: str, **attributes: ObservationValue) -> None:
        # 事件校验/时间戳失败也属于观测故障，不能先于安全发送逃逸到业务路径。
        safe_emit_factory(
            self._observer,
            lambda: ObservationEvent(run_id, event_type, datetime.now(UTC), attributes),
            run_id=run_id, event_type=event_type,
        )

    async def _complete(
        self, request: LLMRequest, iteration: int, *, run_id: str, llm_call_id: str,
    ) -> LLMResponse:
        """一个逻辑调用最多两次尝试；工具执行不在恢复边界内。"""
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + LLM_CALL_BUDGET_SECONDS
        observed_started = monotonic()
        attributes = {"llm_call_id": llm_call_id, "iteration": iteration}
        self._emit(run_id, "llm.started", **attributes)
        budget = asyncio.timeout_at(deadline)
        waiting_error: Exception | None = None
        attempt = 0
        try:
            try:
                async with budget:
                    for attempt in (1, 2):
                        try:
                            attempt_request = deepcopy(request)
                            # 实际请求开始才计 Attempt/Retry；保留原有逻辑调用和失败事件语义。
                            self._emit(run_id, "llm.attempt_started", **attributes, attempt=attempt)
                            response = await self._llm.complete(attempt_request)
                            if loop.time() >= deadline:
                                raise ExternalTimeoutError("外部服务调用超时")
                            break
                        except Exception as exc:
                            decision = classify_llm_error(exc)
                            remaining = deadline - loop.time()
                            delay = decision.delay_seconds
                            retry = attempt == 1 and delay is not None and remaining > delay
                            failure = {
                                **attributes,
                                "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "category": decision.category,
                                "status_code": decision.status_code,
                                "elapsed_seconds": monotonic() - observed_started,
                                "remaining_seconds": max(0.0, remaining),
                                "retry_delay_seconds": delay,
                                "will_retry": retry,
                            }
                            _log(logging.WARNING, "agent_runtime.llm_failure", {"run_id": run_id, **failure})
                            self._emit(run_id, "llm.attempt_failed", **failure)
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
                _log(
                    logging.WARNING, "agent_runtime.llm_budget_exhausted",
                    {"run_id": run_id, **attributes, "attempt": attempt},
                )
                if waiting_error is not None:
                    # 等待耗尽预算，尚未开始第二次调用：保留首次失败。
                    raise waiting_error
                # timeout_at 用取消中止的在途尝试不会进入上面的 Exception 分支。
                self._emit(
                    run_id, "llm.attempt_failed", **attributes, attempt=attempt,
                    error_type="ExternalTimeoutError", category="timeout",
                    status_code=None, will_retry=False,
                    elapsed_seconds=monotonic() - observed_started,
                    remaining_seconds=0.0, retry_delay_seconds=None,
                )
                raise ExternalTimeoutError("外部服务调用超时") from exc
        except asyncio.CancelledError:
            self._emit(
                run_id, "llm.cancelled", **attributes, attempts=attempt,
                elapsed_seconds=monotonic() - observed_started,
            )
            raise
        except Exception as exc:
            self._emit(
                run_id, "llm.exception", **attributes, attempts=attempt,
                error_type=type(exc).__name__, elapsed_seconds=monotonic() - observed_started,
            )
            raise
        self._emit(
            run_id, "llm.completed", **attributes, attempts=attempt,
            elapsed_seconds=monotonic() - observed_started,
        )
        return response

    async def run(
        self,
        messages: list[LLMMessage],
        tool_context: ToolContext,
    ) -> AgentRunResult:
        """执行一次标准用户请求；维护完整 History，每次调用构造独立 Context。"""
        if not messages:
            raise ValueError("messages must not be empty")
        run = RunContext.create()
        self._emit(run.run_id, "run.started")
        try:
            result = await self._run_loop(messages, tool_context, run)
        except asyncio.CancelledError:
            self._emit(
                run.run_id, "run.cancelled", elapsed_seconds=monotonic() - run.started_monotonic,
            )
            raise
        except Exception as exc:
            self._emit(
                run.run_id, "run.exception", error_type=type(exc).__name__,
                elapsed_seconds=monotonic() - run.started_monotonic,
            )
            raise
        self._emit(
            run.run_id, "run.finished", status=result.status.value, iterations=result.iterations,
            elapsed_seconds=monotonic() - run.started_monotonic,
        )
        return result

    async def _run_loop(
        self, messages: list[LLMMessage], tool_context: ToolContext, run: RunContext,
    ) -> AgentRunResult:
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
            llm_call_id = uuid4().hex
            response = await self._complete(
                LLMRequest(messages=deepcopy(context), tools=deepcopy(tools)), iteration,
                run_id=run.run_id, llm_call_id=llm_call_id,
            )
            validate_llm_response(response)
            _log(
                logging.INFO, "agent_runtime.iteration",
                {
                    "run_id": run.run_id,
                    "llm_call_id": llm_call_id,
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
                _log(logging.INFO, "agent_runtime.max_iterations", {"run_id": run.run_id, "iteration": iteration})
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
                tool_started = monotonic()
                attributes = {
                    "iteration": iteration, "tool_call_id": call.id,
                    "parent_llm_call_id": llm_call_id, "tool_name": call.name,
                }
                self._emit(run.run_id, "tool.started", **attributes)
                try:
                    result = await self._process_tool(call, history, tool_context)
                    _log(
                        logging.INFO, "agent_runtime.tool_result",
                        {
                            "run_id": run.run_id, **attributes,
                            "tool_status": result.status,
                            "tool_error_code": result.error.code if isinstance(result, ToolFailure) else None,
                        },
                    )
                except asyncio.CancelledError:
                    self._emit(
                        run.run_id, "tool.cancelled", **attributes,
                        elapsed_seconds=monotonic() - tool_started,
                    )
                    raise
                except Exception as exc:
                    self._emit(
                        run.run_id, "tool.exception", **attributes, error_type=type(exc).__name__,
                        elapsed_seconds=monotonic() - tool_started,
                    )
                    raise
                self._emit(
                    run.run_id, "tool.completed", **attributes,
                    status=result.status,
                    error_code=result.error.code if isinstance(result, ToolFailure) else None,
                    elapsed_seconds=monotonic() - tool_started,
                )

        raise AssertionError("Unreachable: iteration limit must return a result")

    async def _process_tool(
        self, call: LLMToolCall, history: list[LLMMessage], tool_context: ToolContext,
    ) -> ToolResult:
        """完整工具分支；观测边界包含参数错误、序列化及 History 写入。"""
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
        return result
