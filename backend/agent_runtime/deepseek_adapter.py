"""DeepSeek Function Calling 适配；客户端的生命周期、HTTP 超时和重试归上游。"""

import asyncio
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import errno
import json
import logging
import math
import ssl
import socket
import time
from typing import Any

import httpcore2
import httpx2
from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletion

from backend.agent_runtime.contracts import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    validate_llm_response,
    validate_tool_calls,
)
from backend.agent_runtime.errors import LLMProtocolError, LLMResponseTruncatedError
from backend.exceptions.external_exceptions import ExternalServiceError, ExternalTimeoutError

logger = logging.getLogger(__name__)
ErrorTranslator = Callable[[TimeoutError | APIError], ExternalServiceError]


@dataclass(frozen=True)
class LLMRetryDecision:
    """Agent 边界的分类结果；None 表示不授权 Runtime 重试。"""

    category: str
    delay_seconds: float | None = None
    status_code: int | None = None


def classify_llm_error(error: Exception) -> LLMRetryDecision:
    """只读取现有原因链；不改变共享异常或 SDK 的重试策略。"""
    if isinstance(error, ExternalTimeoutError):
        return LLMRetryDecision("timeout", 0.0, getattr(error.__cause__, "status_code", None))
    if not isinstance(error, ExternalServiceError):
        return LLMRetryDecision("unclassified")
    cause = error.__cause__
    if isinstance(cause, APIStatusError):
        status = cause.status_code
        if 500 <= status < 600:
            return LLMRetryDecision("server", 0.0, status)
        if status == 429:
            headers = cause.response.headers
            delay = None
            try:
                if "retry-after-ms" in headers:
                    delay = float(headers["retry-after-ms"]) / 1000
                elif "retry-after" in headers:
                    value = headers["retry-after"]
                    try:
                        delay = float(value)
                    except ValueError:
                        date = parsedate_to_datetime(value)
                        if date.tzinfo is not None:
                            delay = date.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError, OSError):
                pass
            # 与已安装 SDK 的最大服务端等待提示保持一致；预算仍由 Runtime 检查。
            if delay is not None and math.isfinite(delay) and 0 <= delay <= 120:
                return LLMRetryDecision("rate_limit", delay, status)
            return LLMRetryDecision("rate_limit", status_code=status)
        return LLMRetryDecision("http_non_retryable", status_code=status)
    if isinstance(cause, APIConnectionError):
        source = cause.__cause__
        # SDK 也会包装任意程序异常；要求存在明确的网络错误来源。
        if isinstance(source, (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.ReadError, httpx2.WriteError)):
            require_transient_root = isinstance(source, (httpx2.ReadError, httpx2.WriteError))
            # Read/Write 的泛化 ConnectionError 必须经过 errno 白名单核验。
            known_os_types = (
                (ConnectionResetError, ConnectionRefusedError, ConnectionAbortedError, BrokenPipeError, TimeoutError)
                if require_transient_root else (ConnectionError, TimeoutError)
            )
            seen: set[int] = set()
            while source is not None and id(source) not in seen:
                seen.add(id(source))
                if isinstance(source, (
                    httpcore2.ConnectError, httpcore2.ConnectTimeout, httpcore2.ReadError, httpcore2.WriteError,
                )):
                    if source.__cause__ is None:
                        return LLMRetryDecision("unclassified_connection")
                    source = source.__cause__
                    continue
                if type(source) is OSError and source.errno is None and source.__cause__ is not None:
                    # AnyIO 包装不单独授予重试权限，继续检查真实底层原因。
                    source = source.__cause__
                    continue
                # Read/Write 包装必须最终落到已有策略允许的 OS 原因，不能止于类型标签。
                if require_transient_root and source.__cause__ is None and not isinstance(source, OSError):
                    return LLMRetryDecision("unclassified_connection")
                if isinstance(source, ssl.SSLError) or not isinstance(
                    source, (httpx2.NetworkError, httpx2.ConnectTimeout, ConnectionError, OSError)
                ):
                    return LLMRetryDecision("unclassified_connection")
                if isinstance(source, socket.gaierror):
                    if source.errno != socket.EAI_AGAIN:
                        return LLMRetryDecision("unclassified_connection")
                elif isinstance(source, OSError) and not isinstance(source, known_os_types):
                    transient = {
                        errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED,
                        errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE,
                        # Windows Winsock 错误码。
                        10054, 10061, 10053, 10060, 10065, 10051,
                    }
                    if source.errno not in transient:
                        return LLMRetryDecision("unclassified_connection")
                source = source.__cause__
            if source is None:
                return LLMRetryDecision("connection", 0.0)
        return LLMRetryDecision("unclassified_connection")
    return LLMRetryDecision("unclassified")


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-standard JSON numeric constant")


class DeepSeekAdapter:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        overall_timeout_seconds: float,
        translate_error: ErrorTranslator,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            isinstance(overall_timeout_seconds, bool)
            or not math.isfinite(overall_timeout_seconds)
            or overall_timeout_seconds <= 0
        ):
            raise ValueError("overall_timeout_seconds must be finite and positive")
        self._client = client
        self._model = model
        self._overall_timeout_seconds = overall_timeout_seconds
        self._translate_error = translate_error

    #执行一次完整的模型请求
    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._serialize_message(message) for message in request.messages],
            # 沿用已经通过真实 DeepSeek 实验的配置；V1 不携带推理历史。
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if not payload["messages"]:
            raise LLMProtocolError("LLM request requires messages")
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": deepcopy(tool.parameters),
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = "auto"

        try:
            # 覆盖 SDK 内部重试的整个 await；不覆盖调用方取消信号。
            async with asyncio.timeout(self._overall_timeout_seconds):
                sdk_response = await self._client.chat.completions.create(**payload)
        except (TimeoutError, APIError) as exc:
            logger.warning("agent_llm.external_error", extra={"error_type": type(exc).__name__})
            raise self._translate_error(exc) from exc

        response = self._convert_response(sdk_response)
        logger.info(
            "agent_llm.response",
            extra={"finish_reason": response.finish_reason, "tool_count": len(response.tool_calls)},
        )
        return response

    #把内部 LLMMessage 转换成 SDK 消息格式
    @staticmethod
    def _serialize_message(message: LLMMessage) -> dict[str, Any]:
        if message.role not in ("system", "user", "assistant", "tool"):
            raise LLMProtocolError("Unsupported message role")
        if message.content is not None and not isinstance(message.content, str):
            raise LLMProtocolError("Message content must be text or None")
        if message.role != "assistant" and message.tool_calls:
            raise LLMProtocolError("Only assistant messages may contain tool calls")
        if message.role != "tool" and message.tool_call_id is not None:
            raise LLMProtocolError("Only tool messages may contain tool_call_id")
        if message.role != "assistant" and message.content is None:
            raise LLMProtocolError("Non-assistant messages require text content")
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "tool":
            if not isinstance(message.tool_call_id, str) or not message.tool_call_id.strip():
                raise LLMProtocolError("Tool message requires tool_call_id")
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            validate_tool_calls(message.tool_calls)
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments},
                }
                for call in message.tool_calls
            ]
        return payload

    #把 SDK 原始响应转换成 LLMResponse
    @staticmethod
    def _convert_response(response: ChatCompletion) -> LLMResponse:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or len(choices) != 1:
            raise LLMProtocolError("Expected exactly one completion choice")
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise LLMResponseTruncatedError("LLM response was truncated")
        message = getattr(choice, "message", None)
        if message is None or getattr(message, "role", None) != "assistant":
            raise LLMProtocolError("Expected an assistant response message")
        sdk_calls = getattr(message, "tool_calls", None)
        if sdk_calls is None:
            sdk_calls = []
        if not isinstance(sdk_calls, list):
            raise LLMProtocolError("tool_calls must be a list")
        calls: list[LLMToolCall] = []
        for call in sdk_calls:
            if getattr(call, "type", None) != "function":
                raise LLMProtocolError("Only function tool calls are supported")
            function = getattr(call, "function", None)
            raw_arguments = getattr(function, "arguments", None)
            if not isinstance(raw_arguments, str):
                raise LLMProtocolError("Raw tool arguments must be a string")
            arguments = None
            parse_error = None
            try:
                parsed = json.loads(raw_arguments, parse_constant=_reject_json_constant)
                if not isinstance(parsed, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                # 同时拒绝 1e999 这类被 Python 解为无穷大的数值。
                json.dumps(parsed, allow_nan=False)
                arguments = parsed
            except ValueError:
                parse_error = "工具参数不是合法的 JSON 对象"
            calls.append(
                LLMToolCall(
                    id=getattr(call, "id", None),
                    name=getattr(function, "name", None),
                    raw_arguments=raw_arguments,
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )
        result = LLMResponse(
            content=getattr(message, "content", None),
            tool_calls=calls,
            finish_reason=finish_reason,
        )
        validate_llm_response(result)
        return result
