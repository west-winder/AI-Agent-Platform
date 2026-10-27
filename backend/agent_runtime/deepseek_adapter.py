"""DeepSeek Function Calling 适配；客户端的生命周期、HTTP 超时和重试归上游。"""

import asyncio
from collections.abc import Callable
from copy import deepcopy
import json
import logging
import math
from typing import Any

from openai import APIError, AsyncOpenAI
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
from backend.exceptions.external_exceptions import ExternalServiceError

logger = logging.getLogger(__name__)
ErrorTranslator = Callable[[TimeoutError | APIError], ExternalServiceError]


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
