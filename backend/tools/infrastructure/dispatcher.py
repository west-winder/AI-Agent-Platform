import json
from typing import Any

from pydantic import ValidationError

from backend.tools.infrastructure.contracts import (
    ToolContext,
    ToolErrorInfo,
    ToolFailure,
    ToolResult,
    ToolSuccess,
)
from backend.tools.infrastructure.errors import (
    ToolExecutionError,
    ToolNotFoundError,
)
from backend.tools.infrastructure.registry import ToolRegistry


class ToolDispatcher:
    """统一工具执行入口。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:

        # ======================================
        # 1. Tool Lookup
        # ======================================

        try:
            tool = self._registry.get(tool_name)

        except ToolNotFoundError:
            return ToolFailure(
                error=ToolErrorInfo(
                    code="TOOL_NOT_FOUND",
                    message="请求的工具不存在",
                    retryable=False,
                )
            )

        # ======================================
        # 2. Schema Validation
        # ======================================

        try:
            validated_args = tool.args_schema.model_validate(
                arguments
            )

        except ValidationError:
            return ToolFailure(
                error=ToolErrorInfo(
                    code="INVALID_ARGUMENTS",
                    message="工具参数不符合 Schema 约束",
                    retryable=False,
                )
            )

        # ======================================
        # 3. Handler Execution
        # ======================================

        try:
            result = await tool.handler(
                validated_args,
                context,
            )

        except ToolExecutionError as exc:
            return ToolFailure(
                error=ToolErrorInfo(
                    code=exc.code,
                    message=exc.message,
                    retryable=exc.retryable,
                )
            )

        # ======================================
        # 4. Result Validation
        # ======================================

        if not isinstance(result, dict):
            raise TypeError(
                "Tool handler must return a dict"
            )

        # 确保结果能够序列化为标准 JSON
        try:
            json.dumps(result, allow_nan=False)

        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Tool handler must return JSON-serializable data"
            ) from exc


        # ======================================
        # 5. Success Result
        # ======================================

        return ToolSuccess(data=result)
