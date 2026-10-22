
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session


@dataclass(frozen=True, slots=True)
class ToolContext:
    """由可信的上游 Runtime 提供工具执行上下文。"""

    user_id: int | None = None
    conversation_id: int | None = None
    db: Session | None = None


# 统一 Handler 签名：
# async handler(validated_args, context) -> dict
ToolHandler = Callable[
    [BaseModel, ToolContext],
    Awaitable[dict[str, Any]],
]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """工具注册记录，不负责执行工具。"""

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler



class ToolErrorInfo(BaseModel):
    """工具执行失败时的结构化错误信息。"""

    code: str
    message: str
    retryable: bool = False


class ToolSuccess(BaseModel):
    """工具执行成功。"""

    status: Literal["success"] = "success"
    data: dict[str, Any]
    error: None = None


class ToolFailure(BaseModel):
    """工具执行失败。"""

    status: Literal["error"] = "error"
    data: None = None
    error: ToolErrorInfo


ToolResult = ToolSuccess | ToolFailure
