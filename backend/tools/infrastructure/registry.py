
import inspect

from backend.tools.infrastructure.contracts import RegisteredTool
from backend.tools.infrastructure.errors import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistryError,
)


class ToolRegistry:
    """工具注册表：负责注册、查找和列出工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """注册工具，拒绝重复名称和同步 Handler。"""

        # 1. 拒绝重复注册
        if tool.name in self._tools:
            raise DuplicateToolError(
                f"Tool already registered: {tool.name}"
            )

        # 2. V1 仅接受 async def Handler
        if not inspect.iscoroutinefunction(tool.handler):
            raise ToolRegistryError(
                f"Tool handler must be async: {tool.name}"
            )

        # 3. 保存工具注册记录
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool:
        """通过工具名称查找注册记录。"""

        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool not found: {name}"
            )

        return self._tools[name]

    def list_tools(self) -> list[RegisteredTool]:
        """返回已注册的工具列表。"""

        return list(self._tools.values())
