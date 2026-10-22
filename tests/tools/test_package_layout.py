"""Tool 包迁移后的导入路径、类型身份与组件接线回归。"""

import asyncio
from importlib import import_module
from typing import Any

import pytest
from pydantic import BaseModel


@pytest.mark.parametrize(
    "module_name",
    [
        "backend.tools.infrastructure.contracts",
        "backend.tools.infrastructure.registry",
        "backend.tools.infrastructure.dispatcher",
        "backend.tools.infrastructure.errors",
        "backend.tools.implementations.weather",
    ],
)
def test_tool_modules_import_from_new_packages(module_name: str) -> None:
    module = import_module(module_name)

    assert module.__name__ == module_name


def test_infrastructure_modules_share_types_and_can_dispatch() -> None:
    contracts = import_module("backend.tools.infrastructure.contracts")
    registry_module = import_module("backend.tools.infrastructure.registry")
    dispatcher_module = import_module("backend.tools.infrastructure.dispatcher")
    errors = import_module("backend.tools.infrastructure.errors")

    assert registry_module.RegisteredTool is contracts.RegisteredTool
    assert dispatcher_module.ToolRegistry is registry_module.ToolRegistry
    assert dispatcher_module.ToolContext is contracts.ToolContext
    assert dispatcher_module.ToolSuccess is contracts.ToolSuccess
    assert registry_module.ToolNotFoundError is errors.ToolNotFoundError
    assert dispatcher_module.ToolNotFoundError is errors.ToolNotFoundError
    assert dispatcher_module.ToolExecutionError is errors.ToolExecutionError

    class EchoArgs(BaseModel):
        message: str

    context = contracts.ToolContext(user_id=42, conversation_id=7)

    async def echo_handler(
        validated_args: BaseModel,
        received_context: contracts.ToolContext,
    ) -> dict[str, Any]:
        assert type(validated_args) is EchoArgs
        assert received_context is context
        return validated_args.model_dump()

    registry = registry_module.ToolRegistry()
    registry.register(
        contracts.RegisteredTool(
            name="echo",
            description="验证迁移后的组件接线",
            args_schema=EchoArgs,
            handler=echo_handler,
        )
    )
    dispatcher = dispatcher_module.ToolDispatcher(registry)

    result = asyncio.run(
        dispatcher.dispatch("echo", {"message": "package-layout"}, context)
    )

    assert type(result) is contracts.ToolSuccess
    assert result.data == {"message": "package-layout"}
