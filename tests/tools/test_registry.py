"""正式 Tool Registry 单元测试，不执行真实工具或访问外部服务。"""

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from backend.tools.infrastructure.contracts import RegisteredTool, ToolContext
from backend.tools.infrastructure.errors import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistryError,
)
from backend.tools.infrastructure.registry import ToolRegistry


class WeatherArgs(BaseModel):
    city: str


async def get_weather(
    validated_args: BaseModel,
    context: ToolContext,
) -> dict[str, Any]:
    return {
        "city": validated_args.model_dump()["city"],
        "weather": "晴（模拟数据）",
    }


def make_weather_tool() -> RegisteredTool:
    return RegisteredTool(
        name="get_weather",
        description="查询天气（模拟）",
        args_schema=WeatherArgs,
        handler=get_weather,
    )


def test_empty_registry() -> None:
    assert ToolRegistry().list_tools() == []


def test_register_then_get() -> None:
    registry = ToolRegistry()
    tool = make_weather_tool()

    registry.register(tool)

    assert registry.get("get_weather") is tool
    assert registry.get("get_weather").handler is get_weather
    assert registry.get("get_weather").args_schema is WeatherArgs
    assert registry.list_tools() == [tool]


def test_duplicate_registration_fails_without_overwrite() -> None:
    registry = ToolRegistry()
    original = make_weather_tool()
    registry.register(original)
    original_tools = registry.list_tools()
    replacement = replace(original, description="不应覆盖原工具的描述")

    with pytest.raises(DuplicateToolError, match="get_weather"):
        registry.register(replacement)

    assert registry.get("get_weather") is original
    assert registry.list_tools() == original_tools


def test_missing_tool_has_explicit_error() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError, match="missing"):
        registry.get("missing")


def test_list_tools_returns_a_new_list() -> None:
    registry = ToolRegistry()
    tool = make_weather_tool()
    registry.register(tool)
    external_list = registry.list_tools()

    assert external_list is not registry.list_tools()
    external_list.clear()

    assert registry.list_tools() == [tool]
    assert registry.get("get_weather") is tool


def test_registry_does_not_execute_tool() -> None:
    # AsyncMock 同时记录调用和 await，避免漏掉创建协程但未 await 的误调用。
    handler = AsyncMock(spec=get_weather, wraps=get_weather)
    registry = ToolRegistry()
    tool = replace(make_weather_tool(), handler=handler)

    registry.register(tool)
    registry.get("get_weather")
    registry.list_tools()

    handler.assert_not_called()
    handler.assert_not_awaited()


@pytest.mark.parametrize("prepopulate", [False, True], ids=["empty", "populated"])
def test_sync_handler_registration_fails_without_state_change(
    prepopulate: bool,
) -> None:
    def sync_handler(
        validated_args: BaseModel,
        context: ToolContext,
    ) -> dict[str, Any]:
        return {"city": validated_args.model_dump()["city"]}

    registry = ToolRegistry()
    original = make_weather_tool()
    if prepopulate:
        registry.register(original)
    original_tools = registry.list_tools()
    invalid_tool = RegisteredTool(
        name="sync_weather",
        description="同步 Handler 必须被拒绝",
        args_schema=WeatherArgs,
        handler=sync_handler,  # 故意违反 ToolHandler 的异步契约。
    )

    with pytest.raises(ToolRegistryError, match="async.*sync_weather"):
        registry.register(invalid_tool)

    assert registry.list_tools() == original_tools
    if prepopulate:
        assert registry.get("get_weather") is original
    with pytest.raises(ToolNotFoundError, match="sync_weather"):
        registry.get("sync_weather")
