"""Dispatcher V1 的确定性测试；使用项目现有的 asyncio.run 测试方式。"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field, ValidationError

from backend.tools.infrastructure.contracts import (
    RegisteredTool,
    ToolContext,
    ToolErrorInfo,
    ToolFailure,
    ToolHandler,
    ToolResult,
    ToolSuccess,
)
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.errors import ToolExecutionError, ToolNotFoundError
from backend.tools.infrastructure.registry import ToolRegistry


class WeatherArgs(BaseModel):
    city: str = Field(min_length=1)
    days: int = Field(default=1, ge=1, le=7)


async def fake_weather_handler(
    validated_args: BaseModel,
    context: ToolContext,
) -> dict[str, Any]:
    return {**validated_args.model_dump(), "weather": "晴（模拟数据）"}


def make_dispatcher(handler: ToolHandler) -> ToolDispatcher:
    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            name="get_weather",
            description="查询天气（模拟）",
            args_schema=WeatherArgs,
            handler=handler,
        )
    )
    return ToolDispatcher(registry)


@pytest.fixture
def handler() -> AsyncMock:
    return AsyncMock(spec=fake_weather_handler, wraps=fake_weather_handler)


@pytest.fixture
def dispatcher(handler: AsyncMock) -> ToolDispatcher:
    return make_dispatcher(handler)


def assert_failure(
    result: ToolResult,
    *,
    code: str,
    message: str,
    retryable: bool,
) -> None:
    assert isinstance(result, ToolFailure)
    assert result.status == "error"
    assert result.data is None
    assert isinstance(result.error, ToolErrorInfo)
    assert result.model_dump() == {
        "status": "error",
        "data": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }


def test_dispatch_returns_success_and_awaits_handler_once(
    dispatcher: ToolDispatcher,
    handler: AsyncMock,
) -> None:
    result = asyncio.run(
        dispatcher.dispatch("get_weather", {"city": "上海"}, ToolContext())
    )

    assert isinstance(result, ToolSuccess)
    assert result.status == "success"
    assert result.data == {
        "city": "上海",
        "days": 1,
        "weather": "晴（模拟数据）",
    }
    assert result.error is None
    assert result.model_dump() == {
        "status": "success",
        "data": result.data,
        "error": None,
    }
    handler.assert_called_once()
    handler.assert_awaited_once()


def test_unknown_tool_returns_failure_without_executing_handler(
    dispatcher: ToolDispatcher,
    handler: AsyncMock,
) -> None:
    result = asyncio.run(
        dispatcher.dispatch("missing", {"city": "上海"}, ToolContext())
    )

    assert_failure(
        result,
        code="TOOL_NOT_FOUND",
        message="请求的工具不存在",
        retryable=False,
    )
    handler.assert_not_called()
    handler.assert_not_awaited()


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({}, id="missing-required-field"),
        pytest.param({"city": {"invalid": "city"}}, id="invalid-field-type"),
        pytest.param({"city": "上海", "days": 8}, id="field-constraint"),
    ],
)
def test_invalid_arguments_return_failure_without_executing_handler(
    dispatcher: ToolDispatcher,
    handler: AsyncMock,
    arguments: dict[str, Any],
) -> None:
    result = asyncio.run(
        dispatcher.dispatch("get_weather", arguments, ToolContext())
    )

    assert_failure(
        result,
        code="INVALID_ARGUMENTS",
        message="工具参数不符合 Schema 约束",
        retryable=False,
    )
    handler.assert_not_called()
    handler.assert_not_awaited()


def test_handler_receives_validated_schema_and_preserves_business_values(
    dispatcher: ToolDispatcher,
    handler: AsyncMock,
) -> None:
    arguments = {"city": "上海", "days": "2"}
    context = ToolContext()

    result = asyncio.run(dispatcher.dispatch("get_weather", arguments, context))

    handler.assert_called_once()
    handler.assert_awaited_once()
    validated_args, received_context = handler.await_args.args
    assert isinstance(validated_args, WeatherArgs)
    assert validated_args is not arguments
    assert validated_args.model_dump() == {"city": "上海", "days": 2}
    assert received_context is context
    assert arguments == {"city": "上海", "days": "2"}
    assert isinstance(result, ToolSuccess)
    assert result.data == {
        "city": "上海",
        "days": 2,
        "weather": "晴（模拟数据）",
    }


def test_dispatch_passes_trusted_context_unchanged(
    dispatcher: ToolDispatcher,
    handler: AsyncMock,
) -> None:
    context = ToolContext(user_id=42, conversation_id=7)
    arguments = {
        "city": "上海",
        "days": 3,
        "user_id": 999,
        "conversation_id": 888,
        "context": {"user_id": 999, "conversation_id": 888},
    }

    result = asyncio.run(dispatcher.dispatch("get_weather", arguments, context))

    assert isinstance(result, ToolSuccess)
    handler.assert_called_once()
    handler.assert_awaited_once()
    validated_args, received_context = handler.await_args.args
    assert isinstance(validated_args, WeatherArgs)
    assert validated_args.model_dump() == {"city": "上海", "days": 3}
    assert received_context is context
    assert received_context.user_id == 42
    assert received_context.conversation_id == 7
    assert received_context.db is None


@pytest.mark.parametrize("retryable", [False, True], ids=["not-retryable", "retryable"])
def test_known_execution_error_becomes_failure_without_retry(
    retryable: bool,
) -> None:
    error = ToolExecutionError(
        code="WEATHER_UNAVAILABLE",
        message="天气服务暂时不可用",
        retryable=retryable,
    )

    async def failing_handler(
        validated_args: BaseModel,
        context: ToolContext,
    ) -> dict[str, Any]:
        raise error

    handler = AsyncMock(spec=failing_handler, wraps=failing_handler)
    dispatcher = make_dispatcher(handler)

    result = asyncio.run(
        dispatcher.dispatch("get_weather", {"city": "上海"}, ToolContext())
    )

    assert_failure(
        result,
        code="WEATHER_UNAVAILABLE",
        message="天气服务暂时不可用",
        retryable=retryable,
    )
    handler.assert_called_once()
    handler.assert_awaited_once()


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(NameError("missing_local"), id="name-error"),
        pytest.param(RuntimeError("unexpected failure"), id="runtime-error"),
    ],
)
def test_unexpected_handler_error_propagates_without_retry(error: Exception) -> None:
    async def failing_handler(
        validated_args: BaseModel,
        context: ToolContext,
    ) -> dict[str, Any]:
        raise error

    handler = AsyncMock(spec=failing_handler, wraps=failing_handler)
    dispatcher = make_dispatcher(handler)

    with pytest.raises(type(error)) as raised:
        asyncio.run(
            dispatcher.dispatch("get_weather", {"city": "上海"}, ToolContext())
        )

    assert raised.value is error
    handler.assert_called_once()
    handler.assert_awaited_once()


@pytest.mark.parametrize(
    ("invalid_result", "cause_type"),
    [
        pytest.param({"opaque": object()}, TypeError, id="opaque-object"),
        pytest.param({"value": float("nan")}, ValueError, id="nan"),
    ],
)
def test_non_json_serializable_handler_result_raises_type_error(
    invalid_result: dict[str, Any],
    cause_type: type[Exception],
) -> None:
    async def invalid_handler(
        validated_args: BaseModel,
        context: ToolContext,
    ) -> dict[str, Any]:
        return invalid_result

    handler = AsyncMock(spec=invalid_handler, wraps=invalid_handler)
    dispatcher = make_dispatcher(handler)

    with pytest.raises(TypeError, match="JSON-serializable data") as raised:
        asyncio.run(
            dispatcher.dispatch("get_weather", {"city": "上海"}, ToolContext())
        )

    handler.assert_called_once()
    handler.assert_awaited_once()
    assert type(raised.value.__cause__) is cause_type
    assert raised.value.__cause__ is raised.value.__context__


@pytest.mark.parametrize(
    "invalid_result",
    [
        pytest.param("晴", id="string"),
        pytest.param(["晴"], id="list"),
    ],
)
def test_non_dict_handler_result_raises_type_error(invalid_result: Any) -> None:
    async def invalid_handler(
        validated_args: BaseModel,
        context: ToolContext,
    ) -> Any:
        return invalid_result

    handler = AsyncMock(spec=invalid_handler, wraps=invalid_handler)
    dispatcher = make_dispatcher(handler)

    with pytest.raises(TypeError, match="must return a dict"):
        asyncio.run(
            dispatcher.dispatch("get_weather", {"city": "上海"}, ToolContext())
        )

    handler.assert_called_once()
    handler.assert_awaited_once()


@pytest.mark.parametrize("error_origin", ["lookup", "validation"])
def test_handler_errors_are_not_misclassified_as_dispatch_input_errors(
    error_origin: str,
) -> None:
    # 相同异常类型发生在 Handler 内时，不应被归类为入口查找或参数校验失败。
    if error_origin == "lookup":
        error = ToolNotFoundError("Handler 内部查找失败")
    else:
        with pytest.raises(ValidationError) as invalid_args:
            WeatherArgs.model_validate({})
        error = invalid_args.value

    async def failing_handler(
        validated_args: BaseModel,
        context: ToolContext,
    ) -> dict[str, Any]:
        raise error

    handler = AsyncMock(spec=failing_handler, wraps=failing_handler)
    dispatcher = make_dispatcher(handler)

    with pytest.raises(type(error)) as raised:
        asyncio.run(
            dispatcher.dispatch("get_weather", {"city": "上海"}, ToolContext())
        )

    assert raised.value is error
    handler.assert_called_once()
    handler.assert_awaited_once()
