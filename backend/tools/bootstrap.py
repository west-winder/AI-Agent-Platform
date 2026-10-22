
from backend.tools.infrastructure.contracts import (
    RegisteredTool,
)
from backend.tools.infrastructure.registry import (
    ToolRegistry,
)
from backend.tools.implementations.weather import (
    WeatherArgs,
    get_weather_handler,
)


def build_tool_registry() -> ToolRegistry:
    """创建并配置本系统的工具注册表。"""

    registry = ToolRegistry()

    # 注册 Weather Tool
    registry.register(
        RegisteredTool(
            name="get_weather",
            description="查询指定城市的当前天气",
            args_schema=WeatherArgs,
            handler=get_weather_handler,
        )
    )

    return registry
