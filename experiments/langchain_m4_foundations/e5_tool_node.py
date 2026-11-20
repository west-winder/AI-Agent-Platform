from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from e4_tool_calling import get_weather


def execute_calls(message: AIMessage, node: ToolNode) -> list[ToolMessage]:
    # 图通常注入 Runtime。单独调用节点时，用公开的 Runtime 类型显式传入。
    return node.invoke({"messages": [message]}, runtime=Runtime())["messages"]


if __name__ == "__main__":
    valid_shape = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {"city": "Beijing"}, "id": "beijing"},
            {"name": "get_weather", "args": {"city": "Shanghai"}, "id": "shanghai"},
            {"name": "get_weather", "args": {"city": "Unknown"}, "id": "unknown-city"},
        ],
    )
    invalid_call = AIMessage(
        content="",
        tool_calls=[{"name": "get_weather", "args": {}, "id": "missing-argument"}],
    )
    # 显式捕获 ValueError 会替换默认策略，缺参用默认策略单独演示。
    for result in [
        *execute_calls(valid_shape, ToolNode([get_weather], handle_tool_errors=ValueError)),
        *execute_calls(invalid_call, ToolNode([get_weather])),
    ]:
        print(type(result).__name__, result.tool_call_id, result.status, result.content)
