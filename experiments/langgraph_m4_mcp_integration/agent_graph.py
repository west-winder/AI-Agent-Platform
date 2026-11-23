from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def build_agent_graph(model: BaseChatModel, tools: Sequence[BaseTool]) -> CompiledStateGraph:
    """绑定调用方已经发现的工具，返回可用 ainvoke 执行的图。"""
    model_with_tools = model.bind_tools(tools)

    async def llm_node(state: MessagesState) -> MessagesState:
        reply = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(MessagesState)
    builder.add_node("LLM", llm_node)
    # MCPAdapter 的工具已负责 isError。基础设施异常继续向调用方传播。
    builder.add_node("ToolNode", ToolNode(tools, handle_tool_errors=False))
    builder.add_edge(START, "LLM")
    builder.add_conditional_edges("LLM", tools_condition, {"tools": "ToolNode", END: END})
    builder.add_edge("ToolNode", "LLM")
    return builder.compile().with_config(recursion_limit=8)
