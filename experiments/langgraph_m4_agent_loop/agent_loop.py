from typing import Literal, NotRequired, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode


MAX_LLM_CALLS = 3
RECURSION_LIMIT = 8
SKIPPED_TOOL = "因预算限制未执行"
WEATHER = {"Beijing": "sunny", "Shanghai": "cloudy"}

INSTRUCTIONS = SystemMessage(
    content=(
        "Use get_weather for simulated weather lookups. Report weather only when the tool "
        "returns a result. If a lookup fails or was skipped, explain which request is unresolved."
    )
)


class AgentState(MessagesState):
    llm_calls: NotRequired[int]
    final_answer: NotRequired[str]
    stop_reason: NotRequired[Literal["answered", "llm_limit"]]


@tool
def get_weather(city: str) -> str:
    """Look up simulated weather for a city by its exact name."""
    if city not in WEATHER:
        raise ValueError(f"No simulated weather for {city}")
    return WEATHER[city]


def route_after_llm(state: AgentState) -> Literal["ToolNode", "Finish"]:
    reply = cast(AIMessage, state["messages"][-1])
    if reply.tool_calls and state["llm_calls"] < MAX_LLM_CALLS:
        return "ToolNode"
    return "Finish"


def finish(state: AgentState) -> dict:
    """根据最后一轮模型输出和本次执行的工具结果组装最终状态。"""
    reply = cast(AIMessage, state["messages"][-1])
    # 有待执行请求却进入 Finish，说明路由已耗尽模型调用预算。
    skipped = [
        ToolMessage(
            name=call["name"], tool_call_id=call["id"], content=SKIPPED_TOOL, status="error"
        )
        for call in reply.tool_calls
    ]
    if skipped:
        lines = [f"任务未完成，已达到 {MAX_LLM_CALLS} 次模型调用上限。"]
    else:
        lines = [str(reply.text).strip() or "模型未提供文本回答。"]

    # 倒数本次生成的 AIMessage，限定结果摘要的范围，排除输入历史里的工具结果。
    remaining_replies = state["llm_calls"]
    tool_results = []
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            remaining_replies -= 1
            if remaining_replies == 0:
                break
        elif isinstance(message, ToolMessage):
            tool_results.append(message)

    for result in reversed(tool_results):
        label = "成功结果" if result.status == "success" else "工具失败"
        lines.append(f"{label} [{result.tool_call_id}] {result.text}")
    for pending in skipped:
        lines.append(f"未完成 [{pending.tool_call_id}] {pending.content}")

    return {
        "messages": skipped,
        "final_answer": "\n".join(lines),
        "stop_reason": "llm_limit" if skipped else "answered",
    }


def build_agent_graph(model: BaseChatModel) -> CompiledStateGraph:
    """在图构造时绑定工具，返回接受 messages 的编译图。"""
    tools = [get_weather]
    model_with_tools = model.bind_tools(tools)

    def llm_node(state: AgentState) -> dict:
        # 闭包持有已经绑定工具的模型，State 只保存本轮输入和执行结果。
        reply = model_with_tools.invoke([INSTRUCTIONS, *state["messages"]])
        return {"messages": [reply], "llm_calls": state.get("llm_calls", 0) + 1}

    builder = StateGraph(AgentState, input_schema=MessagesState)
    builder.add_node("LLM", llm_node)
    builder.add_node("ToolNode", ToolNode(tools, handle_tool_errors=ValueError))
    builder.add_node("Finish", finish)
    builder.add_edge(START, "LLM")
    builder.add_conditional_edges("LLM", route_after_llm)
    builder.add_edge("ToolNode", "LLM")
    builder.add_edge("Finish", END)
    return builder.compile().with_config(recursion_limit=RECURSION_LIMIT)
