from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph


def build_messages_graph(model: BaseChatModel) -> CompiledStateGraph:
    """接收已创建的模型对象，返回以消息列表为 State 的编译图。"""

    def llm_node(state: MessagesState) -> MessagesState:
        # model 来自外层函数。MessagesState 的 reducer 会合并这条完整回复。
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("LLM", llm_node)
    builder.add_edge(START, "LLM")
    builder.add_edge("LLM", END)
    return builder.compile()
