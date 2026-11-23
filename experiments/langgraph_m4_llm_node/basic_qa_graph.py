from typing import NotRequired, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


class BasicQAState(TypedDict):
    question: str
    answer: NotRequired[str]


def build_basic_qa_graph(model: BaseChatModel) -> CompiledStateGraph:
    """接收已创建的模型对象，返回可用 question 启动的编译图。"""

    def llm_node(state: BasicQAState) -> dict[str, str]:
        # 闭包引用外层的 model。LangGraph 执行节点时才传入 state。
        response = model.invoke(state["question"])
        if not isinstance(response.content, str):
            raise TypeError("Basic QA requires plain text model content")
        # 只更新 answer，原有 question 由 LangGraph 保留。
        return {"answer": response.content}

    builder = StateGraph(BasicQAState)
    builder.add_node("LLM", llm_node)
    builder.add_edge(START, "LLM")
    builder.add_edge("LLM", END)
    return builder.compile()
