from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langsmith import tracing_context

from basic_qa_graph import build_basic_qa_graph
from messages_graph import build_messages_graph


def main() -> None:
    model = FakeListChatModel(
        responses=["A node receives state.", "A reducer merges updates into state."],
        cache=False,
    )
    qa_graph = build_basic_qa_graph(model)
    messages_graph = build_messages_graph(model)

    # 两张图共享同一个假模型，依次消费它的两条预设回复。
    with tracing_context(enabled=False):
        qa = qa_graph.invoke({"question": "What does a node receive?"})
        conversation = messages_graph.invoke(
            {"messages": [HumanMessage(content="What does a reducer do?")]}
        )

    print("E1 Basic QA")
    print(f"  question ({type(qa['question']).__name__}): {qa['question']}")
    print(f"  answer   ({type(qa['answer']).__name__}): {qa['answer']}")
    print("E2 MessagesState")
    for message in conversation["messages"]:
        print(f"  {type(message).__name__}: {message.content}")


if __name__ == "__main__":
    main()
