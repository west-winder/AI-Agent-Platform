from unittest.mock import call

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START

from basic_qa_graph import build_basic_qa_graph
from messages_graph import build_messages_graph


def test_basic_qa_compiles_without_calling_model_and_merges_answer(model_invoke):
    model = GenericFakeChatModel(messages=iter(["The node receives state."]), cache=False)
    graph = build_basic_qa_graph(model)
    model_invoke.assert_not_called()

    question = "What does the node receive?"
    result = graph.invoke({"question": question})

    assert result == {"question": question, "answer": "The node receives state."}
    assert type(result["answer"]) is str
    model_invoke.assert_called_once_with(model, question)


def test_basic_qa_node_only_returns_the_answer():
    model = GenericFakeChatModel(messages=iter(["New answer"]), cache=False)
    topology = build_basic_qa_graph(model).get_graph()
    assert {(edge.source, edge.target) for edge in topology.edges} == {
        (START, "LLM"),
        ("LLM", END),
    }
    state = {"question": "What changed?", "answer": "Old answer"}
    assert topology.nodes["LLM"].data.invoke(state) == {"answer": "New answer"}
    assert state == {"question": "What changed?", "answer": "Old answer"}


def test_basic_qa_uses_the_model_injected_into_each_graph(model_invoke):
    first_model = GenericFakeChatModel(messages=iter(["First reply"]), cache=False)
    second_model = GenericFakeChatModel(messages=iter(["Second reply"]), cache=False)
    first_graph = build_basic_qa_graph(first_model)
    second_graph = build_basic_qa_graph(second_model)

    question = "Which model is bound?"
    assert first_graph.invoke({"question": question})["answer"] == "First reply"
    assert second_graph.invoke({"question": question})["answer"] == "Second reply"
    assert model_invoke.call_args_list == [call(first_model, question), call(second_model, question)]


def test_basic_qa_requires_plain_text_content():
    response = AIMessage(content=[{"type": "text", "text": "A content block"}])
    graph = build_basic_qa_graph(GenericFakeChatModel(messages=iter([response]), cache=False))

    with pytest.raises(TypeError, match="requires plain text"):
        graph.invoke({"question": "Return a content block."})


@pytest.mark.parametrize(
    ("build_graph", "state"),
    [
        (build_basic_qa_graph, {"question": "Hello"}),
        (build_messages_graph, {"messages": [HumanMessage(content="Hello", id="human-1")]}),
    ],
)
def test_model_error_propagates_without_retry(build_graph, state, model_invoke):
    error = RuntimeError("Model failed")
    model_invoke.side_effect = error
    model = GenericFakeChatModel(messages=iter([]), cache=False)
    graph = build_graph(model)

    with pytest.raises(RuntimeError, match="Model failed") as raised:
        graph.invoke(state)

    assert raised.value is error
    assert model_invoke.call_count == 1
