from unittest.mock import call

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from messages_graph import build_messages_graph


def test_messages_appends_reply_and_preserves_full_message(model_invoke):
    original = [
        SystemMessage(content="Answer briefly.", id="system-1"),
        HumanMessage(content="What is a reducer?", id="human-1"),
    ]
    reply = AIMessage(
        content="A reducer merges updates.",
        id="ai-1",
        response_metadata={"source": "fake"},
        usage_metadata={"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
    )
    model = GenericFakeChatModel(messages=iter([reply]), cache=False)
    graph = build_messages_graph(model)
    model_invoke.assert_not_called()

    result = graph.invoke({"messages": original})

    model_invoke.assert_called_once_with(model, original)
    assert len(original) == 2
    assert result["messages"] == [*original, reply]
    assert isinstance(result["messages"][1], HumanMessage)
    assert isinstance(result["messages"][2], AIMessage)


def test_messages_reducer_replaces_existing_message_with_same_id():
    question = HumanMessage(content="How does merging work?", id="human-1")
    previous = AIMessage(content="Draft answer", id="ai-1")
    replacement = AIMessage(content="Revised answer", id="ai-1")
    model = GenericFakeChatModel(messages=iter([replacement]), cache=False)
    graph = build_messages_graph(model)

    result = graph.invoke({"messages": [question, previous]})

    assert result["messages"] == [question, replacement]


def test_independent_invokes_have_fresh_state_but_share_model(model_invoke):
    first_question = HumanMessage(content="First question", id="human-1")
    second_question = HumanMessage(content="Second question", id="human-2")
    first_reply = AIMessage(content="First answer", id="ai-1")
    second_reply = AIMessage(content="Second answer", id="ai-2")
    model = GenericFakeChatModel(messages=iter([first_reply, second_reply]), cache=False)
    graph = build_messages_graph(model)

    first = graph.invoke({"messages": [first_question]})
    second = graph.invoke({"messages": [second_question]})

    assert first["messages"] == [first_question, first_reply]
    assert second["messages"] == [second_question, second_reply]
    assert model_invoke.call_args_list == [
        call(model, [first_question]),
        call(model, [second_question]),
    ]
