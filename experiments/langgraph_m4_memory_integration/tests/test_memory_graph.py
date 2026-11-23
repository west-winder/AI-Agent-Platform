import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START
from pydantic import ValidationError

from fake_memory import FakeMemoryService
from fake_model import RecordingChatModel, decision_reply
from memory_graph import MemoryDecision, build_memory_graph


PREFERENCE = "I always prefer short Python examples."


def make_graph(service, answers, decisions):
    answer_model = RecordingChatModel(messages=iter(answers), cache=False)
    extraction_model = RecordingChatModel(messages=iter(decisions), cache=False)
    return build_memory_graph(service, answer_model, extraction_model), answer_model, extraction_model


def test_relevant_memory_enters_answer_prompt_only(monkeypatch):
    memory = "I use Python for teaching."
    service = FakeMemoryService({1: [memory, "I play chess."], 2: ["I use Python for games."]})
    load = AsyncMock(wraps=service.load_context)
    monkeypatch.setattr(service, "load_context", load)
    query = "Could you suggest Python exercises?"
    graph, answer_model, extractor = make_graph(
        service, ["Here are three exercises."], [decision_reply(False, None)]
    )
    assert {(edge.source, edge.target) for edge in graph.get_graph().edges} == {
        (START, "LoadMemory"), ("LoadMemory", "Answer"), ("Answer", "ExtractMemory"),
        ("ExtractMemory", "SaveMemory"), ("SaveMemory", END),
    }

    state = asyncio.run(graph.ainvoke({"user_id": 1, "query": query}))

    load.assert_awaited_once_with(user_id=1, query=query)
    (answer_prompt,) = answer_model.requests
    assert isinstance(answer_prompt[0], SystemMessage)
    assert json.loads(answer_prompt[1].content) == {"query": query, "memory_context": memory}
    (extraction_prompt,) = extractor.requests
    assert isinstance(extraction_prompt[0], SystemMessage)
    assert extraction_prompt[1:] == [HumanMessage(content=query)]
    assert state == {
        "user_id": 1, "query": query, "memory_context": memory, "answer": "Here are three exercises.",
        "memory_decision": MemoryDecision(should_save=False, content=None), "memory_saved": False,
    }


def test_long_term_preference_writes_once_and_is_available_to_owner(monkeypatch):
    service = FakeMemoryService()
    save = AsyncMock(wraps=service.save_memory)
    monkeypatch.setattr(service, "save_memory", save)
    query = f"By the way, {PREFERENCE} Can you show one?"
    graph, _, _ = make_graph(service, ["Sure."], [decision_reply(True, PREFERENCE)])

    async def run():
        state = await graph.ainvoke({"user_id": 1, "query": query})
        assert state["memory_context"] == ""
        assert state["memory_decision"] == MemoryDecision(should_save=True, content=PREFERENCE)
        assert state["memory_saved"] is True
        save.assert_awaited_once_with(user_id=1, content=PREFERENCE)
        assert await service.load_context(1, "PYTHON") == PREFERENCE
        assert await service.load_context(2, "PYTHON") == ""

    asyncio.run(run())


def test_small_talk_does_not_save_assistant_generated_fact(monkeypatch):
    service = FakeMemoryService()
    save = AsyncMock(wraps=service.save_memory)
    monkeypatch.setattr(service, "save_memory", save)
    graph, _, extractor = make_graph(service, [PREFERENCE], [decision_reply(False, None)])
    state = asyncio.run(graph.ainvoke({"user_id": 1, "query": "Hello!"}))

    assert state["answer"] == PREFERENCE
    assert extractor.requests[0][1:] == [HumanMessage(content="Hello!")]
    assert state["memory_decision"] == MemoryDecision(should_save=False, content=None)
    assert state["memory_saved"] is False
    save.assert_not_called()


def test_write_failure_reaches_caller_without_saving(monkeypatch):
    service = FakeMemoryService()
    error = RuntimeError("memory write failed")
    save = AsyncMock(side_effect=error)
    monkeypatch.setattr(service, "save_memory", save)
    graph, _, _ = make_graph(service, ["Sure."], [decision_reply(True, PREFERENCE)])

    async def run():
        with pytest.raises(RuntimeError, match="memory write failed") as raised:
            await graph.ainvoke({"user_id": 1, "query": PREFERENCE})
        assert raised.value is error
        save.assert_awaited_once_with(user_id=1, content=PREFERENCE)
        assert await service.load_context(1, "Python") == ""

    asyncio.run(run())


def test_same_compiled_graph_keeps_user_context_and_state_separate():
    service = FakeMemoryService({2: ["I use Python with FastAPI."]})
    graph, answer_model, _ = make_graph(
        service,
        ["Got it.", "Here is a web app.", "Here is a short example."],
        [decision_reply(True, PREFERENCE), decision_reply(False, None), decision_reply(False, None)],
    )

    async def run():
        first = await graph.ainvoke({"user_id": 1, "query": PREFERENCE})
        other = await graph.ainvoke({"user_id": 2, "query": "Show me Python."})
        same = await graph.ainvoke({"user_id": 1, "query": "Show me Python."})

        assert first["memory_context"] == "" and first["memory_saved"] is True
        assert other["memory_context"] == "I use Python with FastAPI."
        assert same["memory_context"] == PREFERENCE
        assert other["answer"] == "Here is a web app."
        assert same["answer"] == "Here is a short example."
        for state, prompt in zip((other, same), answer_model.requests[1:], strict=True):
            assert json.loads(prompt[1].content)["memory_context"] == state["memory_context"]
            assert state["memory_decision"] == MemoryDecision(should_save=False, content=None)
            assert state["memory_saved"] is False

    asyncio.run(run())


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "nonblank"), ("  ", "nonblank"), ("I prefer Rust.", "verbatim excerpt")],
)
def test_invalid_save_candidate_never_calls_writer(monkeypatch, content, message):
    service = FakeMemoryService()
    save = AsyncMock(wraps=service.save_memory)
    monkeypatch.setattr(service, "save_memory", save)
    graph, _, _ = make_graph(service, ["I prefer Rust."], [decision_reply(True, content)])

    with pytest.raises(ValueError, match=message):
        asyncio.run(graph.ainvoke({"user_id": 1, "query": PREFERENCE}))

    save.assert_not_called()


def test_structured_output_rejects_non_boolean_decision(monkeypatch):
    service = FakeMemoryService()
    save = AsyncMock(wraps=service.save_memory)
    monkeypatch.setattr(service, "save_memory", save)
    malformed = AIMessage(content="", tool_calls=[{
        "name": "MemoryDecision", "args": {"should_save": "true", "content": PREFERENCE}, "id": "bad-decision",
    }])
    graph, _, _ = make_graph(service, ["Got it."], [malformed])

    with pytest.raises(ValidationError, match="should_save"):
        asyncio.run(graph.ainvoke({"user_id": 1, "query": PREFERENCE}))

    save.assert_not_called()
