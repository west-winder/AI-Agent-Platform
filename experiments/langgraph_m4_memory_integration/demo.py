import asyncio

from langsmith import tracing_context

from fake_memory import FakeMemoryService
from fake_model import RecordingChatModel, decision_reply
from memory_graph import build_memory_graph


async def main() -> None:
    preference = "I always prefer short Python examples."
    service = FakeMemoryService({1: ["I use Python for notebooks."], 2: ["I use Python for web apps."]})
    answer_model = RecordingChatModel(
        messages=iter(["Got it.", "Let's use a web app example.", "Here is a short notebook example."]),
        cache=False,
    )
    extraction_model = RecordingChatModel(
        messages=iter([decision_reply(True, preference), decision_reply(False, None), decision_reply(False, None)]),
        cache=False,
    )
    graph = build_memory_graph(service, answer_model, extraction_model)

    with tracing_context(enabled=False):
        for user_id, query in [(1, preference), (2, "Show me Python."), (1, "Show me Python.")]:
            state = await graph.ainvoke({"user_id": user_id, "query": query})
            print(f"\nuser_id={user_id} query={query}")
            print("  memory_context:", repr(state["memory_context"]))
            print("  answer:", state["answer"])
            print("  decision:", state["memory_decision"].model_dump())
            print("  memory_saved:", state["memory_saved"])


if __name__ == "__main__":
    asyncio.run(main())
