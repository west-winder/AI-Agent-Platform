import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, StateSnapshot

from .mail_store import SQLiteMailStore
from .state import ApprovalRequest, EmailInput, ReviewAction
from .workflow import build_email_graph


SAMPLE: EmailInput = {"query": "讨论石墨烯样品的表征结果", "recipient": "materials@example.test"}


def disable_tracing() -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def review_command(request: ApprovalRequest, action: ReviewAction, feedback: str = "") -> Command:
    return Command(resume={
        "action": action,
        "request_id": request["request_id"],
        "draft_version": request["draft_version"],
        "feedback": feedback,
    })


def summarize(snapshot: StateSnapshot) -> dict:
    config = snapshot.config["configurable"]
    return {
        "thread_id": config["thread_id"],
        "checkpoint_id": config["checkpoint_id"],
        "step": (snapshot.metadata or {}).get("step"),
        "next": list(snapshot.next),
        "values": snapshot.values,
        "interrupts": [{"id": item.id, "value": item.value} for item in snapshot.interrupts],
    }


def main() -> None:
    disable_tracing()
    with TemporaryDirectory(prefix="m6-offline-") as directory:
        store = SQLiteMailStore(Path(directory) / "mail.sqlite3")
        store.setup()
        graph = build_email_graph(InMemorySaver(), store.send)
        config = thread_config("m6-demo-offline")
        start = graph.invoke(SAMPLE, config, version="v2", durability="sync")
        revised = graph.invoke(
            review_command(start.interrupts[0].value, "revise", "请附上会议议程"),
            config, version="v2", durability="sync",
        )
        final = graph.invoke(
            review_command(revised.interrupts[0].value, "approve"), config, version="v2", durability="sync",
        )
        print(json.dumps({
            "first_review": start.interrupts[0].value,
            "second_review": revised.interrupts[0].value,
            "final": summarize(graph.get_state(config)),
            "mail": store.evidence(final.value["idempotency_key"]),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
