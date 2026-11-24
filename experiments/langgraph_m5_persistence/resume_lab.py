import json
import os
from pathlib import Path
from threading import Event, Lock

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import StateSnapshot

from paper_graph import (
    PaperInput,
    PaperNode,
    PaperState,
    PaperUpdate,
    analyze_content,
    build_paper_graph,
    check_references,
    generate_report,
    initialize_run,
    parse_paper,
)


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def snapshot_summary(snapshot: StateSnapshot) -> dict:
    metadata = snapshot.metadata or {}
    configurable = snapshot.config["configurable"]
    return {
        "thread_id": configurable["thread_id"],
        "checkpoint_id": configurable["checkpoint_id"],
        "step": metadata.get("step"),
        "next": list(snapshot.next),
        "values": snapshot.values,
    }


class NodeAudit:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()

    def record(self, node: str, *, event: str = "attempt", **details: str | int) -> None:
        entry = {"pid": os.getpid(), "node": node, "event": event, **details}
        # 关闭文件后记录可由另一进程读取，锁保护同一进程内的并行分支。
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]


class ReferenceFailure(RuntimeError):
    """模拟首次并行引用检查失败。"""


class ReferenceFailureGate:
    def __init__(self):
        self.analysis_returned = Event()
        self.enabled = True

    def check(self, state: PaperState) -> PaperUpdate:
        if self.enabled:
            if not self.analysis_returned.wait(timeout=15):
                raise TimeoutError("没有收到 AnalyzeContent 的 Runtime 输出")
            raise ReferenceFailure("模拟 CheckReferences 首次执行失败")
        return check_references(state)


def observed_paper_graph(
    saver: BaseCheckpointSaver,
    audit: NodeAudit,
    gate: ReferenceFailureGate | None = None,
):
    def observe(name: str, node: PaperNode) -> PaperNode:
        def observed(state: PaperState) -> PaperUpdate:
            audit.record(name)
            return node(state)

        return observed

    return build_paper_graph(
        saver,
        node_overrides={
            "InitializeRun": observe("InitializeRun", initialize_run),
            "ParsePaper": observe("ParsePaper", parse_paper),
            "AnalyzeContent": observe("AnalyzeContent", analyze_content),
            "CheckReferences": observe("CheckReferences", gate.check if gate else check_references),
            "GenerateReport": observe("GenerateReport", generate_report),
        },
    )


def observe_failed_run(
    graph,
    saver: BaseCheckpointSaver,
    config: RunnableConfig,
    paper: PaperInput,
    gate: ReferenceFailureGate,
) -> dict:
    try:
        # 1.2.22 的同步 Runtime 在 custom 模式开启时会并发转发流事件。
        for part in graph.stream(
            paper, config, stream_mode=["updates", "custom"], version="v2", durability="sync"
        ):
            # Runtime 发出这条节点更新时，AnalyzeContent 已返回。
            if part["type"] == "updates" and "AnalyzeContent" in part["data"]:
                gate.analysis_returned.set()
    except ReferenceFailure as failure:
        error = str(failure)
    else:
        raise AssertionError("模拟异常应传播给 graph.stream 调用方")

    latest = graph.get_state(config)
    boundary = graph.get_state(latest.config)
    checkpoint = saver.get_tuple(boundary.config)
    assert checkpoint is not None
    tasks = {task.name: task for task in boundary.tasks}
    analysis = tasks["AnalyzeContent"]
    pending_analysis = [
        value for task_id, channel, value in checkpoint.pending_writes or ()
        if task_id == analysis.id and channel == "analysis"
    ]
    return {
        **snapshot_summary(boundary),
        "exception": error,
        "next_on_resume": list(latest.next),
        "analysis_task_result": analysis.result,
        "pending_analysis": pending_analysis,
        "reference_task_failed": tasks["CheckReferences"].error is not None,
    }
