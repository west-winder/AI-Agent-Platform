import socket

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START
from langgraph.types import StateSnapshot

from paper_graph import PaperInput, build_paper_graph


FIRST: PaperInput = {
    "run_id": "A-1",
    "raw_paper_content": "  石墨烯\n 导电 [ref:g1]  [ref:g2]  ",
}
SECOND: PaperInput = {"run_id": "A-2", "raw_paper_content": "月壤 含水"}
OTHER: PaperInput = {"run_id": "B-1", "raw_paper_content": "珊瑚礁 温度 [ref:b1]"}

FIRST_CONTENT = "石墨烯 导电 [ref:g1] [ref:g2]"
FIRST_ANALYSIS = f"内容摘要｜{FIRST_CONTENT}"
FIRST_REPORT = f"A-1｜{FIRST_ANALYSIS}｜引用 g1、g2"
SECOND_REPORT = "A-2｜内容摘要｜月壤 含水｜引用 无"
OTHER_REPORT = "B-1｜内容摘要｜珊瑚礁 温度 [ref:b1]｜引用 b1"

THREAD_A: RunnableConfig = {"configurable": {"thread_id": "A"}}
THREAD_B: RunnableConfig = {"configurable": {"thread_id": "B"}}


@pytest.fixture(autouse=True)
def forbid_external_network(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")

    def blocked(*args, **kwargs):
        raise AssertionError("M5 Part A 测试禁止外部网络调用")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


def checkpoint_step(snapshot: StateSnapshot) -> int:
    assert snapshot.metadata is not None
    return snapshot.metadata["step"]


def test_a1_invocations_without_checkpointer_are_independent():
    graph = build_paper_graph()
    first = graph.invoke(FIRST)
    second = graph.invoke(SECOND)

    assert first["completed_runs"] == second["completed_runs"] == 1
    assert first["report_history"] == [FIRST_REPORT]
    assert second["report_history"] == [SECOND_REPORT]


def test_a2_same_thread_continues_only_thread_scoped_state():
    graph = build_paper_graph(InMemorySaver())
    first = graph.invoke(FIRST, THREAD_A)
    second = graph.invoke(SECOND, THREAD_A)

    assert first["completed_runs"] == 1
    assert first["report_history"] == [FIRST_REPORT]
    assert first["references"] == ["g1", "g2"]
    assert second == {
        **SECOND,
        "paper_content": "月壤 含水",
        "analysis": "内容摘要｜月壤 含水",
        "references": [],
        "report": SECOND_REPORT,
        "completed_runs": 2,
        "report_history": [FIRST_REPORT, SECOND_REPORT],
    }


def test_a3_interleaved_threads_are_isolated():
    graph = build_paper_graph(InMemorySaver())
    graph.invoke(FIRST, THREAD_A)
    other = graph.invoke(OTHER, THREAD_B)
    other_snapshot = graph.get_state(THREAD_B)
    second = graph.invoke(SECOND, THREAD_A)

    assert graph.get_state(THREAD_A).values == second
    assert second["completed_runs"] == 2
    assert second["report_history"] == [FIRST_REPORT, SECOND_REPORT]
    assert graph.get_state(THREAD_B) == other_snapshot
    assert other_snapshot.values == other == {
        **OTHER,
        "paper_content": "珊瑚礁 温度 [ref:b1]",
        "analysis": "内容摘要｜珊瑚礁 温度 [ref:b1]",
        "references": ["b1"],
        "report": OTHER_REPORT,
        "completed_runs": 1,
        "report_history": [OTHER_REPORT],
    }
    assert all(
        snapshot.config["configurable"]["thread_id"] == "B"
        and snapshot.values.get("run_id") in (None, "B-1")
        for snapshot in graph.get_state_history(THREAD_B)
    )


def test_a4_initialize_run_clears_previous_values_before_parse():
    graph = build_paper_graph(InMemorySaver())
    first = graph.invoke(FIRST, THREAD_A)
    graph.invoke(SECOND, THREAD_A)
    second_run = [
        snapshot for snapshot in graph.get_state_history(THREAD_A)
        if snapshot.values.get("run_id") == SECOND["run_id"]
    ]
    before = next(snapshot for snapshot in second_run if snapshot.next == ("InitializeRun",))
    after = next(snapshot for snapshot in second_run if snapshot.next == ("ParsePaper",))

    # 输入已更新，但上一轮派生字段此时仍存在，随后由 InitializeRun 清空。
    assert before.values == {**first, **SECOND}
    assert after.parent_config == before.config
    assert checkpoint_step(after) == checkpoint_step(before) + 1
    assert after.values == {
        **SECOND,
        "paper_content": "",
        "analysis": "",
        "references": [],
        "report": "",
        "completed_runs": 1,
        "report_history": [FIRST_REPORT],
    }


def test_a5_get_state_returns_completed_snapshot():
    graph = build_paper_graph(InMemorySaver())
    result = graph.invoke(FIRST, THREAD_A)
    snapshot = graph.get_state(THREAD_A)

    assert isinstance(snapshot, StateSnapshot)
    assert snapshot.values == result
    assert snapshot.next == ()
    assert snapshot.tasks == ()
    configurable = snapshot.config["configurable"]
    assert configurable["thread_id"] == "A"
    assert configurable["checkpoint_ns"] == ""
    assert isinstance(configurable["checkpoint_id"], str) and configurable["checkpoint_id"]
    assert snapshot.metadata is not None
    assert snapshot.metadata["source"] == "loop"
    assert isinstance(checkpoint_step(snapshot), int)
    assert "thread_id" not in snapshot.values


def test_a6_history_is_newest_first_and_allows_checkpoint_lookup():
    graph = build_paper_graph(InMemorySaver())
    graph.invoke(FIRST, THREAD_A)
    history = list(graph.get_state_history(THREAD_A))

    assert len(history) > 1
    assert history[0] == graph.get_state(THREAD_A)
    ids = {snapshot.config["configurable"]["checkpoint_id"] for snapshot in history}
    assert len(ids) == len(history)
    for newer, older in zip(history, history[1:]):
        assert checkpoint_step(newer) == checkpoint_step(older) + 1
        assert newer.parent_config == older.config

    input_checkpoint = history[-1]
    assert input_checkpoint.next == (START,)
    assert input_checkpoint.values == {}
    assert input_checkpoint.metadata is not None
    assert input_checkpoint.metadata["source"] == "input"

    before_parse = next(snapshot for snapshot in history if snapshot.next == ("ParsePaper",))
    assert before_parse.values["raw_paper_content"] == FIRST["raw_paper_content"]
    assert before_parse.values["paper_content"] == ""
    before_report = next(snapshot for snapshot in history if snapshot.next == ("GenerateReport",))
    assert before_report.values["paper_content"] == FIRST_CONTENT
    assert before_report.values["report"] == ""
    assert history[0].values["report"] == FIRST_REPORT

    checkpoint_id = before_parse.config["configurable"]["checkpoint_id"]
    historical_config: RunnableConfig = {
        "configurable": {
            "thread_id": "A",
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }
    retrieved = graph.get_state(historical_config)
    assert retrieved == before_parse


def test_a7_parallel_tasks_share_one_superstep_and_merge_before_report():
    graph = build_paper_graph(InMemorySaver())
    graph.invoke(FIRST, THREAD_A)
    history = list(graph.get_state_history(THREAD_A))

    parallel_nodes = {"AnalyzeContent", "CheckReferences"}
    [before_parallel] = [snapshot for snapshot in history if set(snapshot.next) == parallel_nodes]
    [after_parallel] = [snapshot for snapshot in history if snapshot.next == ("GenerateReport",)]

    # tasks 含各节点的结果，该快照的 values 仍是并行步执行前的完整状态。
    assert len(before_parallel.tasks) == len(parallel_nodes)
    assert {task.name: task.result for task in before_parallel.tasks} == {
        "AnalyzeContent": {"analysis": FIRST_ANALYSIS},
        "CheckReferences": {"references": ["g1", "g2"]},
    }
    assert before_parallel.values["analysis"] == ""
    assert before_parallel.values["references"] == []

    # 唯一直接后继是合并后的完整快照，中间没有只含一个分支结果的检查点。
    assert [s.config for s in history if s.parent_config == before_parallel.config] == [
        after_parallel.config
    ]
    assert checkpoint_step(after_parallel) == checkpoint_step(before_parallel) + 1
    assert after_parallel.values == {
        **before_parallel.values,
        "analysis": FIRST_ANALYSIS,
        "references": ["g1", "g2"],
    }
    for snapshot in history:
        assert bool(snapshot.values.get("analysis")) == bool(snapshot.values.get("references"))
    assert history[0].parent_config == after_parallel.config
    assert history[0].values["report_history"] == [FIRST_REPORT]


def test_a8_a_new_saver_starts_with_no_thread_state():
    first_graph = build_paper_graph(InMemorySaver())
    first_result = first_graph.invoke(FIRST, THREAD_A)
    original_history = list(first_graph.get_state_history(THREAD_A))

    fresh_graph = build_paper_graph(InMemorySaver())
    empty = fresh_graph.get_state(THREAD_A)
    assert empty.values == {}
    assert empty.metadata is None
    assert list(fresh_graph.get_state_history(THREAD_A)) == []

    fresh_result = fresh_graph.invoke(SECOND, THREAD_A)
    assert fresh_result["completed_runs"] == 1
    assert fresh_result["report_history"] == [SECOND_REPORT]
    assert first_graph.get_state(THREAD_A).values == first_result
    assert list(first_graph.get_state_history(THREAD_A)) == original_history
