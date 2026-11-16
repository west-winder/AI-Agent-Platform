from collections import Counter
from copy import deepcopy
from threading import Barrier, Event

import pytest

import paper_review as review
from superstep_demo import run_superstep_demo


TRACE = ["ExtractReferences", "ValidateReferences", "GenerateReferenceResult"]


@pytest.mark.parametrize("n", [1, 3, 5])
def test_send_tasks_isolation_reducers_and_report(n):
    paper = list(reversed(review.mock_paper(n)))
    original = deepcopy(paper)
    # 同时读 tasks 和 updates：不仅检查分发计划，还检查 Runtime 的实际任务。
    stream = list(review.build_paper_graph().stream(
        {"paper": paper}, config={"max_concurrency": 16},
        stream_mode=["tasks", "updates"]))
    starts = [data for mode, data in stream if mode == "tasks" and "input" in data]
    workers = [t for t in starts if t["name"] in ("AnalyzeContent", "CheckReferences")]
    assert len(workers) == 2 * n
    expected = {c["chapter_id"]: c for c in paper}
    assert Counter((t["name"], t["input"]["chapter_id"]) for t in workers) == Counter(
        (node, i) for i in expected for node in ("AnalyzeContent", "CheckReferences"))
    for task in workers:
        assert task["input"] == expected[task["input"]["chapter_id"]]
    reports = [t for t in starts if t["name"] == "GenerateReport"]
    assert len(reports) == 1
    state = reports[0]["input"]
    assert state["events"][0]["task_count"] == 2 * n
    for key in ("content_results", "reference_results"):
        assert sorted(r["chapter_id"] for r in state[key]) == sorted(expected)
    for r in state["content_results"]:
        assert r["result"] == f"characters={len(expected[r['chapter_id']]['content'])}"
    events = [e for e in state["events"] if "chapter_id" in e]
    assert len(events) == 2 * n
    assert {e["step"] for e in events} == {3}
    for e in events:
        assert e["input_keys"] == ["chapter_id", "content"]
        assert e["content"] == expected[e["chapter_id"]]["content"]
        if e["node"] == "CheckReferences":
            assert e["subgraph_input"] == expected[e["chapter_id"]]
            assert e["subgraph_trace"] == TRACE
    report_updates = [data["GenerateReport"] for mode, data in stream
                      if mode == "updates" and "GenerateReport" in data]
    assert len(report_updates) == 1
    update = report_updates[0]
    assert [r["chapter_id"] for r in update["report"]] == sorted(expected)
    assert update["events"] == [{"node": "GenerateReport", "content_count": n,
                                  "reference_count": n, "step": 4}]
    assert paper == original


def test_subgraph_nodes_and_wrapper_conversion():
    chapter = {"chapter_id": "standalone", "content": "abc [ref:missing]"}
    subgraph = review.build_reference_graph()
    updates = list(subgraph.stream(chapter, stream_mode="updates"))
    assert [next(iter(u)) for u in updates] == TRACE
    config = {"metadata": {"langgraph_step": 3}}
    update = review.check_references_wrapper(subgraph, chapter, config)
    assert set(update) == {"reference_results", "events"}
    assert update["reference_results"] == [{"chapter_id": "standalone", "status": "failed",
                                            "result": "extracted=1",
                                            "failure_reason": "模拟引用不存在：missing"}]
    assert update["events"][0]["subgraph_trace"] == TRACE


def test_business_failure_preserves_content_result():
    state = review.build_paper_graph().invoke({"paper": review.mock_paper(3)})
    failed = next(r for r in state["report"] if r["chapter_id"] == "ch02")
    assert failed["content_analysis"]["status"] == "ok"
    assert failed["reference_check"]["status"] == "failed"
    assert failed["reference_check"]["failure_reason"] == "模拟引用不存在：missing"
    assert len(state["content_results"]) == len(state["reference_results"]) == 3


def test_independent_invokes_on_same_compiled_graph():
    graph = review.build_paper_graph()
    first = graph.invoke({"paper": review.mock_paper(5)})
    snapshot = deepcopy(first)
    second = graph.invoke({"paper": [{"chapter_id": "other", "content": ""}]})
    third = graph.invoke({"paper": []})
    assert first == snapshot
    assert [r["chapter_id"] for r in second["report"]] == ["other"]
    assert len(second["content_results"]) == len(second["reference_results"]) == 1
    assert second["reference_results"][0]["failure_reason"] == "未找到引用"
    assert third["report"] == []
    for state in (first, second, third):
        assert sum(e["node"] == "GenerateReport" for e in state["events"]) == 1


def test_empty_paper_generates_one_empty_report():
    state = review.build_paper_graph().invoke({"paper": []})
    assert state["content_results"] == state["reference_results"] == state["report"] == []
    assert state["events"][0]["task_count"] == 0
    assert len(state["events"]) == 2


@pytest.mark.parametrize("ids", [["same", "same"], [""]])
def test_invalid_chapter_ids(ids):
    with pytest.raises(ValueError, match="chapter_id"):
        review.build_paper_graph().invoke({"paper": [
            {"chapter_id": i, "content": "x"} for i in ids]})


def test_send_workers_really_overlap(monkeypatch):
    # 若 Runtime 顺序执行 3 个 AnalyzeContent，这个屏障必定超时。
    # 不替换调度器，只在模拟业务函数中加入确定性的同步点。
    barrier = Barrier(3, timeout=10)
    original = review.analyze_content

    def gated(chapter: review.Chapter, config):
        barrier.wait()
        return original(chapter, config)

    monkeypatch.setattr(review, "analyze_content", gated)
    state = review.build_paper_graph().invoke(
        {"paper": review.mock_paper(3)}, config={"max_concurrency": 16})
    assert len(state["report"]) == 3


def test_superstep_order():
    events = run_superstep_demo()
    assert [e["event"] for e in events] == [
        "A", "C:blocked", "B:return",
        "observer:B update received; C blocked; D not started",
        "C:return", "D:start (sees B,C)"]
    assert [e["step"] for e in events] == [1, 2, 2, None, 2, 3]


def test_report_waits_for_blocked_reference_wrapper(monkeypatch):
    release = Event()
    original = review.check_references_wrapper

    def gated(subgraph, chapter, config):
        if chapter["chapter_id"] == "ch02" and not release.wait(10):
            raise TimeoutError("观察者未释放引用检查")
        return original(subgraph, chapter, config)

    monkeypatch.setattr(review, "check_references_wrapper", gated)
    graph = review.build_paper_graph()
    content_done = set()
    report_updates = []
    try:
        for update in graph.stream({"paper": review.mock_paper(3)},
                                   config={"max_concurrency": 16}, stream_mode="updates"):
            if "AnalyzeContent" in update:
                content_done.add(update["AnalyzeContent"]["content_results"][0]["chapter_id"])
                if len(content_done) == 3:
                    assert not report_updates
                    release.set()
            if "GenerateReport" in update:
                assert release.is_set()
                report_updates.append(update["GenerateReport"])
    finally:
        release.set()
    assert len(report_updates) == 1
    report = report_updates[0]["report"]
    assert len(report) == 3
    assert report[1]["reference_check"]["failure_reason"] == "模拟引用不存在：missing"
