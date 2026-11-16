"""只有 B -> D，没有 C -> D；用 stream 的 B 更新确认 B 已经返回。"""

import operator
from threading import Event, Lock
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    completed: Annotated[list[str], operator.add]


def run_superstep_demo():
    c_started, release_c, c_finished, d_started = (Event() for _ in range(4))
    lock = Lock()
    events = []

    def record(name, config=None):
        with lock:
            events.append({"event": name, "step": (
                config["metadata"]["langgraph_step"] if config else None)})

    def a(state: State, config: RunnableConfig):
        record("A", config)
        return {}

    def b(state: State, config: RunnableConfig):
        # 超时仅作死锁保护，不用时间差推断顺序。
        if not c_started.wait(10):
            raise TimeoutError("C 未启动")
        record("B:return", config)
        return {"completed": ["B"]}

    def c(state: State, config: RunnableConfig):
        record("C:blocked", config)
        c_started.set()
        if not release_c.wait(10):
            raise TimeoutError("观察者未释放 C")
        record("C:return", config)
        c_finished.set()
        return {"completed": ["C"]}

    def d(state: State, config: RunnableConfig):
        d_started.set()
        assert c_finished.is_set(), "D 在 C 完成前执行了"
        assert sorted(state["completed"]) == ["B", "C"]
        record("D:start (sees B,C)", config)
        return {"completed": ["D"]}

    builder = StateGraph(State)
    for name, node in (("A", a), ("B", b), ("C", c), ("D", d)):
        builder.add_node(name, node)
    for source, target in ((START, "A"), ("A", "B"), ("A", "C"),
                           ("B", "D"), ("C", END), ("D", END)):
        builder.add_edge(source, target)
    graph = builder.compile()
    # stream 在主线程消费，B/C 是 Runtime 管理的线程；没有手工调用节点。
    try:
        for update in graph.stream({}, config={"max_concurrency": 4}, stream_mode="updates"):
            if "B" in update:
                assert not d_started.is_set()
                assert not c_finished.is_set()
                record("observer:B update received; C blocked; D not started")
                release_c.set()
    finally:
        release_c.set()
    assert d_started.is_set()
    return events


if __name__ == "__main__":
    for event in run_superstep_demo():
        print(f"step={event['step']} {event['event']}")
