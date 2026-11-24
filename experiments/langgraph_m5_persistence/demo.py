import json
import os

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from paper_graph import PaperInput, PaperState, build_paper_graph


def print_run(label: str, state: PaperState) -> None:
    print(f"{label}｜run_id={state['run_id']}｜completed_runs={state['completed_runs']}")
    print(f"  report_history = {json.dumps(state['report_history'], ensure_ascii=False)}")


def main() -> None:
    # 即使本机已配置 LangSmith，本地 Demo 也不发出追踪请求。
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"

    paper_a1: PaperInput = {"run_id": "A-1", "raw_paper_content": "  石墨烯   导电 [ref:a1]  "}
    paper_b1: PaperInput = {"run_id": "B-1", "raw_paper_content": "珊瑚礁 温度 [ref:b1]"}
    paper_a2: PaperInput = {"run_id": "A-2", "raw_paper_content": "月壤 含水"}

    print("无 Checkpointer")
    stateless_graph = build_paper_graph()
    print_run("第一次", stateless_graph.invoke({**paper_a1, "run_id": "local-1"}))
    print_run("第二次", stateless_graph.invoke({**paper_a2, "run_id": "local-2"}))

    print("\n同一个 InMemorySaver 与 Compiled Graph")
    graph = build_paper_graph(InMemorySaver())
    thread_a: RunnableConfig = {"configurable": {"thread_id": "A"}}
    thread_b: RunnableConfig = {"configurable": {"thread_id": "B"}}
    print_run("Thread A 第一次", graph.invoke(paper_a1, thread_a))
    print_run("Thread B 第一次", graph.invoke(paper_b1, thread_b))
    print_run("Thread A 第二次", graph.invoke(paper_a2, thread_a))

    for label, config in (("A", thread_a), ("B", thread_b)):
        snapshot = graph.get_state(config)
        print(f"\nThread {label} 最终 State（next={snapshot.next}）")
        print(json.dumps(snapshot.values, ensure_ascii=False, indent=2))

    print("\nThread A 的 Checkpoint 历史（最新在前）")
    for snapshot in graph.get_state_history(thread_a):
        metadata = snapshot.metadata or {}
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        print(f"step={metadata.get('step')}｜checkpoint_id={checkpoint_id}")
        print(f"  next   = {snapshot.next}")
        print(f"  values = {json.dumps(snapshot.values, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
