import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from order_graph import OrderStore, build_order_graph, order_key
from paper_graph import PaperInput, build_paper_graph
from postgres_lab import connect_lab, open_postgres_saver
from resume_lab import (
    NodeAudit,
    ReferenceFailureGate,
    observe_failed_run,
    observed_paper_graph,
    snapshot_summary,
    thread_config,
)


PAPER_FIRST: PaperInput = {"run_id": "paper-1", "raw_paper_content": "石墨烯 导电 [ref:g1]"}
PAPER_SECOND: PaperInput = {"run_id": "paper-2", "raw_paper_content": "月壤 含水"}
PAPER_OTHER: PaperInput = {"run_id": "paper-other", "raw_paper_content": "珊瑚礁 温度 [ref:b1]"}
CRASH_EXIT_CODE = 73
WORKER_TIMEOUT = 45


class PostCommitFailure(RuntimeError):
    """订单事务已提交，SaveOrder 尚未返回。"""


def prepare_lab() -> str:
    with open_postgres_saver() as saver:
        saver.setup()
    with connect_lab() as connection:
        for idempotent in (False, True):
            OrderStore(connection, idempotent=idempotent).setup()
        return connection.execute("SHOW server_version").fetchone()["server_version"]


@contextmanager
def lab_threads(*labels: str) -> Iterator[tuple[str, ...]]:
    scope = uuid4().hex
    threads = tuple(f"m5-{scope}-{label}" for label in labels)
    try:
        yield threads
    finally:
        with open_postgres_saver() as saver:
            for thread_id in threads:
                saver.delete_thread(thread_id)


@contextmanager
def lab_order(*, idempotent: bool) -> Iterator[str]:
    with lab_threads("order") as (thread_id,):
        try:
            yield thread_id
        finally:
            with connect_lab() as connection:
                OrderStore(connection, idempotent=idempotent).delete(order_key(thread_id))


def paper_snapshot(graph, config) -> dict:
    return {
        **snapshot_summary(graph.get_state(config)),
        "history_ids": [
            entry.config["configurable"]["checkpoint_id"]
            for entry in graph.get_state_history(config)
        ],
    }


def persistence_instances() -> dict:
    with lab_threads("a", "b") as (thread_a, thread_b):
        a, b = thread_config(thread_a), thread_config(thread_b)
        with open_postgres_saver() as saver:
            graph = build_paper_graph(saver)
            graph.invoke(PAPER_FIRST, a, durability="sync")
            first = paper_snapshot(graph, a)
            graph.invoke(PAPER_OTHER, b, durability="sync")
            other_before = paper_snapshot(graph, b)

        # 上一个编译图及 Saver 的连接已关闭，再打开独立连接读取旧历史。
        with open_postgres_saver() as saver:
            graph = build_paper_graph(saver)
            reopened = paper_snapshot(graph, a)
            graph.invoke(PAPER_SECOND, a, durability="sync")
            second = paper_snapshot(graph, a)
            other_after = paper_snapshot(graph, b)
    return {
        "first": first, "reopened": reopened, "second": second,
        "other_before": other_before, "other_after": other_after,
    }


def run_worker(action: str, thread_id: str, audit_path: Path) -> dict:
    expected_exit = CRASH_EXIT_CODE if action == "order-crash" else 0
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", action, thread_id, str(audit_path)],
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=WORKER_TIMEOUT,
    )
    if result.returncode != expected_exit:
        raise RuntimeError(f"{action} 子进程退出码 {result.returncode}\n{result.stderr}")
    return {**json.loads(result.stdout), "exit_code": result.returncode}


def persistence_processes(audit_path: Path) -> dict:
    with lab_threads("persistence") as (thread_id,):
        first = run_worker("persist-first", thread_id, audit_path)
        second = run_worker("persist-second", thread_id, audit_path)
    return {"first": first, "second": second}


def resume_in_memory(audit_path: Path) -> dict:
    saver = InMemorySaver()
    gate = ReferenceFailureGate()
    audit = NodeAudit(audit_path)
    graph = observed_paper_graph(saver, audit, gate)
    config = thread_config("resume-memory")
    before = observe_failed_run(graph, saver, config, PAPER_FIRST, gate)
    gate.enabled = False
    graph.invoke(None, config=config, durability="sync")
    return {"before": before, "final": paper_snapshot(graph, config), "audit": audit.read()}


def resume_in_processes(audit_path: Path) -> dict:
    with lab_threads("resume") as (thread_id,):
        failed = run_worker("paper-fail", thread_id, audit_path)
        resumed = run_worker("paper-resume", thread_id, audit_path)
    return {"failed": failed, "resumed": resumed, "audit": NodeAudit(audit_path).read()}


def order_worker(
    phase: Literal["fail", "crash", "resume"],
    thread_id: str,
    audit_path: Path,
    *,
    idempotent: bool,
) -> dict:
    audit = NodeAudit(audit_path)
    config = thread_config(thread_id)

    def attempt(key: str) -> None:
        audit.record("SaveOrder", idempotency_key=key)

    def after_commit(key: str, order_id: int) -> None:
        audit.record("SaveOrder", event="committed", idempotency_key=key, order_id=order_id)
        if phase == "crash":
            payload = {"pid": os.getpid(), "idempotency_key": key, "committed_order_id": order_id}
            print(json.dumps(payload), flush=True)
            # 只终止本实验子进程。跳过栈展开和 Saver 连接的正常清理。
            os._exit(CRASH_EXIT_CODE)
        if phase == "fail":
            raise PostCommitFailure("模拟订单事务提交后、节点返回前的故障")

    with open_postgres_saver() as saver, connect_lab() as connection:
        store = OrderStore(connection, idempotent=idempotent)
        graph = build_order_graph(store, saver, on_attempt=attempt, after_commit=after_commit)
        if phase == "resume":
            graph.invoke(None, config=config, durability="sync")
            return {"pid": os.getpid(), "final": snapshot_summary(graph.get_state(config))}

        try:
            graph.invoke(
                {"run_id": thread_id, "item": "实验笔记本", "quantity": 2},
                config=config, durability="sync",
            )
        except PostCommitFailure as failure:
            return {"pid": os.getpid(), "exception": str(failure)}
        raise AssertionError("SaveOrder 故障应发生在返回 State 更新之前")


def observe_order(thread_id: str, *, idempotent: bool) -> dict:
    config = thread_config(thread_id)
    with open_postgres_saver() as saver, connect_lab() as connection:
        store = OrderStore(connection, idempotent=idempotent)
        graph = build_order_graph(store, saver)
        current = graph.get_state(config)
        boundary = graph.get_state(current.config)
        saved = saver.get_tuple(boundary.config)
        assert saved is not None
        save_task = next((task for task in boundary.tasks if task.name == "SaveOrder"), None)
        return {
            "checkpoint": snapshot_summary(boundary),
            "orders": store.find(order_key(thread_id)),
            "save_task_result": save_task.result if save_task else None,
            "save_pending_channels": sorted({
                channel for task_id, channel, _ in saved.pending_writes or ()
                if save_task and task_id == save_task.id
            }),
        }


def order_case(audit_path: Path, *, mode: Literal["unprotected", "idempotent", "crash"]) -> dict:
    idempotent = mode != "unprotected"
    crash = mode == "crash"
    with lab_order(idempotent=idempotent) as thread_id:
        if crash:
            first = run_worker("order-crash", thread_id, audit_path)
        else:
            first = order_worker("fail", thread_id, audit_path, idempotent=idempotent)
        before = observe_order(thread_id, idempotent=idempotent)

        if crash:
            resumed = run_worker("order-resume", thread_id, audit_path)
        else:
            resumed = order_worker("resume", thread_id, audit_path, idempotent=idempotent)
        after = observe_order(thread_id, idempotent=idempotent)
    return {
        "first": first, "before": before, "resumed": resumed, "after": after,
        "audit": NodeAudit(audit_path).read(),
    }


def worker(action: str, thread_id: str, audit_path: Path) -> dict:
    if action == "order-crash":
        return order_worker("crash", thread_id, audit_path, idempotent=True)
    if action == "order-resume":
        return order_worker("resume", thread_id, audit_path, idempotent=True)

    config = thread_config(thread_id)
    audit = NodeAudit(audit_path)
    details = {"pid": os.getpid()}
    with open_postgres_saver() as saver:
        if action == "persist-first":
            graph = build_paper_graph(saver)
            graph.invoke(PAPER_FIRST, config, durability="sync")
            details["final"] = paper_snapshot(graph, config)
        elif action == "persist-second":
            graph = build_paper_graph(saver)
            details["before"] = paper_snapshot(graph, config)
            graph.invoke(PAPER_SECOND, config, durability="sync")
            details["final"] = paper_snapshot(graph, config)
        elif action == "paper-fail":
            gate = ReferenceFailureGate()
            graph = observed_paper_graph(saver, audit, gate)
            details["failed"] = observe_failed_run(graph, saver, config, PAPER_FIRST, gate)
        elif action == "paper-resume":
            graph = observed_paper_graph(saver, audit)
            details["before"] = paper_snapshot(graph, config)
            graph.invoke(None, config=config, durability="sync")
            details["final"] = paper_snapshot(graph, config)
        else:
            raise ValueError("未知的实验子进程操作")
    return details


def print_case(label: str, details: dict) -> None:
    print(f"\n{label}")
    for field, value in details.items():
        print(f"  {field} = {json.dumps(value, ensure_ascii=False)}")


def main(args: list[str]) -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    if len(args) == 4 and args[0] == "--worker":
        action, thread_id, audit_path = args[1:]
        print(json.dumps(worker(action, thread_id, Path(audit_path)), ensure_ascii=False))
        return
    if args:
        raise SystemExit("用法：python demo_part_b.py")

    print(f"M5 Part B｜PostgreSQL {prepare_lab()}")
    with TemporaryDirectory(prefix="m5-part-b-") as folder:
        audit_dir = Path(folder)
        print_case("B1-1 / B1-3｜新 Saver 和 Thread 隔离", persistence_instances())
        print_case("B1-2｜跨进程持久化", persistence_processes(audit_dir / "b1.jsonl"))
        print_case("B2-1｜InMemorySaver 恢复", resume_in_memory(audit_dir / "b2-memory.jsonl"))
        print_case("B2-2｜PostgresSaver 跨进程恢复", resume_in_processes(audit_dir / "b2-pg.jsonl"))
        print_case("B3-1｜无幂等保护", order_case(audit_dir / "b3-plain.jsonl", mode="unprotected"))
        print_case("B3-2｜有幂等保护", order_case(audit_dir / "b3-keyed.jsonl", mode="idempotent"))
        print_case("B3-3｜提交后硬退出", order_case(audit_dir / "b3-crash.jsonl", mode="crash"))


if __name__ == "__main__":
    main(sys.argv[1:])
