import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from experiments.langgraph_m5_persistence.postgres_lab import connect_lab, open_postgres_saver

from .demo import SAMPLE, disable_tracing, review_command, summarize, thread_config
from .mail_store import SQLiteMailStore
from .state import EmailInput, SendResult
from .workflow import build_email_graph


OTHER: EmailInput = {"query": "交流珊瑚礁的观测记录", "recipient": "ocean@example.test"}
CRASH_EXIT_CODE = 73


def prepare_lab() -> str:
    with open_postgres_saver() as saver:
        saver.setup()
    with connect_lab() as connection:
        return connection.execute("SHOW server_version").fetchone()["server_version"]


@contextmanager
def lab_threads(*labels: str) -> Iterator[tuple[str, ...]]:
    scope = uuid4().hex
    threads = tuple(f"m6-{scope}-{label}" for label in labels)
    try:
        yield threads
    finally:
        with open_postgres_saver() as saver:
            for thread_id in threads:
                saver.delete_thread(thread_id)


def worker(action: str, thread_id: str, store_path: Path) -> dict:
    store = SQLiteMailStore(store_path)
    config = thread_config(thread_id)

    def crash_after_send(key: str, result: SendResult) -> None:
        print(json.dumps({"pid": os.getpid(), "thread_id": thread_id, "key": key, "send_result": result}), flush=True)
        # SQLite 已提交，SendEmail 仍未返回，直接退出这个实验子进程。
        os._exit(CRASH_EXIT_CODE)

    with open_postgres_saver() as saver:
        after_send = crash_after_send if action == "crash" else None
        graph = build_email_graph(saver, store.send, after_send=after_send)
        details = {"pid": os.getpid()}
        if action == "start":
            graph.invoke(SAMPLE, config, durability="sync")
        else:
            state = graph.get_state(config)
            details["before"] = summarize(state)
            if action in ("approve", "crash"):
                (pending,) = state.interrupts
                graph.invoke(review_command(pending.value, "approve"), config, durability="sync")
                if action == "crash":
                    raise AssertionError("SendEmail 应在返回前使子进程退出")
            elif action == "recover":
                # 审批更新已经持久化。没有新的人工反馈，继续执行未完成的节点。
                graph.invoke(None, config, durability="sync")
            else:
                raise ValueError("未知的实验子进程操作")
        return {**details, "final": summarize(graph.get_state(config))}


def run_worker(action: str, thread_id: str, store_path: Path) -> dict:
    process = subprocess.run(
        [sys.executable, "-m", "experiments.langgraph_m6.demo_part_b", "--worker", action, thread_id,
         str(store_path.resolve())],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, encoding="utf-8", timeout=45,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    expected_exit = CRASH_EXIT_CODE if action == "crash" else 0
    if process.returncode != expected_exit:
        raise RuntimeError(f"{action} 子进程退出码为 {process.returncode}\n{process.stderr}")
    return {**json.loads(process.stdout), "exit_code": process.returncode}


def resume_review_in_processes(store: SQLiteMailStore) -> dict:
    with lab_threads("review") as (thread_id,):
        first = run_worker("start", thread_id, store.path)
        key = first["final"]["values"]["idempotency_key"]
        before = store.evidence(key)
        second = run_worker("approve", thread_id, store.path)
        return {"first": first, "second": second, "mail_before": before, "mail_after": store.evidence(key)}


def verify_thread_isolation(store: SQLiteMailStore) -> dict:
    with lab_threads("a", "b") as (thread_a, thread_b):
        a, b = thread_config(thread_a), thread_config(thread_b)
        with open_postgres_saver() as saver:
            graph = build_email_graph(saver, store.send)
            request_a = graph.invoke(SAMPLE, a, version="v2", durability="sync").interrupts[0].value
            request_b = graph.invoke(OTHER, b, version="v2", durability="sync").interrupts[0].value
            a_before = summarize(graph.get_state(a))
            b_before = summarize(graph.get_state(b))
            graph.invoke(review_command(request_a, "approve"), a, durability="sync")
            a_sent = summarize(graph.get_state(a))
            b_while_a_sent = summarize(graph.get_state(b))
            graph.invoke(review_command(request_b, "reject", "观测记录尚未公开"), b, durability="sync")

        with open_postgres_saver() as saver:
            reopened = build_email_graph(saver, store.send)
            final_a = summarize(reopened.get_state(a))
            final_b = summarize(reopened.get_state(b))
        return {
            "a_before": a_before, "b_before": b_before, "b_while_a_sent": b_while_a_sent,
            "a_sent": a_sent, "final_a": final_a, "final_b": final_b,
            "mail_a": store.evidence(final_a["values"]["idempotency_key"]),
            "mail_b": store.evidence(final_b["values"]["idempotency_key"]),
        }


def observe_crash_boundary(thread_id: str, store: SQLiteMailStore) -> dict:
    with open_postgres_saver() as saver:
        graph = build_email_graph(saver, store.send)
        snapshot = graph.get_state(thread_config(thread_id))
        saved = saver.get_tuple(snapshot.config)
        if saved is None:
            raise RuntimeError("未读取到故障边界的 Checkpoint")
        (task,) = snapshot.tasks
        return {
            "checkpoint": summarize(snapshot),
            "send_task_name": task.name,
            "send_task_result": task.result,
            "send_pending_channels": sorted({
                channel for task_id, channel, _ in (saved.pending_writes or ()) if task_id == task.id
            }),
        }


def recover_send_in_processes(store: SQLiteMailStore) -> dict:
    with lab_threads("crash") as (thread_id,):
        first = run_worker("start", thread_id, store.path)
        key = first["final"]["values"]["idempotency_key"]
        crashed = run_worker("crash", thread_id, store.path)
        boundary = observe_crash_boundary(thread_id, store)
        mail_before = store.evidence(key)
        resumed = run_worker("recover", thread_id, store.path)
        return {
            "first": first, "crashed": crashed, "boundary": boundary, "resumed": resumed,
            "mail_before": mail_before, "mail_after": store.evidence(key),
        }


def main(args: list[str]) -> None:
    disable_tracing()
    if len(args) == 4 and args[0] == "--worker":
        action, thread_id, store_path = args[1:]
        print(json.dumps(worker(action, thread_id, Path(store_path)), ensure_ascii=False))
        return
    if args:
        raise SystemExit("用法：python -m experiments.langgraph_m6.demo_part_b")

    print(f"M6 Part B | PostgreSQL {prepare_lab()}", flush=True)
    with TemporaryDirectory(prefix="m6-postgres-") as directory:
        store = SQLiteMailStore(Path(directory) / "mail.sqlite3")
        store.setup()
        for label, scenario in (
            ("B1 跨进程审批", resume_review_in_processes),
            ("B2 Thread 隔离", verify_thread_isolation),
            ("B3 提交后硬退出", recover_send_in_processes),
        ):
            print(f"\n{label}")
            print(json.dumps(scenario(store), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
