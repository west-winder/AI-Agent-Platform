import os
from collections import Counter

import pytest

from demo_part_b import (
    CRASH_EXIT_CODE,
    lab_order,
    order_case,
    persistence_instances,
    persistence_processes,
    prepare_lab,
    print_case,
    resume_in_memory,
    resume_in_processes,
)
from order_graph import OrderStore, order_key
from postgres_lab import connect_lab


FIRST_REPORT = "paper-1｜内容摘要｜石墨烯 导电 [ref:g1]｜引用 g1"
SECOND_REPORT = "paper-2｜内容摘要｜月壤 含水｜引用 无"
OTHER_REPORT = "paper-other｜内容摘要｜珊瑚礁 温度 [ref:b1]｜引用 b1"


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "false")


@pytest.fixture(scope="module")
def postgres_ready():
    if not os.environ.get("M5_POSTGRES_URI"):
        pytest.skip("未配置本机 M5_POSTGRES_URI")
    prepare_lab()


def assert_continued_paper(first, second):
    assert first["next"] == second["next"] == []
    assert first["thread_id"] == second["thread_id"]
    assert first["values"]["completed_runs"] == 1
    assert first["values"]["report_history"] == [FIRST_REPORT]
    assert second["values"]["run_id"] == "paper-2"
    assert second["values"]["completed_runs"] == 2
    assert second["values"]["report_history"] == [FIRST_REPORT, SECOND_REPORT]
    assert second["values"]["references"] == []
    assert second["values"]["analysis"] == "内容摘要｜月壤 含水"
    assert second["checkpoint_id"] != first["checkpoint_id"]
    assert second["history_ids"][0] == second["checkpoint_id"]
    assert first["checkpoint_id"] in second["history_ids"]
    assert second["history_ids"][-len(first["history_ids"]):] == first["history_ids"]


@pytest.mark.postgres
def test_b1_fresh_saver_continues_and_keeps_threads_separate(postgres_ready):
    result = persistence_instances()
    assert result["reopened"] == result["first"]
    assert_continued_paper(result["first"], result["second"])

    other = result["other_after"]
    assert result["other_before"] == other
    assert other["thread_id"] != result["second"]["thread_id"]
    assert other["values"]["run_id"] == "paper-other"
    assert other["values"]["completed_runs"] == 1
    assert other["values"]["report_history"] == [OTHER_REPORT]
    assert other["values"]["references"] == ["b1"]
    assert not set(other["history_ids"]) & set(result["second"]["history_ids"])


@pytest.mark.postgres
@pytest.mark.cross_process
def test_b1_checkpoint_survives_two_independent_processes(postgres_ready, tmp_path):
    result = persistence_processes(tmp_path / "b1.jsonl")
    first, second = result["first"], result["second"]
    assert first["pid"] != second["pid"]
    assert first["exit_code"] == second["exit_code"] == 0
    assert first["final"] == second["before"]
    assert_continued_paper(first["final"], second["final"])
    print_case("B1-2｜跨进程持久化", result)


def assert_parallel_recovery(before, final, audit):
    assert before["exception"] == "模拟 CheckReferences 首次执行失败"
    assert set(before["next"]) == {"AnalyzeContent", "CheckReferences"}
    assert before["next_on_resume"] == ["CheckReferences"]
    assert before["values"]["analysis"] == ""
    assert before["values"]["references"] == []
    assert before["values"]["report"] == ""
    assert before["analysis_task_result"] == {"analysis": "内容摘要｜石墨烯 导电 [ref:g1]"}
    assert before["pending_analysis"] == ["内容摘要｜石墨烯 导电 [ref:g1]"]
    assert before["reference_task_failed"]
    assert before["checkpoint_id"] in final["history_ids"]
    assert before["thread_id"] == final["thread_id"]
    assert before["step"] < final["step"]
    assert final["next"] == []
    assert final["values"]["run_id"] == before["values"]["run_id"] == "paper-1"
    assert final["values"]["report"] == FIRST_REPORT
    assert final["values"]["references"] == ["g1"]
    assert final["values"]["completed_runs"] == 1
    assert final["values"]["report_history"] == [FIRST_REPORT]
    assert Counter(record["node"] for record in audit) == {
        "InitializeRun": 1, "ParsePaper": 1, "AnalyzeContent": 1,
        "CheckReferences": 2, "GenerateReport": 1,
    }


def test_b2_in_memory_resumes_failed_branch_only(tmp_path):
    result = resume_in_memory(tmp_path / "b2-memory.jsonl")
    assert_parallel_recovery(result["before"], result["final"], result["audit"])


@pytest.mark.postgres
@pytest.mark.cross_process
def test_b2_postgres_resumes_failed_branch_in_a_new_process(postgres_ready, tmp_path):
    result = resume_in_processes(tmp_path / "b2-postgres.jsonl")
    failed, resumed = result["failed"], result["resumed"]
    assert failed["pid"] != resumed["pid"]
    assert failed["exit_code"] == resumed["exit_code"] == 0
    assert failed["failed"]["checkpoint_id"] == resumed["before"]["checkpoint_id"]
    assert resumed["before"]["values"]["run_id"] == "paper-1"
    assert_parallel_recovery(failed["failed"], resumed["final"], result["audit"])
    by_pid = {
        pid: Counter(entry["node"] for entry in result["audit"] if entry["pid"] == pid)
        for pid in (failed["pid"], resumed["pid"])
    }
    assert by_pid[failed["pid"]] == {
        "InitializeRun": 1, "ParsePaper": 1, "AnalyzeContent": 1, "CheckReferences": 1,
    }
    assert by_pid[resumed["pid"]] == {"CheckReferences": 1, "GenerateReport": 1}
    print_case("B2-2｜跨进程并行恢复", result)


def assert_post_commit_window(result):
    before, after = result["before"], result["after"]
    pending = before["checkpoint"]
    saved = after["checkpoint"]
    key = pending["values"]["idempotency_key"]
    assert pending["next"] == ["SaveOrder"]
    assert pending["values"]["save_status"] == "pending"
    assert pending["values"]["order_id"] is None
    assert before["save_task_result"] is None
    assert not {"order_id", "save_status"}.intersection(before["save_pending_channels"])
    assert len(before["orders"]) == 1
    assert all((row["item"], row["quantity"]) == ("实验笔记本", 2) for row in after["orders"])
    assert saved == result["resumed"]["final"]
    assert saved["checkpoint_id"] != pending["checkpoint_id"]
    assert saved["next"] == []
    assert saved["values"]["save_status"] == "saved"
    assert saved["values"]["idempotency_key"] == key
    attempts = [row for row in result["audit"] if row["event"] == "attempt"]
    committed = [row for row in result["audit"] if row["event"] == "committed"]
    assert len(attempts) == len(committed) == 2
    assert {row["node"] for row in result["audit"]} == {"SaveOrder"}
    assert {row["idempotency_key"] for row in result["audit"]} == {key}
    assert committed[0]["order_id"] == before["orders"][0]["id"]
    assert committed[1]["order_id"] == saved["values"]["order_id"]


@pytest.mark.postgres
def test_b3_unprotected_resume_creates_a_duplicate_order(postgres_ready, tmp_path):
    result = order_case(tmp_path / "b3-plain.jsonl", mode="unprotected")
    assert "事务提交后" in result["first"]["exception"]
    assert_post_commit_window(result)
    assert len(result["after"]["orders"]) == 2
    assert result["before"]["orders"][0] == result["after"]["orders"][0]
    assert result["after"]["checkpoint"]["values"]["order_id"] == result["after"]["orders"][1]["id"]


@pytest.mark.postgres
def test_b3_idempotent_resume_returns_the_existing_order(postgres_ready, tmp_path):
    result = order_case(tmp_path / "b3-keyed.jsonl", mode="idempotent")
    assert "事务提交后" in result["first"]["exception"]
    assert_post_commit_window(result)
    assert result["after"]["orders"] == result["before"]["orders"]
    assert result["after"]["checkpoint"]["values"]["order_id"] == result["before"]["orders"][0]["id"]


@pytest.mark.postgres
def test_b3_same_key_with_different_parameters_is_rejected(postgres_ready):
    with lab_order(idempotent=True) as thread_id, connect_lab() as connection:
        store = OrderStore(connection, idempotent=True)
        key = order_key(thread_id)
        order_id = store.save(key, "笔记本", 2)

        for item, quantity in (("铅笔", 2), ("笔记本", 3)):
            with pytest.raises(ValueError, match="不同订单参数"):
                store.save(key, item, quantity)
        assert store.find(key) == [{"id": order_id, "item": "笔记本", "quantity": 2}]


@pytest.mark.postgres
@pytest.mark.cross_process
def test_b3_worker_crash_after_commit_keeps_exactly_one_order(postgres_ready, tmp_path):
    result = order_case(tmp_path / "b3-crash.jsonl", mode="crash")
    first, resumed = result["first"], result["resumed"]
    assert first["pid"] != resumed["pid"]
    assert first["exit_code"] == CRASH_EXIT_CODE
    assert resumed["exit_code"] == 0
    assert_post_commit_window(result)
    assert result["before"]["save_pending_channels"] == []
    assert first["committed_order_id"] == result["before"]["orders"][0]["id"]
    assert result["after"]["orders"] == result["before"]["orders"]
    assert resumed["final"]["values"]["order_id"] == first["committed_order_id"]
    assert [row["pid"] for row in result["audit"] if row["event"] == "attempt"] == [
        first["pid"], resumed["pid"],
    ]
    print_case("B3-3｜订单提交后子进程硬退出", result)


@pytest.mark.parametrize("conninfo", [
    "dbname=postgres host=127.0.0.1",
    "dbname=agent_platform_m5_lab host=example.com",
    "dbname=agent_platform_m5_lab host=localhost hostaddr=192.0.2.10",
])
def test_lab_rejects_unsafe_connection_targets(monkeypatch, conninfo):
    monkeypatch.setenv("M5_POSTGRES_URI", conninfo)
    with pytest.raises(ValueError, match="仅连接本机 PostgreSQL 实验库"), connect_lab():
        pytest.fail("校验失败时不得建立数据库连接")
