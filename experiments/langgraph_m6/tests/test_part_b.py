import json
import os

import pytest

from experiments.langgraph_m6.demo_part_b import (
    CRASH_EXIT_CODE,
    prepare_lab,
    recover_send_in_processes,
    resume_review_in_processes,
    verify_thread_isolation,
)
from experiments.langgraph_m6.mail_store import SQLiteMailStore


pytestmark = pytest.mark.postgres
EMPTY_MAIL = {"requests": [], "deliveries": []}


@pytest.fixture(scope="module")
def postgres_ready():
    if not os.environ.get("M5_POSTGRES_URI"):
        pytest.skip("未配置本机 M5_POSTGRES_URI")
    prepare_lab()


@pytest.fixture
def store(postgres_ready, tmp_path):
    mail = SQLiteMailStore(tmp_path / "mail.sqlite3")
    mail.setup()
    return mail


@pytest.mark.cross_process
def test_b1_paused_review_survives_process_exit_and_resumes(store):
    result = resume_review_in_processes(store)
    first, second = result["first"], result["second"]
    assert first["pid"] != second["pid"]
    assert first["exit_code"] == second["exit_code"] == 0
    pending, final = first["final"], second["final"]
    assert pending == second["before"]
    assert pending["next"] == ["HumanApproval"]
    assert len(pending["interrupts"]) == 1
    assert pending["interrupts"][0]["value"]["email_draft"] == pending["values"]["email_draft"]
    assert pending["values"]["status"] == "awaiting_approval"
    assert result["mail_before"] == EMPTY_MAIL
    assert pending["thread_id"] == final["thread_id"]
    assert pending["checkpoint_id"] != final["checkpoint_id"]
    assert final["next"] == final["interrupts"] == []
    assert final["values"]["status"] == "sent"
    assert final["values"]["approved_version"] == pending["values"]["draft_version"]
    assert final["values"]["idempotency_key"] == pending["values"]["idempotency_key"]
    (request,) = result["mail_after"]["requests"]
    (delivery,) = result["mail_after"]["deliveries"]
    assert (request["outcome"], request["pid"]) == ("sent", second["pid"])
    assert final["values"]["send_result"] == {"message_id": delivery["message_id"]}
    print(json.dumps(result, ensure_ascii=False))


def test_b2_threads_keep_independent_reviews_and_persist_separate_decisions(store):
    result = verify_thread_isolation(store)
    a, b = result["a_before"], result["b_before"]
    assert a["thread_id"] != b["thread_id"]
    assert a["checkpoint_id"] != b["checkpoint_id"]
    assert a["next"] == b["next"] == ["HumanApproval"]
    assert a["values"]["status"] == b["values"]["status"] == "awaiting_approval"
    assert a["values"]["decision"] is b["values"]["decision"] is None
    assert a["values"]["email_draft"] != b["values"]["email_draft"]
    assert a["values"]["recipient"] != b["values"]["recipient"]
    assert a["values"]["idempotency_key"] != b["values"]["idempotency_key"]
    assert a["interrupts"][0]["value"]["request_id"] != b["interrupts"][0]["value"]["request_id"]
    assert result["b_while_a_sent"] == b

    final_a, final_b = result["final_a"], result["final_b"]
    assert final_a == result["a_sent"]
    assert final_a["thread_id"] == a["thread_id"]
    assert final_b["thread_id"] == b["thread_id"]
    assert final_a["next"] == final_b["next"] == []
    assert final_a["values"]["status"] == "sent"
    assert final_a["values"]["decision"] == "approve"
    (delivery,) = result["mail_a"]["deliveries"]
    assert delivery["recipient"] == a["values"]["recipient"]
    assert final_b["values"]["status"] == "rejected"
    assert final_b["values"]["decision"] == "reject"
    assert final_b["values"]["rejection_reason"] == "观测记录尚未公开"
    assert final_b["values"]["send_result"] is None
    assert result["mail_b"] == EMPTY_MAIL


@pytest.mark.cross_process
def test_b3_hard_exit_after_send_commit_retries_same_key_without_reapproval(store):
    result = recover_send_in_processes(store)
    first, crashed, resumed = result["first"], result["crashed"], result["resumed"]
    assert len({first["pid"], crashed["pid"], resumed["pid"]}) == 3
    assert first["exit_code"] == resumed["exit_code"] == 0
    assert crashed["exit_code"] == CRASH_EXIT_CODE

    boundary = result["boundary"]
    pending, approved, final = first["final"], boundary["checkpoint"], resumed["final"]
    assert pending["thread_id"] == approved["thread_id"] == final["thread_id"] == crashed["thread_id"]
    assert pending["step"] < approved["step"] < final["step"]
    assert len({pending["checkpoint_id"], approved["checkpoint_id"], final["checkpoint_id"]}) == 3
    assert approved == resumed["before"]
    assert approved["next"] == ["SendEmail"]
    assert approved["interrupts"] == []
    assert approved["values"]["status"] == "approved"
    assert approved["values"]["approved_version"] == approved["values"]["draft_version"] == 1
    assert approved["values"]["decision"] == "approve"
    assert approved["values"]["send_result"] is None
    assert boundary["send_task_name"] == "SendEmail"
    assert boundary["send_task_result"] is None
    assert boundary["send_pending_channels"] == []

    key = crashed["key"]
    assert all(snapshot["values"]["idempotency_key"] == key for snapshot in (pending, approved, final))
    before, after = result["mail_before"], result["mail_after"]
    (delivery,) = before["deliveries"]
    assert after["deliveries"] == before["deliveries"]
    assert delivery["idempotency_key"] == key
    assert delivery["email_draft"] == approved["values"]["email_draft"]
    assert delivery["recipient"] == approved["values"]["recipient"]
    assert before["requests"] == after["requests"][:1]
    assert [(row["pid"], row["idempotency_key"], row["outcome"]) for row in after["requests"]] == [
        (crashed["pid"], key, "sent"), (resumed["pid"], key, "replayed"),
    ]
    assert final["next"] == final["interrupts"] == []
    assert final["values"]["status"] == "sent"
    assert final["values"]["approved_version"] == 1
    assert final["values"]["send_result"] == crashed["send_result"] == {"message_id": delivery["message_id"]}
    print(json.dumps(result, ensure_ascii=False))
