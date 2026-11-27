import socket
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from experiments.langgraph_m6 import workflow
from experiments.langgraph_m6.workflow import build_email_graph


EMAIL = {"query": "交流石墨烯样品表征结果", "recipient": "researcher@example.test"}
CONFIG = {"configurable": {"thread_id": "m6-offline-review"}}
RESULT = {"message_id": "mock-message-1"}
DRAFT = "主题：科研合作交流\n\n您好！\n\n想与您交流以下研究事项：交流石墨烯样品表征结果\n\n祝好！"


@pytest.fixture
def lab(monkeypatch):
    def deny_network(*args, **kwargs):
        pytest.fail("Part A 不应建立网络连接")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    send = Mock(return_value=RESULT)
    return build_email_graph(InMemorySaver(), send), send


def start(graph):
    return graph.invoke(EMAIL, CONFIG, version="v2", durability="sync")


def resume(graph, feedback, **kwargs):
    return graph.invoke(Command(resume=feedback), CONFIG, version="v2", durability="sync", **kwargs)


def feedback_for(output, action="approve", **fields):
    request = output.interrupts[0].value
    return {"action": action, "draft_version": request["draft_version"], "request_id": request["request_id"], **fields}


def test_a1_start_pauses_and_exposes_draft_and_snapshot(lab):
    graph, send = lab
    output = start(graph)
    assert len(output.interrupts) == 1
    pending = graph.get_state(CONFIG)
    assert pending.interrupts == output.interrupts
    assert pending.next == ("HumanApproval",)
    assert pending.config["configurable"]["checkpoint_id"]
    assert pending.values == output.value
    assert "thread_id" not in pending.values
    assert pending.values["status"] == "awaiting_approval"
    assert pending.values["send_result"] is None
    assert output.interrupts[0].value == {
        "request_id": f"{pending.values['idempotency_key']}/draft/1",
        "recipient": EMAIL["recipient"], "email_draft": DRAFT, "draft_version": 1, "feedback_error": None,
    }
    send.assert_not_called()


def test_a2_approval_sends_and_finishes(lab):
    graph, send = lab
    pending = start(graph)
    final = resume(graph, feedback_for(pending, feedback="  同意发送  "))
    assert final.interrupts == ()
    assert graph.get_state(CONFIG).next == ()
    assert final.value["decision"] == "approve"
    assert final.value["approved_version"] == final.value["draft_version"] == 1
    assert final.value["human_feedback"] == "同意发送"
    assert final.value["status"] == "sent"
    assert final.value["send_result"] == RESULT
    send.assert_called_once_with(pending.value["idempotency_key"], EMAIL["recipient"], DRAFT)


def test_a2_send_node_checks_approval_version_again(lab):
    graph, send = lab
    pending = start(graph)
    resume(graph, feedback_for(pending), interrupt_before=["SendEmail"])
    assert graph.get_state(CONFIG).next == ("SendEmail",)
    graph.update_state(CONFIG, {"approved_version": 0}, as_node="HumanApproval")
    with pytest.raises(ValueError, match="发送前必须批准当前版本"):
        graph.invoke(None, CONFIG, durability="sync")
    send.assert_not_called()


def test_a3_reject_lands_in_reject_node_without_sending(lab):
    graph, send = lab
    final = resume(graph, feedback_for(start(graph), "reject", feedback="研究结果尚未公开"))
    assert final.interrupts == ()
    assert graph.get_state(CONFIG).next == ()
    assert final.value["status"] == "rejected"
    assert final.value["decision"] == "reject"
    assert final.value["rejection_reason"] == "研究结果尚未公开"
    assert final.value["approved_version"] is final.value["send_result"] is None
    send.assert_not_called()


def test_a4_revision_requires_new_review_and_sends_only_latest_draft(lab):
    graph, send = lab
    first = start(graph)
    revised = resume(graph, feedback_for(first, "revise", feedback="请补充会议安排"))
    assert len(revised.interrupts) == 1
    assert revised.interrupts[0].id != first.interrupts[0].id
    request = revised.interrupts[0].value
    assert request["request_id"] != first.interrupts[0].value["request_id"]
    assert request["email_draft"] == f"{DRAFT}\n\n修订说明：请补充会议安排"
    assert revised.value["draft_version"] == request["draft_version"] == 2
    assert revised.value["idempotency_key"] == first.value["idempotency_key"]
    assert revised.value["decision"] is revised.value["approved_version"] is None
    assert revised.value["status"] == "awaiting_approval"
    assert graph.get_state(CONFIG).next == ("HumanApproval",)
    send.assert_not_called()

    final = resume(graph, feedback_for(revised))
    assert final.value["status"] == "sent"
    assert final.value["approved_version"] == 2
    send.assert_called_once_with(first.value["idempotency_key"], EMAIL["recipient"], request["email_draft"])


def test_a5_resume_reexecutes_code_before_interrupt_without_redrafting(lab, monkeypatch):
    graph, send = lab
    build_request = Mock(wraps=workflow.approval_request)
    monkeypatch.setattr(workflow, "approval_request", build_request)
    pending = start(graph)
    assert build_request.call_count == 1
    send.assert_not_called()

    events = graph.stream(
        Command(resume=feedback_for(pending)), CONFIG, stream_mode="tasks", version="v2", durability="sync",
    )
    started = [event["data"]["name"] for event in events if "input" in event["data"]]
    assert started == ["HumanApproval", "SendEmail"]
    assert build_request.call_count == 2
    assert build_request.call_args_list[0] == build_request.call_args_list[1]
    assert graph.get_state(CONFIG).values["idempotency_key"] == pending.value["idempotency_key"]
    send.assert_called_once()


@pytest.mark.parametrize(("changes", "error_field"), [
    ({"action": "send"}, "action"),
    ({"draft_version": None}, "draft_version"),
    ({"draft_version": 9}, "draft_version"),
    ({"draft_version": True}, "draft_version"),
    ({"request_id": "other-task/draft/1"}, "request_id"),
    ({"request_id": None}, "request_id"),
    ({"action": "reject", "feedback": " "}, "feedback"),
    ({"action": "revise", "feedback": 123}, "feedback"),
])
def test_a6_invalid_feedback_reopens_review_and_accepts_valid_feedback(lab, changes, error_field):
    graph, send = lab
    pending = start(graph)
    response = feedback_for(pending)
    response.update(changes)
    for field, value in changes.items():
        if value is None:
            del response[field]
    retry = resume(graph, response)
    assert len(retry.interrupts) == 1
    assert retry.interrupts[0].id != pending.interrupts[0].id
    assert error_field in retry.interrupts[0].value["feedback_error"]
    assert retry.interrupts[0].value["request_id"] == pending.interrupts[0].value["request_id"]
    assert retry.value["decision"] is retry.value["approved_version"] is None
    assert retry.value["status"] == "awaiting_approval"
    assert graph.get_state(CONFIG).next == ("HumanApproval",)
    send.assert_not_called()

    final = resume(graph, feedback_for(retry))
    assert final.value["status"] == "sent"
    assert final.value["feedback_error"] is None
    send.assert_called_once()


def test_a6_old_approval_cannot_apply_to_revised_draft(lab):
    graph, send = lab
    first = start(graph)
    revised = resume(graph, feedback_for(first, "revise", feedback="请增加摘要"))
    stale_feedback = feedback_for(first)
    retry = resume(graph, stale_feedback)
    assert retry.interrupts[0].id != revised.interrupts[0].id
    assert retry.interrupts[0].value["draft_version"] == 2
    assert "draft_version" in retry.interrupts[0].value["feedback_error"]

    # 即使替换版本号，第一版的审批请求 ID 也不能用于第二版。
    retry = resume(graph, {**stale_feedback, "draft_version": 2})
    assert "request_id" in retry.interrupts[0].value["feedback_error"]
    assert retry.value["approved_version"] is None
    send.assert_not_called()
    final = resume(graph, feedback_for(retry))
    assert final.value["approved_version"] == 2
    assert final.value["status"] == "sent"
    send.assert_called_once()
