from collections.abc import Callable
from typing import Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, interrupt

from .state import ApprovalRequest, EmailInput, EmailState, EmailStatus, ReviewAction, SendResult


SendEmail = Callable[[str, str, str], SendResult]


def draft_email(state: EmailState) -> dict:
    return {
        "email_draft": f"主题：科研合作交流\n\n您好！\n\n想与您交流以下研究事项：{state['query']}\n\n祝好！",
        "draft_version": 1,
        "human_feedback": "",
        "decision": None,
        "approved_version": None,
        "feedback_error": None,
        "rejection_reason": None,
        # 每次新任务生成一次，随草稿写入 Checkpoint，节点重试时沿用原值。
        "idempotency_key": f"m6-email:{uuid4().hex}",
        "status": "awaiting_approval",
        "send_result": None,
    }


def approval_request(state: EmailState) -> ApprovalRequest:
    version = state["draft_version"]
    return {
        "request_id": f"{state['idempotency_key']}/draft/{version}",
        "recipient": state["recipient"],
        "email_draft": state["email_draft"],
        "draft_version": version,
        "feedback_error": state["feedback_error"],
    }


def validate_feedback(request: ApprovalRequest, response: object) -> tuple[ReviewAction, str]:
    if not isinstance(response, dict):
        raise ValueError("反馈必须是对象")
    action = response.get("action")
    if action not in ("approve", "reject", "revise"):
        raise ValueError("action 必须为 approve、reject 或 revise")
    version = response.get("draft_version")
    if type(version) is not int or version != request["draft_version"]:
        raise ValueError("draft_version 必须为当前草稿版本")
    if response.get("request_id") != request["request_id"]:
        raise ValueError("request_id 必须为当前审批请求的标识")
    feedback = response.get("feedback", "")
    if not isinstance(feedback, str) or (action != "approve" and not feedback.strip()):
        raise ValueError("feedback 必须为字符串，拒绝和请求修改时还需填写理由")
    return action, feedback.strip()


def human_approval(state: EmailState) -> dict:
    # 恢复会重跑这里。根据已保存的草稿构造同一请求，随后读取反馈。
    request = approval_request(state)
    response = interrupt(request)
    try:
        action, feedback = validate_feedback(request, response)
    except ValueError as error:
        # 完成当前节点并重新调度 HumanApproval，使下一条反馈进入新的 Interrupt。
        return {
            "status": "awaiting_approval",
            "decision": None,
            "approved_version": None,
            "human_feedback": "",
            "feedback_error": str(error),
        }

    status: dict[ReviewAction, EmailStatus] = {
        "approve": "approved", "reject": "rejecting", "revise": "revising",
    }
    return {
        "decision": action,
        "human_feedback": feedback,
        "approved_version": request["draft_version"] if action == "approve" else None,
        "feedback_error": None,
        "status": status[action],
    }


def route_approval(state: EmailState) -> Literal["HumanApproval", "SendEmail", "Reject", "ReviseEmail"]:
    routes = {"approve": "SendEmail", "reject": "Reject", "revise": "ReviseEmail", None: "HumanApproval"}
    return routes[state["decision"]]


def revise_email(state: EmailState) -> dict:
    return {
        "email_draft": f"{state['email_draft']}\n\n修订说明：{state['human_feedback']}",
        "draft_version": state["draft_version"] + 1,
        "decision": None,
        "approved_version": None,
        "status": "awaiting_approval",
    }


def reject(state: EmailState) -> dict:
    return {"rejection_reason": state["human_feedback"], "status": "rejected"}


def build_email_graph(
    checkpointer: Checkpointer,
    send: SendEmail,
    *,
    after_send: Callable[[str, SendResult], None] | None = None,
):
    def send_email(state: EmailState) -> dict:
        if (
            state["status"] != "approved"
            or state["decision"] != "approve"
            or state["approved_version"] != state["draft_version"]
        ):
            raise ValueError("发送前必须批准当前版本的草稿")

        key = state["idempotency_key"]
        result = send(key, state["recipient"], state["email_draft"])
        if after_send is not None:
            # 故障点位于业务事务提交之后、LangGraph 收到本节点更新之前。
            after_send(key, result)
        return {"status": "sent", "send_result": result}

    builder = StateGraph(EmailState, input_schema=EmailInput)
    builder.add_node("DraftEmail", draft_email)
    builder.add_node("HumanApproval", human_approval)
    builder.add_node("ReviseEmail", revise_email)
    builder.add_node("SendEmail", send_email)
    builder.add_node("Reject", reject)
    builder.add_edge(START, "DraftEmail")
    builder.add_edge("DraftEmail", "HumanApproval")
    builder.add_conditional_edges("HumanApproval", route_approval)
    builder.add_edge("ReviseEmail", "HumanApproval")
    builder.add_edge("SendEmail", END)
    builder.add_edge("Reject", END)
    return builder.compile(checkpointer=checkpointer)
