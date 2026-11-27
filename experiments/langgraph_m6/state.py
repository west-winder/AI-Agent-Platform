from typing import Literal, TypedDict


ReviewAction = Literal["approve", "reject", "revise"]
EmailStatus = Literal["awaiting_approval", "approved", "rejecting", "revising", "sent", "rejected"]


class EmailInput(TypedDict):
    query: str
    recipient: str


class ApprovalRequest(TypedDict):
    request_id: str
    recipient: str
    email_draft: str
    draft_version: int
    feedback_error: str | None


class SendResult(TypedDict):
    message_id: str


class EmailState(EmailInput, total=False):
    email_draft: str
    draft_version: int
    human_feedback: str
    decision: ReviewAction | None
    approved_version: int | None
    feedback_error: str | None
    rejection_reason: str | None
    idempotency_key: str
    status: EmailStatus
    send_result: SendResult | None
