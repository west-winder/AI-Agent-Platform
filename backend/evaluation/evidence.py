"""Extract only the current run's observed evidence from Runtime history."""

import json

from pydantic import ValidationError

from backend.agent_runtime.contracts import AgentRunStatus
from backend.evaluation.contracts import (
    EvaluationExecutionResult,
    EvidenceAvailability,
    ExecutionEvidence,
    ToolCallEvidence,
)


def collect_evidence(
    execution: EvaluationExecutionResult,
    *,
    input_message_count: int,
    tool_activity_observed: bool = False,
) -> ExecutionEvidence:
    """Read the suffix appended by Runtime, using fixture activity only if it raised."""
    result = execution.agent_result
    if result is None:
        return ExecutionEvidence(
            tool_call_availability=(
                EvidenceAvailability.PARTIAL
                if tool_activity_observed
                else EvidenceAvailability.UNAVAILABLE
            )
        )

    calls: list[ToolCallEvidence] = []
    pending: dict[str, int] = {}
    complete = result.status is AgentRunStatus.COMPLETED
    for message in result.messages[input_message_count:]:
        if message.role == "assistant":
            if pending:
                complete = False
                pending.clear()
            for call in message.tool_calls:
                try:
                    record = ToolCallEvidence(call=call)
                except ValidationError:
                    complete = False
                    continue
                pending[call.id] = len(calls)
                calls.append(record)
        elif message.role == "tool":
            call_id = message.tool_call_id
            index = pending.pop(call_id, None) if call_id is not None else None
            if index is None or message.content is None:
                complete = False
                continue
            try:
                outcome = json.loads(message.content)
                if not isinstance(outcome, dict):
                    complete = False
                    continue
                calls[index] = ToolCallEvidence(call=calls[index].call, tool_result=outcome)
            except (json.JSONDecodeError, ValidationError):
                complete = False

    return ExecutionEvidence(
        final_answer=result.final_answer,
        run_status=result.status,
        iterations=result.iterations,
        tool_call_availability=(
            EvidenceAvailability.COMPLETE if complete and not pending else EvidenceAvailability.PARTIAL
        ),
        tool_calls=calls,
    )
