"""Provider-independent request and response for semantic criteria."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from backend.evaluation.contracts import CriterionDecision, EvaluationCriterion, JsonObject


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    user_query: str
    agent_answer: str
    reference_facts: tuple[str, ...]
    relevant_evidence: JsonObject
    semantic_criteria: tuple[EvaluationCriterion, ...]
    judge_context: str | None = None


class JudgeCriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    criterion_id: Annotated[str, StringConstraints(min_length=1)]
    decision: Literal[CriterionDecision.PASS, CriterionDecision.FAIL, CriterionDecision.UNDETERMINED]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1024)]


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    criteria_results: list[JudgeCriterionResult]


class JudgePort(Protocol):
    async def judge(self, request: JudgeRequest) -> JudgeVerdict: ...


class JudgeErrorCode(StrEnum):
    PROTOCOL_ERROR = "JUDGE_PROTOCOL_ERROR"
    CRITERION_IDS = "JUDGE_CRITERION_IDS"
    TIMEOUT = "JUDGE_TIMEOUT"
    API_ERROR = "JUDGE_API_ERROR"
    EXECUTION_FAILED = "JUDGE_EXECUTION_FAILED"
    TRUNCATED = "JUDGE_TRUNCATED"
    EMPTY_RESPONSE = "JUDGE_EMPTY_RESPONSE"


class JudgeError(Exception):
    """Known Judge failure with a fixed, safe error code."""

    def __init__(self, code: str) -> None:
        self.code = JudgeErrorCode(code)
        super().__init__(self.code.value)


def validate_judge_verdict(request: JudgeRequest, verdict: JudgeVerdict) -> JudgeVerdict:
    if not isinstance(verdict, JudgeVerdict):
        raise JudgeError("JUDGE_PROTOCOL_ERROR")
    try:
        verdict = JudgeVerdict.model_validate(verdict)
    except ValidationError:
        raise JudgeError("JUDGE_PROTOCOL_ERROR") from None
    requested = {criterion.criterion_id for criterion in request.semantic_criteria}
    returned = {result.criterion_id for result in verdict.criteria_results}
    if returned != requested or len(returned) != len(verdict.criteria_results):
        raise JudgeError("JUDGE_CRITERION_IDS")
    return verdict
