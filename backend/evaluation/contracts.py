"""Provider-independent evaluation inputs, evidence, and results."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from backend.agent_runtime.context_builder import find_current_turn_start
from backend.agent_runtime.contracts import (
    AgentRunResult,
    AgentRunStatus,
    LLMMessage,
    LLMToolCall,
    validate_tool_calls,
)
from backend.agent_runtime.errors import LLMProtocolError
from backend.tools.infrastructure.contracts import ToolFailure, ToolResult, ToolSuccess


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Symbol = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")]
ProfileRef = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
JsonObject = dict[str, JsonValue]
_JSON_OBJECT = TypeAdapter(JsonObject, config=ConfigDict(strict=True, allow_inf_nan=False))
_TaggedToolResult = Annotated[ToolResult, Field(discriminator="status")]
_TOOL_RESULT = TypeAdapter(_TaggedToolResult)


def _validate_tool_result(value: object) -> ToolResult:
    if isinstance(value, (ToolSuccess, ToolFailure)):
        value = value.model_dump()
    result = _TOOL_RESULT.validate_python(value, strict=True, extra="forbid")
    if isinstance(result, ToolSuccess):
        _JSON_OBJECT.validate_python(result.data)
    return result


_EvaluationToolResult = Annotated[_TaggedToolResult, BeforeValidator(_validate_tool_result)]


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, frozen=True, hide_input_in_errors=True
    )


class EvaluatorType(StrEnum):
    RULE = "RULE"
    JUDGE = "JUDGE"


class EvaluationCriterion(_Contract):
    """A symbolic evaluator operation with inert JSON operands."""

    criterion_id: NonBlank
    evaluator_type: EvaluatorType
    criterion_type: Symbol
    required: Annotated[bool, Field(strict=True)] = True
    expected: JsonValue
    rubric: NonBlank | None = None

    @model_validator(mode="after")
    def check_rubric(self) -> Self:
        if (self.evaluator_type is EvaluatorType.JUDGE) != (self.rubric is not None):
            raise ValueError("A rubric is required for JUDGE and only applies to JUDGE")
        return self


class ToolFixture(_Contract):
    """One scripted outcome. Each tool consumes its entries in list order."""

    tool_name: NonBlank
    arguments: JsonObject
    result: _EvaluationToolResult

    @field_validator("result")
    @classmethod
    def check_fixture_failure(cls, result: ToolResult) -> ToolResult:
        if isinstance(result, ToolFailure) and (
            not result.error.code.strip() or not result.error.message.strip()
        ):
            raise ValueError("Fixture failure requires a non-blank error code and message")
        return result


class EvaluationCase(_Contract):
    case_id: NonBlank
    input_messages: list[LLMMessage]
    tool_fixtures: list[ToolFixture] = Field(default_factory=list)
    criteria: list[EvaluationCriterion]
    reference_facts: list[NonBlank] = Field(default_factory=list)
    judge_context: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)] | None = None
    judge_context_required: Annotated[bool, Field(strict=True)] = False
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("input_messages")
    @classmethod
    def check_runtime_input(cls, messages: list[LLMMessage]) -> list[LLMMessage]:
        try:
            find_current_turn_start(messages)
        except LLMProtocolError as exc:
            raise ValueError(str(exc)) from exc
        for message in messages:
            for call in message.tool_calls:
                if call.arguments is not None:
                    _JSON_OBJECT.validate_python(call.arguments)
        return messages

    @model_validator(mode="after")
    def check_criteria(self) -> Self:
        ids = {criterion.criterion_id for criterion in self.criteria}
        if len(ids) != len(self.criteria):
            raise ValueError("Criterion IDs must be unique within a case")
        if not any(criterion.required for criterion in self.criteria):
            raise ValueError("A case must have at least one required criterion")
        return self


class DatasetPurpose(StrEnum):
    HARNESS_REGRESSION = "HARNESS_REGRESSION"
    AGENT_QUALITY = "AGENT_QUALITY"


class EvaluationDataset(_Contract):
    schema_version: Literal[1] = 1
    purpose: DatasetPurpose = DatasetPurpose.HARNESS_REGRESSION
    cases: Annotated[list[EvaluationCase], Field(min_length=1)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def check_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Dataset schema_version must be an integer")
        return value

    @field_validator("cases")
    @classmethod
    def check_case_ids(cls, cases: list[EvaluationCase]) -> list[EvaluationCase]:
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("Case IDs must be unique within a dataset")
        return cases

    @model_validator(mode="after")
    def check_quality_criteria(self) -> Self:
        if self.purpose is DatasetPurpose.AGENT_QUALITY and any(
            not any(item.required and item.evaluator_type is EvaluatorType.JUDGE for item in case.criteria)
            for case in self.cases
        ):
            raise ValueError("AGENT_QUALITY cases require a mandatory Judge criterion for answer quality")
        return self


class EvaluationMode(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    MODEL = "MODEL"


class EvaluationRunConfig(_Contract):
    """References name external profiles; profile contents are resolved separately."""

    mode: EvaluationMode
    max_iterations: PositiveInt = 5
    model_ref: ProfileRef | None = None
    fake_llm_script_ref: ProfileRef | None = None
    judge_ref: ProfileRef | None = None

    @model_validator(mode="after")
    def check_mode(self) -> Self:
        if self.mode is EvaluationMode.DETERMINISTIC:
            if (
                self.fake_llm_script_ref is None
                or self.model_ref is not None
                or self.judge_ref is not None
            ):
                raise ValueError(
                    "DETERMINISTIC requires a script reference and forbids model and judge references"
                )
        elif self.model_ref is None or self.fake_llm_script_ref is not None:
            raise ValueError("MODEL requires a model reference and forbids a script reference")
        return self


def validate_dataset_mode(dataset: EvaluationDataset, mode: EvaluationMode) -> None:
    if mode is EvaluationMode.MODEL and dataset.purpose is not DatasetPurpose.AGENT_QUALITY:
        raise ValueError("MODEL evaluation requires an explicitly declared AGENT_QUALITY dataset")
    if mode is EvaluationMode.DETERMINISTIC and dataset.purpose is DatasetPurpose.AGENT_QUALITY:
        raise ValueError("AGENT_QUALITY datasets require MODEL mode")


class ExecutionStatus(StrEnum):
    RETURNED = "RETURNED"
    RAISED = "RAISED"


class ExceptionInfo(_Contract):
    """Safe exception classification from a trusted producer."""

    exception_type: Symbol


class EvaluationExecutionResult(_Contract):
    """Runtime outcome. The embedded AgentRunResult may contain private history."""

    case_id: NonBlank
    execution_status: ExecutionStatus
    agent_result: AgentRunResult | None = None
    exception_info: ExceptionInfo | None = None
    duration_ms: Annotated[float, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def check_execution_status(self) -> Self:
        if self.execution_status is ExecutionStatus.RETURNED:
            if self.agent_result is None or self.exception_info is not None:
                raise ValueError("RETURNED requires agent_result and forbids exception_info")
        elif self.agent_result is not None or self.exception_info is None:
            raise ValueError("RAISED requires exception_info and forbids agent_result")
        return self


class EvidenceAvailability(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class ToolCallEvidence(_Contract):
    call: LLMToolCall
    tool_result: _EvaluationToolResult | None = None
    skipped: Annotated[bool, Field(strict=True)] = False

    @model_validator(mode="after")
    def check_call(self) -> Self:
        if self.skipped and self.tool_result is not None:
            raise ValueError("A skipped call cannot have a tool result")
        try:
            validate_tool_calls([self.call])
        except LLMProtocolError as exc:
            raise ValueError(str(exc)) from exc
        if self.call.arguments is not None:
            _JSON_OBJECT.validate_python(self.call.arguments)
        return self


class ExecutionEvidence(_Contract):
    """Tool evidence covers the current run and includes skipped requests when known.

    COMPLETE with an empty list confirms no calls. PARTIAL may have no records
    if calls are known to have occurred but their details were lost.
    UNAVAILABLE means call occurrence is unknown.
    """

    final_answer: str | None = None
    run_status: AgentRunStatus | None = None
    iterations: PositiveInt | None = None
    tool_call_availability: EvidenceAvailability = EvidenceAvailability.UNAVAILABLE
    tool_calls: list[ToolCallEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_tool_coverage(self) -> Self:
        if self.tool_call_availability is EvidenceAvailability.UNAVAILABLE and self.tool_calls:
            raise ValueError("UNAVAILABLE tool evidence cannot include call records")
        if self.tool_call_availability is EvidenceAvailability.COMPLETE and any(
            record.tool_result is None and not record.skipped for record in self.tool_calls
        ):
            raise ValueError("COMPLETE tool evidence requires a result or known skip for every call")
        return self


class CriterionDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNDETERMINED = "UNDETERMINED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CaseVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNDETERMINED = "UNDETERMINED"


class CriterionResult(_Contract):
    criterion_id: NonBlank
    decision: CriterionDecision
    rationale: NonBlank | None = None


class EvaluationError(_Contract):
    evaluator_type: EvaluatorType
    error_code: Symbol
    criterion_id: NonBlank | None = None


def aggregate_case_verdict(
    required_criterion_ids: Sequence[str],
    criteria_results: Sequence[CriterionResult],
    evaluation_errors: Sequence[EvaluationError] = (),
) -> CaseVerdict:
    """Pass only when required decisions are complete and free of blocking errors."""
    required = set(required_criterion_ids)
    if not required or len(required) != len(required_criterion_ids):
        raise ValueError("Required criterion IDs must be non-empty and unique")
    decisions = {result.criterion_id: result.decision for result in criteria_results}
    if len(decisions) != len(criteria_results):
        raise ValueError("Criterion result IDs must be unique")
    if any(decisions.get(criterion_id) is CriterionDecision.FAIL for criterion_id in required):
        return CaseVerdict.FAIL
    # A scoped error is nonblocking only when its ID is known to be optional.
    optional = decisions.keys() - required
    if any(error.criterion_id not in optional for error in evaluation_errors) or any(
        decisions.get(criterion_id) is not CriterionDecision.PASS for criterion_id in required
    ):
        return CaseVerdict.UNDETERMINED
    return CaseVerdict.PASS


class EvaluationResult(_Contract):
    case_id: NonBlank
    evaluation_mode: EvaluationMode
    execution_result: EvaluationExecutionResult
    required_criterion_ids: list[NonBlank]
    criteria_results: list[CriterionResult]
    case_verdict: CaseVerdict
    evaluation_errors: list[EvaluationError] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_result(self) -> Self:
        if self.case_id != self.execution_result.case_id:
            raise ValueError("Evaluation and execution case IDs must match")
        verdict = aggregate_case_verdict(
            self.required_criterion_ids, self.criteria_results, self.evaluation_errors
        )
        if self.case_verdict is not verdict:
            raise ValueError("Case verdict must match the required criterion outcomes")
        return self
