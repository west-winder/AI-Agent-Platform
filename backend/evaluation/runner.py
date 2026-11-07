"""Serial execution of validated evaluation datasets on AgentRuntime."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json
from time import perf_counter
from typing import overload

from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import LLMRequest, LLMResponse
from backend.agent_runtime.llm_port import LLMPort
from backend.evaluation.contracts import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationExecutionResult,
    EvaluationMode,
    EvaluationRunConfig,
    ExceptionInfo,
    ExecutionEvidence,
    ExecutionStatus,
    validate_dataset_mode,
)
from backend.evaluation.evidence import collect_evidence
from backend.evaluation.fixtures import FixtureTools, FixtureViolation
from backend.tools.bootstrap import build_tool_registry
from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.registry import ToolRegistry


class FakeLLMScriptExhausted(Exception):
    """Runtime requested more completions than the Case script defines."""


class _FakeLLM:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        if not responses or not all(isinstance(response, LLMResponse) for response in responses):
            raise ValueError("A Case script must contain LLMResponse records")
        self._responses = iter(deepcopy(tuple(responses)))

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = next(self._responses, None)
        if response is None:
            raise FakeLLMScriptExhausted("Case script has no remaining responses")
        return response


class ExecutionFailurePhase(StrEnum):
    ASSEMBLY = "ASSEMBLY"
    RUNTIME = "RUNTIME"


@dataclass(frozen=True, slots=True)
class CaseExecution:
    """Execution and diagnostics. Fixture indices refer to the zero-based Case list."""

    execution_result: EvaluationExecutionResult
    evidence: ExecutionEvidence
    failure_phase: ExecutionFailurePhase | None
    fixture_violations: tuple[FixtureViolation, ...]
    unconsumed_fixture_indices: tuple[int, ...]


def content_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BatchProvenance:
    dataset_sha256: str
    run_config_sha256: str
    mode: EvaluationMode
    script_sha256: str | None


_RUNNER_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class ExecutionBatch(Sequence[CaseExecution]):
    """An in-process batch issued by the runner, with immutable record membership."""

    provenance: BatchProvenance
    _records: tuple[CaseExecution, ...]

    def __init__(
        self, provenance: BatchProvenance, records: tuple[CaseExecution, ...], *, _issuer: object = None
    ) -> None:
        if _issuer is not _RUNNER_ISSUER:
            raise TypeError("ExecutionBatch must be issued by EvaluationRunner")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "_records", records)

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: int) -> CaseExecution: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[CaseExecution, ...]: ...

    def __getitem__(self, index: int | slice) -> CaseExecution | tuple[CaseExecution, ...]:
        return self._records[index]


def verify_execution_batch(
    batch: ExecutionBatch, dataset: EvaluationDataset, config: EvaluationRunConfig,
) -> BatchProvenance:
    if type(batch) is not ExecutionBatch:
        raise ValueError("Execution provenance requires an intact runner-issued batch")
    provenance = batch.provenance
    if (
        provenance.mode is not config.mode
        or provenance.dataset_sha256 != content_fingerprint(dataset.model_dump(mode="json"))
        or provenance.run_config_sha256 != content_fingerprint(config.model_dump(mode="json"))
    ):
        raise ValueError("Execution batch provenance does not match the dataset or run config")
    return provenance


class EvaluationRunner:
    """Model factories are caller-owned and may supply an existing DeepSeekAdapter."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        scripts: Mapping[str, Mapping[str, Sequence[LLMResponse]]] | None = None,
        model_factories: Mapping[str, Callable[[], LLMPort]] | None = None,
    ) -> None:
        self._tool_registry = tool_registry if tool_registry is not None else build_tool_registry()
        self._scripts = dict(scripts or {})
        self._model_factories = dict(model_factories or {})

    def _resolve_llm(
        self, dataset: EvaluationDataset, config: EvaluationRunConfig,
    ) -> tuple[Callable[[str], LLMPort], str | None]:
        if config.mode is EvaluationMode.DETERMINISTIC:
            ref = config.fake_llm_script_ref
            if ref is None or ref not in self._scripts:
                raise ValueError("Unknown FakeLLM script profile")
            profile = self._scripts[ref]
            scripts = {
                case.case_id: deepcopy(tuple(profile[case.case_id]))
                for case in dataset.cases if case.case_id in profile
            }
            script_sha256 = content_fingerprint([
                [
                    case.case_id,
                    [asdict(response) for response in scripts[case.case_id]]
                    if case.case_id in scripts else None,
                ]
                for case in dataset.cases
            ])
            return lambda case_id: _FakeLLM(scripts[case_id]), script_sha256
        ref = config.model_ref
        if ref is None or ref not in self._model_factories:
            raise ValueError("Unknown model profile")
        factory = self._model_factories[ref]
        return lambda _case_id: factory(), None

    async def run(
        self, dataset: EvaluationDataset, config: EvaluationRunConfig
    ) -> ExecutionBatch:
        """Resolve the run profile once, then execute cases in dataset order."""
        if not isinstance(dataset, EvaluationDataset) or not isinstance(config, EvaluationRunConfig):
            raise TypeError("run requires a validated EvaluationDataset and EvaluationRunConfig")
        # Snapshot and revalidate mutable collections nested in the frozen contracts.
        dataset = EvaluationDataset.model_validate(dataset.model_dump())
        config = EvaluationRunConfig.model_validate(config.model_dump())
        validate_dataset_mode(dataset, config.mode)
        llm_for_case, script_sha256 = self._resolve_llm(dataset, config)
        provenance = BatchProvenance(
            dataset_sha256=content_fingerprint(dataset.model_dump(mode="json")),
            run_config_sha256=content_fingerprint(config.model_dump(mode="json")),
            mode=config.mode,
            script_sha256=script_sha256,
        )
        prepared = [(case, FixtureTools(case.tool_fixtures)) for case in dataset.cases]
        for index, (_, fixtures) in enumerate(prepared):
            fixtures.validate_arguments(self._tool_registry, index)
        results: list[CaseExecution] = []
        for case, fixtures in prepared:
            results.append(await self._run_case(case, fixtures, config.max_iterations, llm_for_case))
        return ExecutionBatch(provenance, tuple(results), _issuer=_RUNNER_ISSUER)

    async def _run_case(
        self,
        case: EvaluationCase,
        fixtures: FixtureTools,
        max_iterations: int,
        llm_for_case: Callable[[str], LLMPort],
    ) -> CaseExecution:
        started = perf_counter()
        phase = ExecutionFailurePhase.ASSEMBLY
        agent_result = None
        exception_info = None
        try:
            registry = fixtures.build_registry(self._tool_registry)
            llm = llm_for_case(case.case_id)
            if not callable(getattr(llm, "complete", None)):
                raise TypeError("Model factory must return an LLMPort")
            runtime = build_agent_runtime(llm=llm, registry=registry, max_iterations=max_iterations)
            phase = ExecutionFailurePhase.RUNTIME
            agent_result = await runtime.run(case.input_messages, ToolContext())
        except Exception as exc:
            name = type(exc).__name__
            if not (name.isascii() and name.isidentifier() and len(name) <= 128):
                name = "Exception"
            exception_info = ExceptionInfo(exception_type=name)

        execution = EvaluationExecutionResult(
            case_id=case.case_id,
            execution_status=ExecutionStatus.RAISED if exception_info else ExecutionStatus.RETURNED,
            agent_result=agent_result,
            exception_info=exception_info,
            duration_ms=(perf_counter() - started) * 1000,
        )
        return CaseExecution(
            execution_result=execution,
            evidence=collect_evidence(
                execution,
                input_message_count=len(case.input_messages),
                tool_activity_observed=fixtures.activity_observed,
            ),
            failure_phase=phase if exception_info else None,
            fixture_violations=fixtures.violations,
            unconsumed_fixture_indices=fixtures.unconsumed_indices,
        )
