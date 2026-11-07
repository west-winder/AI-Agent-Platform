"""Case-local fixture state behind the existing tool handler contract."""

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.evaluation.contracts import ToolFixture
from backend.tools.infrastructure.contracts import ToolContext, ToolFailure, ToolHandler
from backend.tools.infrastructure.errors import ToolExecutionError, ToolNotFoundError
from backend.tools.infrastructure.registry import ToolRegistry


class FixtureViolationCode(StrEnum):
    ARGUMENT_MISMATCH = "FIXTURE_ARGUMENT_MISMATCH"
    EXHAUSTED = "FIXTURE_EXHAUSTED"


class InvalidFixtureArguments(ValueError):
    """The fixture parameters do not satisfy the registered tool schema."""


@dataclass(frozen=True, slots=True)
class FixtureViolation:
    tool_name: str
    code: FixtureViolationCode
    fixture_index: int | None = None


def same_json(left: Any, right: Any) -> bool:
    # Python equality equates True, 1 and 1.0, including inside containers.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same_json(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same_json(first, second) for first, second in zip(left, right)
        )
    return left == right


class FixtureTools:
    def __init__(self, fixtures: list[ToolFixture]) -> None:
        self._queues: dict[str, deque[tuple[int, ToolFixture]]] = defaultdict(deque)
        for index, fixture in enumerate(fixtures):
            self._queues[fixture.tool_name].append((index, fixture))
        self._violations: list[FixtureViolation] = []
        self.activity_observed = False

    @property
    def violations(self) -> tuple[FixtureViolation, ...]:
        return tuple(self._violations)

    @property
    def unconsumed_indices(self) -> tuple[int, ...]:
        return tuple(sorted(index for queue in self._queues.values() for index, _ in queue))

    def validate_arguments(self, source: ToolRegistry, case_index: int) -> None:
        for name, queue in self._queues.items():
            try:
                schema = source.get(name).args_schema
            except ToolNotFoundError:
                # Preserve per-case isolation for unavailable tools at assembly.
                continue
            normalized = deque()
            for index, fixture in queue:
                try:
                    arguments = schema.model_validate(fixture.arguments)
                except ValidationError:
                    raise InvalidFixtureArguments(
                        f"Fixture {index} in case {case_index} does not satisfy the tool argument schema"
                    ) from None
                normalized.append((index, fixture.model_copy(update={"arguments": self._json_arguments(arguments)})))
            self._queues[name] = normalized

    @staticmethod
    def _json_arguments(arguments: BaseModel) -> dict[str, Any]:
        return arguments.model_dump(mode="json", by_alias=True, exclude_unset=True)

    def build_registry(self, source: ToolRegistry) -> ToolRegistry:
        registry = ToolRegistry()
        for tool in source.list_tools():
            registry.register(replace(tool, handler=self._handler(tool.name)))
        for name in self._queues:
            # An unknown fixture tool is a Case assembly error.
            registry.get(name)
        return registry

    def _handler(self, tool_name: str) -> ToolHandler:
        async def handle(arguments: BaseModel, context: ToolContext) -> dict[str, Any]:
            self.activity_observed = True
            queue = self._queues.get(tool_name)
            if not queue:
                code = FixtureViolationCode.EXHAUSTED
                self._violations.append(FixtureViolation(tool_name, code))
                raise ToolExecutionError(code.value, "No remaining fixture for this tool")

            index, fixture = queue[0]
            actual = self._json_arguments(arguments)
            if not same_json(actual, fixture.arguments):
                code = FixtureViolationCode.ARGUMENT_MISMATCH
                self._violations.append(FixtureViolation(tool_name, code, index))
                raise ToolExecutionError(code.value, "Arguments do not match the next tool fixture")

            queue.popleft()
            result = fixture.result
            if isinstance(result, ToolFailure):
                raise ToolExecutionError(result.error.code, result.error.message, result.error.retryable)
            return deepcopy(result.data)

        return handle
