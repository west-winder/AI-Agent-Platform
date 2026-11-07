"""Runtime 的只读观测契约；仅记录白名单中的标量元数据。"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import math
from time import monotonic
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)

ObservationValue = str | int | float | bool | None
_ATTRIBUTE_NAMES = frozenset({
    "iteration", "iterations", "llm_call_id", "parent_llm_call_id", "tool_call_id",
    "tool_name", "status", "error_code", "error_type", "category", "status_code",
    "attempt", "attempts", "will_retry", "elapsed_seconds", "remaining_seconds",
    "retry_delay_seconds",
})


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    started_at_utc: datetime
    started_monotonic: float

    @classmethod
    def create(cls) -> "RunContext":
        return cls(uuid4().hex, datetime.now(UTC), monotonic())


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    run_id: str
    event_type: str
    occurred_at: datetime
    attributes: Mapping[str, ObservationValue]

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or type(self.event_type) is not str:
            raise TypeError("Observation identifiers must be strings")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))
        # 复制后冻结，调用方不能通过原始 dict 或嵌套对象修改事件。
        attributes = dict(self.attributes)
        for name, value in attributes.items():
            if name not in _ATTRIBUTE_NAMES:
                raise ValueError("Unsupported observation attribute")
            if type(value) not in (str, int, float, bool, type(None)):
                raise TypeError("Observation attributes must be scalar metadata")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("Observation numbers must be finite")
        object.__setattr__(self, "attributes", MappingProxyType(attributes))


class Observer(Protocol):
    def emit(self, event: ObservationEvent) -> None: ...


class LoggingObserver:
    """在现有日志格式内写 JSON 消息，不修改全局 Handler 或 Formatter。"""

    def emit(self, event: ObservationEvent) -> None:
        logger.info(json.dumps({
            "run_id": event.run_id,
            "event_type": event.event_type,
            "occurred_at": event.occurred_at.isoformat(),
            "attributes": dict(event.attributes),
        }, ensure_ascii=False, allow_nan=False))


class NoOpObserver:
    def emit(self, event: ObservationEvent) -> None:
        pass


def safe_emit(observer: Observer, event: ObservationEvent) -> None:
    """隔离普通观测故障及备用日志故障；取消等 BaseException 继续传播。"""
    try:
        observer.emit(event)
    except Exception as exc:
        try:
            logger.warning(
                "agent_runtime.observer_failure",
                extra={
                    "error_type": type(exc).__name__,
                    "event_type": event.event_type,
                    "run_id": event.run_id,
                },
                exc_info=False,
            )
        except Exception:
            pass
