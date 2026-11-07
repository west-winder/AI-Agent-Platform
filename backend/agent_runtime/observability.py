"""Runtime 的只读观测契约；仅记录白名单中的标量元数据。"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import math
from time import monotonic
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from backend.agent_runtime.metrics import MetricsSnapshot, MetricsState

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
            if isinstance(value, str):
                # 工具契约接受 str 子类；取内置字符串值，避免调用自定义 __str__。
                attributes[name] = str.__str__(value)
            elif type(value) not in (int, float, bool, type(None)):
                raise TypeError("Observation attributes must be scalar metadata")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("Observation numbers must be finite")
        object.__setattr__(self, "attributes", MappingProxyType(attributes))


class Observer(Protocol):
    def emit(self, event: ObservationEvent) -> None: ...


class LoggingObserver:
    """统一接收事件；独立隔离日志输出和内部 Metrics 更新。"""

    def __init__(self) -> None:
        self._metrics = MetricsState()

    def emit(self, event: ObservationEvent) -> None:
        # 两个同步分支分别复用 B1 的安全边界，一路普通故障不阻止另一路。
        _safely(lambda: self._write_log(event), run_id=event.run_id, event_type=event.event_type)
        _safely(lambda: self._metrics.record(event), run_id=event.run_id, event_type=event.event_type)

    def snapshot(self) -> MetricsSnapshot:
        return self._metrics.snapshot()

    @staticmethod
    def _write_log(event: ObservationEvent) -> None:
        # 只写 JSON 消息，不修改全局 Handler 或 Formatter。
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
    _safely(lambda: observer.emit(event), run_id=event.run_id, event_type=event.event_type)


def safe_emit_factory(
    observer: Observer, event_factory: Callable[[], ObservationEvent], *, run_id: str, event_type: str,
) -> None:
    """构造与发送共同隔离；构造失败时仍可用关联字段输出安全诊断。"""
    _safely(lambda: observer.emit(event_factory()), run_id=run_id, event_type=event_type)


def _safely(action: Callable[[], None], *, run_id: str, event_type: str) -> None:
    try:
        action()
    except Exception as exc:
        try:
            logger.warning(
                "agent_runtime.observer_failure",
                extra={
                    "error_type": type(exc).__name__,
                    "event_type": event_type,
                    "run_id": run_id,
                },
                exc_info=False,
            )
        except Exception:
            pass


# 默认接线在本进程共享同一 Observer 和指标状态，而不是每个 Runtime/Run 新建。
# 显式构造 LoggingObserver 可为测试或调用方提供独立作用域；NoOp 关闭两路观测。
_default_observer = LoggingObserver()


def get_default_observer() -> LoggingObserver:
    return _default_observer
