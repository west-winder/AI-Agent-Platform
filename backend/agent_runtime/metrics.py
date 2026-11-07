"""Observer 内部的进程内指标状态；同步更新，不保存原始耗时或业务数据。"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.agent_runtime.observability import ObservationEvent

# Run 包含多轮模型/工具调用；LLM 使用既有 180 秒逻辑预算；工具通常更短。
# 边界均为秒，桶采用 <= 上界的累计计数，末尾统一补无上限桶。
RUN_DURATION_BUCKETS = (
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    180.0,
    300.0,
    600.0,
)
LLM_DURATION_BUCKETS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 180.0)
TOOL_DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
RUN_OUTCOMES = ("completed", "max_iterations_reached", "exception", "cancelled")
TOOL_OUTCOMES = ("success", "failure", "exception", "cancelled")


@dataclass(frozen=True, slots=True)
class HistogramSnapshot:
    count: int
    sum: float
    # 字符串上界便于标准 JSON 导出；+Inf 是无上限桶，也等于 count。
    buckets: Mapping[str, int]

    def to_dict(self) -> dict:
        return {"count": self.count, "sum": self.sum, "buckets": dict(self.buckets)}


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    agent_runs_started_total: int
    agent_runs_active: int
    agent_runs_terminal_total: Mapping[str, int]
    agent_run_duration_seconds: Mapping[str, HistogramSnapshot]
    agent_llm_calls_total: int
    agent_llm_attempts_total: int
    agent_llm_retries_total: int
    agent_llm_call_duration_seconds: HistogramSnapshot
    agent_tool_calls_total: Mapping[str, int]
    agent_tool_duration_seconds: Mapping[str, HistogramSnapshot]

    def to_dict(self) -> dict:
        """导出独立的普通字典；标签映射的键只有固定 outcome，无任何关联 ID。"""
        return {
            "counters": {
                "agent_runs_started_total": self.agent_runs_started_total,
                "agent_runs_terminal_total": dict(self.agent_runs_terminal_total),
                "agent_llm_calls_total": self.agent_llm_calls_total,
                "agent_llm_attempts_total": self.agent_llm_attempts_total,
                "agent_llm_retries_total": self.agent_llm_retries_total,
                "agent_tool_calls_total": dict(self.agent_tool_calls_total),
            },
            "gauges": {"agent_runs_active": self.agent_runs_active},
            "histograms": {
                "agent_run_duration_seconds": {
                    outcome: histogram.to_dict()
                    for outcome, histogram in self.agent_run_duration_seconds.items()
                },
                "agent_llm_call_duration_seconds": self.agent_llm_call_duration_seconds.to_dict(),
                "agent_tool_duration_seconds": {
                    outcome: histogram.to_dict()
                    for outcome, histogram in self.agent_tool_duration_seconds.items()
                },
            },
        }


class _Histogram:
    def __init__(self, bounds: tuple[float, ...]) -> None:
        self._bounds = bounds
        self._buckets = [0] * (len(bounds) + 1)
        self._count = 0
        self._sum = 0.0

    def observe(self, value: float) -> None:
        # 只保存固定数量的累计桶，不保留每次观测样本。
        first = bisect_left(self._bounds, value)
        for index in range(first, len(self._buckets)):
            self._buckets[index] += 1
        self._count += 1
        self._sum += value

    def snapshot(self) -> HistogramSnapshot:
        buckets = {
            str(bound): count for bound, count in zip(self._bounds, self._buckets)
        }
        buckets["+Inf"] = self._buckets[-1]
        return HistogramSnapshot(self._count, self._sum, MappingProxyType(buckets))


class MetricsState:
    """由 Observer 持有的普通状态对象，不是独立服务或额外事件入口。

    当前 Runtime 在 asyncio 线程内同步交付事件，record/snapshot 均无 await，
    所以一次更新或快照不会被另一个协程中途打断；不增加线程锁。
    """

    def __init__(self) -> None:
        # 启动 Counter 统计 Run；活跃 Gauge 取此集合长度，ID 只作内部状态管理。
        self._runs_started = 0
        self._active_run_ids: set[str] = set()
        # 四种终态都结束 Run；ToolFailure 本身不代表 Run exception。
        self._run_terminals = dict.fromkeys(RUN_OUTCOMES, 0)
        self._run_durations = {
            outcome: _Histogram(RUN_DURATION_BUCKETS) for outcome in RUN_OUTCOMES
        }
        # 逻辑调用与实际请求分别计数；一次逻辑调用允许含多次实际尝试。
        self._llm_calls = 0
        self._llm_attempts = 0
        self._llm_retries = 0
        self._llm_duration = _Histogram(LLM_DURATION_BUCKETS)
        # 工具按终态计数：failure 是受控业务失败，exception 是非预期程序异常。
        # 不按工具名、error_code 或关联 ID 分组，未知名称不会制造新标签。
        self._tool_calls = dict.fromkeys(TOOL_OUTCOMES, 0)
        self._tool_durations = {
            outcome: _Histogram(TOOL_DURATION_BUCKETS) for outcome in TOOL_OUTCOMES
        }

    @staticmethod
    def _duration(event: ObservationEvent) -> float:
        value = event.attributes["elapsed_seconds"]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Metric duration must be finite and non-negative")
        return float(value)

    def record(self, event: ObservationEvent) -> None:
        kind = event.event_type
        if kind == "run.started":
            if event.run_id not in self._active_run_ids:
                self._active_run_ids.add(event.run_id)
                self._runs_started += 1
            return

        run_outcome = {
            "run.exception": "exception",
            "run.cancelled": "cancelled",
        }.get(kind)
        if kind == "run.finished":
            run_outcome = {
                "COMPLETED": "completed",
                "MAX_ITERATIONS_REACHED": "max_iterations_reached",
            }.get(event.attributes.get("status"))
        if run_outcome is not None:
            # 只有活跃 Run 的首次终态有效，重复/无对应 started 的终态不减 Gauge。
            # 终止即移除 ID，不为去重永久保存所有已结束的 Run。
            if event.run_id in self._active_run_ids:
                duration = self._duration(event)
                self._active_run_ids.remove(event.run_id)
                self._run_terminals[run_outcome] += 1
                self._run_durations[run_outcome].observe(duration)
            return

        if kind == "llm.started":
            # 逻辑调用只在 started 增加，与接下来的重试次数无关。
            self._llm_calls += 1
        elif kind == "llm.attempt_started":
            attempt = event.attributes["attempt"]
            if type(attempt) is not int or attempt < 1:
                raise ValueError("Metric attempt must be a positive integer")
            # 只认真正启动请求的事件；失败时 will_retry=True 不提前增加重试数。
            # 等待重试期间取消时不会出现第二个 attempt_started。
            self._llm_attempts += 1
            if attempt > 1:
                self._llm_retries += 1
        elif kind in ("llm.completed", "llm.exception", "llm.cancelled"):
            # Runtime 提供逻辑调用的单调时钟累计耗时，已包含全部请求和重试等待。
            self._llm_duration.observe(self._duration(event))
        else:
            outcome = {
                "tool.exception": "exception",
                "tool.cancelled": "cancelled",
            }.get(kind)
            if kind == "tool.completed":
                outcome = {"success": "success", "error": "failure"}.get(
                    event.attributes.get("status")
                )
            if outcome is not None:
                duration = self._duration(event)
                self._tool_calls[outcome] += 1
                self._tool_durations[outcome].observe(duration)

    def snapshot(self) -> MetricsSnapshot:
        # 每层映射均先复制再冻结；返回的快照既不暴露活跃集合，也不会随新事件改变。
        return MetricsSnapshot(
            agent_runs_started_total=self._runs_started,
            agent_runs_active=len(self._active_run_ids),
            agent_runs_terminal_total=MappingProxyType(dict(self._run_terminals)),
            agent_run_duration_seconds=MappingProxyType(
                {
                    outcome: histogram.snapshot()
                    for outcome, histogram in self._run_durations.items()
                }
            ),
            agent_llm_calls_total=self._llm_calls,
            agent_llm_attempts_total=self._llm_attempts,
            agent_llm_retries_total=self._llm_retries,
            agent_llm_call_duration_seconds=self._llm_duration.snapshot(),
            agent_tool_calls_total=MappingProxyType(dict(self._tool_calls)),
            agent_tool_duration_seconds=MappingProxyType(
                {
                    outcome: histogram.snapshot()
                    for outcome, histogram in self._tool_durations.items()
                }
            ),
        )
