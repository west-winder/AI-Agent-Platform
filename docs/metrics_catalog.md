# Agent Runtime Metrics V1 指标目录

状态：**COMPLETE / FROZEN**。日期：2026-11-07（Asia/Shanghai）。

## 公共统计约定

默认 `get_default_observer()` 在进程内共享，所有默认 Runtime 的事件汇入同一份
指标。重启进程后计数清零；不跨进程汇总。显式注入 LoggingObserver 可选独立
作用域，NoOpObserver 关闭日志与指标两路观测。

唯一标签为固定的 outcome（终态分类）。不使用 run_id、llm_call_id、tool_call_id、
user_id、conversation_id、工具名称、错误代码或任意异常类型作为指标标签。
Snapshot 的 outcome 映射键就是标签值；不带标签的指标是单个标量或 Histogram。

耗时由 Runtime 通过单调时钟计算并放入终止事件；聚合器不读取墙钟重新计时。
Counter 累计增加；Gauge 表示当前数量；Histogram 使用累计桶（值 <= 上界）。
指标只处理同步事件，不影响重试、业务成功判定或取消传播。

## 1. agent_runs_started_total

| 字段 | 内容 |
| --- | --- |
| 中文名称 | 已启动 Agent Run 总数 |
| 指标名称 | `agent_runs_started_total` |
| 类型 | Counter |
| 单位 | 次 |
| 统计对象 | Run |
| 更新时机 | run.started 到达，run_id 尚未在活跃集合中时 +1 |
| 标签 | 无 |
| 边界语义 | 非空输入校验失败尚未创建 Run，不计数；同一活跃 Run 重复 started 忽略 |
| 示例 | Run A、B 各启动一次，总数 +2；A 发生 LLM 重试不增加该数 |

## 2. agent_runs_active

| 字段 | 内容 |
| --- | --- |
| 中文名称 | 当前活跃 Agent Run 数 |
| 指标名称 | `agent_runs_active` |
| 类型 | Gauge |
| 单位 | 个 |
| 统计对象 | 已启动但未终止的 Run |
| 更新时机 | started 加入活跃集合时 +1；首次有效终态移除时 -1，快照读取集合长度 |
| 标签 | 无 |
| 边界语义 | completed、max_iterations_reached、exception、cancelled 都减少；重复或无 started 的终态不减少，不产生负值 |
| 示例 | A、B 同时执行为 2；A 完成为 1；B 取消后为 0 |

run_id 仅保存在内部活跃集合，终止即移除，不进入 Snapshot 或标签，也不永久保存
已终止 ID。Runtime 保证每次 Run 使用新 UUID；V1 不处理已结束 UUID 的重放。

## 3. agent_runs_terminal_total

| 字段 | 内容 |
| --- | --- |
| 中文名称 | Agent Run 各终态累计次数 |
| 指标名称 | `agent_runs_terminal_total` |
| 类型 | Counter |
| 单位 | 次 |
| 统计对象 | Run |
| 更新时机 | 活跃 Run 的首次 run.finished / run.exception / run.cancelled 到达时对应 outcome +1 |
| 标签 | outcome：completed、max_iterations_reached、exception、cancelled |
| 边界语义 | 前两项属于正常返回；后两项分别为异常和主动取消；ToolFailure 不直接增加 exception；重复终态忽略 |
| 示例 | 工具返回 ToolFailure 后模型正常回答：completed +1、exception +0 |

## 4. agent_run_duration_seconds

| 字段 | 内容 |
| --- | --- |
| 中文名称 | Agent Run 完整执行耗时分布 |
| 指标名称 | `agent_run_duration_seconds` |
| 类型 | Histogram |
| 单位 | 秒；count 单位为次 |
| 统计对象 | Run |
| 更新时机 | 与有效 Run 终态 Counter 同时记录 elapsed_seconds |
| 标签 | outcome：completed、max_iterations_reached、exception、cancelled |
| 边界语义 | 包含上下文构建、多轮模型、重试等待和工具处理；异常和取消也记录；重复终态不重复采样 |
| 示例 | completed 的 2 秒、4 秒两个 Run：count=2、sum=6、平均耗时=3 秒 |

## 5. agent_llm_calls_total

| 字段 | 内容 |
| --- | --- |
| 中文名称 | 逻辑 LLM 调用累计次数 |
| 指标名称 | `agent_llm_calls_total` |
| 类型 | Counter |
| 单位 | 次 |
| 统计对象 | 逻辑 LLM Call |
| 更新时机 | 每个 llm.started 到达时 +1 |
| 标签 | 无 |
| 边界语义 | 同一逻辑调用的所有尝试共用 ID，只计一次；后续请求准备失败或取消仍是已启动的逻辑调用 |
| 示例 | A 超时后重试成功，B 首次成功：calls +2 |

## 6. agent_llm_attempts_total

| 字段 | 内容 |
| --- | --- |
| 中文名称 | 实际 LLM 请求尝试累计次数 |
| 指标名称 | `agent_llm_attempts_total` |
| 类型 | Counter |
| 单位 | 次 |
| 统计对象 | Runtime 对 LLMPort.complete 的一次实际调用 |
| 更新时机 | 请求快照准备成功后、调用 LLMPort.complete 前的 llm.attempt_started，每次 +1 |
| 标签 | 无 |
| 边界语义 | 成功、失败及在途取消的实际请求都计数；重试等待中取消不增加下一次尝试；SDK 内部 HTTP 重试不计入；请求拷贝失败不计为实际请求 |
| 示例 | A 首次超时、第二次成功，B 首次成功：attempts +3 |

## 7. agent_llm_retries_total

| 字段 | 内容 |
| --- | --- |
| 中文名称 | 实际启动的额外 LLM 请求累计次数 |
| 指标名称 | `agent_llm_retries_total` |
| 类型 | Counter |
| 单位 | 次 |
| 统计对象 | 同一逻辑调用中 attempt > 1 的实际请求 |
| 更新时机 | llm.attempt_started 的 attempt > 1 时 +1 |
| 标签 | 无 |
| 边界语义 | will_retry=True 只是当时的重试决策，不触发计数；等待时取消或预算耗尽不虚增；不统计 SDK 内部重试 |
| 示例 | A 超时后真正启动第二次请求：retries +1；若等待时取消：retries +0 |

## 8. agent_llm_call_duration_seconds

| 字段 | 内容 |
| --- | --- |
| 中文名称 | 逻辑 LLM 调用完整耗时分布 |
| 指标名称 | `agent_llm_call_duration_seconds` |
| 类型 | Histogram |
| 单位 | 秒；count 单位为次 |
| 统计对象 | 逻辑 LLM Call |
| 更新时机 | llm.completed / llm.exception / llm.cancelled 到达时记录 elapsed_seconds |
| 标签 | 无 |
| 边界语义 | 全部实际请求和重试等待共同计时，每次逻辑调用最终记录一个样本；在途预算超时属于 exception，主动取消属于 cancelled，但汇入同一耗时分布 |
| 示例 | 首次请求 2 秒、等待 30 秒、第二次请求 3 秒：count +1、sum +35 秒 |

B1 的 llm.started 和 llm.attempt_failed 语义没有改变；attempt_failed 的
elapsed_seconds 仍表示该逻辑调用截至当时的累计耗时，不作为独立 Attempt 耗时。
指标不依赖该失败事件或终态中的 attempts 汇总来推测实际请求开始次数。

## 9. agent_tool_calls_total

| 字段 | 内容 |
| --- | --- |
| 中文名称 | 工具各结果累计调用次数 |
| 指标名称 | `agent_tool_calls_total` |
| 类型 | Counter |
| 单位 | 次 |
| 统计对象 | Runtime 的一次完整 Tool Call 处理 |
| 更新时机 | tool.completed / tool.exception / tool.cancelled 到达时对应 outcome +1 |
| 标签 | outcome：success、failure、exception、cancelled |
| 边界语义 | ToolSuccess→success；ToolFailure（含解析、Schema、未知工具、受控执行失败）→failure；非预期异常→exception；取消→cancelled；最后一轮未执行的工具不计数 |
| 示例 | A ToolSuccess、B ToolFailure，模型最终完成：success +1、failure +1、Run completed +1、Run exception +0 |

工具名完全不作标签，未知工具和任意新增名字都进入同一固定 outcome 系列。

## 10. agent_tool_duration_seconds

| 字段 | 内容 |
| --- | --- |
| 中文名称 | 工具完整处理耗时分布 |
| 指标名称 | `agent_tool_duration_seconds` |
| 类型 | Histogram |
| 单位 | 秒；count 单位为次 |
| 统计对象 | Runtime 的一次完整 Tool Call 处理 |
| 更新时机 | 工具终态到达时记录 elapsed_seconds，与调用 Counter 相同 outcome |
| 标签 | outcome：success、failure、exception、cancelled |
| 边界语义 | 沿用 B1 边界，包含参数解析失败处理、Dispatcher、结果序列化和 History 写入；异常或取消也记录 |
| 示例 | failure 的工具处理花费 0.5 秒：该 outcome 的 count +1、sum +0.5 秒 |

## 固定桶、平均值与分位数基础

| Histogram | 有限上界（秒，均为累计 <=） | 选择原因 |
| --- | --- | --- |
| Run | 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 180, 300, 600 | 多轮执行可超过单个 LLM 的预算，覆盖毫秒到多分钟 |
| LLM Call | 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 180 | 兼顾快速返回和既有 180 秒逻辑调用预算 |
| Tool Call | 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60 | 工具常比模型快，细分毫秒段，同时覆盖网络工具等待 |

每组都附加 `+Inf` 桶，接受超过最大有限上界的值，计数等于 count。
Snapshot 的桶键使用字符串，如 `"1.0"`、`"+Inf"`，方便标准 JSON 导出。
count 为样本数，sum 为秒数总和；count>0 时平均耗时=sum/count，无样本时平均值
未定义。累计桶支持后续对 P50/P95/P99 定位或插值估算，V1 不输出精确分位数；
落在 +Inf 桶的分位数只能确定超过最大有限边界，不能推断精确值。

不保存原始耗时数组；四个 Run outcome、一个 LLM 分布、四个 Tool outcome 的
桶数固定。唯一随并发量变化的内部集合是活跃 Run ID 集合。

## Snapshot 读取与导出

```python
from backend.agent_runtime.observability import get_default_observer

snapshot = get_default_observer().snapshot()
active = snapshot.agent_runs_active
completed = snapshot.agent_runs_terminal_total["completed"]
histogram = snapshot.agent_llm_call_duration_seconds
mean_seconds = histogram.sum / histogram.count if histogram.count else None
projection = snapshot.to_dict()  # counters / gauges / histograms 三组、共 10 个指标
```

Snapshot 是独立的冻结 dataclass，所有嵌套映射也复制并冻结；后续事件不会改变
旧 Snapshot。to_dict 返回新的普通字典，可用 `json.dumps(..., allow_nan=False)`
导出；修改导出字典不影响 Snapshot 或内部状态。没有 HTTP 端点或持久化服务。

Observer 普通故障可能导致部分事件未计入，指标是尽力记录，不能作为业务决策或
计费账本。失败后的补偿、事件重放、跨线程访问、多进程合并均不在 V1 范围内。
