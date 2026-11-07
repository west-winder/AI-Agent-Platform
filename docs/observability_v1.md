# Stage 3 Module 8 Part B1 — Runtime Observability V1

实现日期：2026-11-07。范围：Agent Runtime 的同步、轻量观测事件。

## 组件与接入

`backend/agent_runtime/observability.py` 提供不可变的 `RunContext`、
`ObservationEvent`、同步 `Observer` Protocol、`LoggingObserver`、
`NoOpObserver` 以及统一的 `safe_emit()`。

每次 `run()` 在输入非空校验后创建局部 RunContext，以 UUID4 生成 run_id；
Runtime 实例不保存当前 Run 状态，可以并发处理独立 Run。原循环位于
`_run_loop()`，外围只负责记录正常返回、异常和取消。

Runtime 每轮生成新的 llm_call_id，传入 `_complete()`；该轮的两次尝试及
重试等待共用该 ID。工具使用模型给出的原始 tool_call_id，通过
parent_llm_call_id 指向该轮 LLM 调用。跨 Run 查询工具时使用
`(run_id, tool_call_id)`，因为不同 Run 的原始工具 ID 可能相同。

`AgentRuntime(..., observer=...)` 和 `build_agent_runtime(observer=...)`
支持关键字注入；省略或传 None 时使用 LoggingObserver，显式传 NoOpObserver
时关闭新增观测。原位置参数、ContextBuilder、Registry 和 LLM 注入保持兼容；
MCP Host 无须改变现有调用方式。InMemoryObserver 仅存在于测试。

## 时间、不可变性与日志

RunContext 保存 UTC 开始时间和单调时钟起点。事件时间使用带时区的 UTC
datetime；elapsed_seconds 使用 monotonic 差值，墙钟回拨不会改变耗时。
LLM 原有的 180 秒截止时间仍使用 asyncio 事件循环单调时钟。

事件是 frozen、slots dataclass。attributes 先复制，再通过 MappingProxyType
冻结，只接受白名单字段和 JSON 标量（str/int/float/bool/None）；拒绝嵌套可变
对象及 NaN/Infinity。白名单包括关联 ID、工具名、迭代和尝试计数、状态、
错误类型/代码/分类、HTTP 状态码、重试决策及耗时。

LoggingObserver 在模块 logger 的 INFO 层写入 JSON 消息，字段为 run_id、
event_type、occurred_at 和 attributes，不设置全局 Handler、Formatter 或级别。
部署时沿用现有日志配置并启用该 logger 的 INFO 层。原 Runtime 安全诊断日志
仍保留，补充 run_id 和调用关联字段。

所有新增事件和备用诊断均不记录 Prompt、回答、原始 Arguments/Result、
parse_error、异常消息、原因链或 traceback。ID、工具名、error_code 是结构化
元数据，调用方应遵循现有契约，不把业务内容或凭据塞进这些标识字段。

## 事件契约

| 事件 | attributes（除 run_id 外的关联也在 attributes 中） |
| --- | --- |
| run.started | 无 |
| run.finished | status、iterations、elapsed_seconds |
| run.exception | error_type、elapsed_seconds |
| run.cancelled | elapsed_seconds |
| llm.started | llm_call_id、iteration |
| llm.attempt_failed | llm_call_id、iteration、attempt、error_type、category、status_code、will_retry、elapsed_seconds、remaining_seconds、retry_delay_seconds |
| llm.completed | llm_call_id、iteration、attempts、elapsed_seconds |
| llm.exception | llm_call_id、iteration、attempts、error_type、elapsed_seconds |
| llm.cancelled | llm_call_id、iteration、attempts、elapsed_seconds |
| tool.started | tool_call_id、parent_llm_call_id、tool_name、iteration |
| tool.completed | 工具关联字段、status、error_code、elapsed_seconds |
| tool.exception | 工具关联字段、error_type、elapsed_seconds |
| tool.cancelled | 工具关联字段、elapsed_seconds |

每个正常返回只发出一次 run.finished，status 为 COMPLETED 或
MAX_ITERATIONS_REACHED。工具达到末轮限制时沿用原规则：不执行该批工具，
不产生 tool.started。

llm.completed 表示 LLMPort 调用返回；保留 Runtime 随后的独立响应协议校验。
若自定义 LLMPort 返回无效响应，Runtime 校验失败仍产生 run.exception 并原样
传播，不把校验移入重试边界。

逻辑 LLM 调用数按 llm.started 计数；实际尝试数使用 LLM 终止事件的 attempts，
其中包含成功和被取消的在途尝试。attempt 从 1 起算。这里的尝试指 Runtime
调用 LLMPort.complete 的次数，SDK 内部 HTTP 重试不单独观测。
工具处理数按 tool.started 计数，成功/业务失败按 tool.completed 的 status 分组。

## 失败与取消边界

safe_emit 只捕获 Observer 的普通 Exception，以固定消息记录故障类型、事件类型
和 run_id；备用日志再抛普通 Exception 时也会隔离。不捕获 BaseException，
包括 Observer 主动抛出的 CancelledError、KeyboardInterrupt 和 SystemExit。
Observer 是同步扩展点，实现应保持轻量，不参与 Runtime 决策。

Runtime 既有诊断也通过仅包围日志写入的安全边界输出，防止与
LoggingObserver 共用的 Handler 故障重新进入业务异常路径；该边界不包围
LLM、Dispatcher 或 Handler 的业务执行。

LLM 预算、分类器、至多一次额外重试和最终异常对象均沿用原规则。
预算到期中断在途调用时，补记该尝试的 attempt_failed，再记录 llm.exception；
若预算耗尽于重试等待，只保留之前已记录的失败，不虚增第二次尝试。
主动取消记录 llm.cancelled 和 run.cancelled，取消信号原样传播。

完整工具分支位于 `_process_tool()`，观测边界覆盖参数解析错误的处理、
Dispatcher 调用、JSON 序列化和 History 写入。ToolFailure 记录为
tool.completed(status=error, error_code=...)，循环继续；程序异常记录
tool.exception，取消记录 tool.cancelled，随后均原样传播。

## 范围

本次不修改 Evaluation V1、ToolDispatcher、ToolRegistry、LLMPort 公共契约、
ToolContext 或 AgentRunResult。不引入 OpenTelemetry、数据库日志、远程日志
服务或 Dashboard。Observer 故障时允许事件缺失，不以重试观测事件改变业务行为。

验收命令和结果见 [B1 验证记录](testing/observability_b1_validation.md)。
