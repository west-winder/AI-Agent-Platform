# Stage 3 Module 8 B2 — Metrics V1 设计

日期：2026-11-07（Asia/Shanghai）。状态：**COMPLETE / FROZEN**。

## 数据流与职责

```text
AgentRuntime → 不可变 ObservationEvent → 默认进程共享 LoggingObserver
                                          ├─ 安全日志分支 → JSON 结构化消息
                                          └─ 安全指标分支 → 内部 MetricsState
                                                              ↓
                                                    只读 MetricsSnapshot
```

Observer 是统一事件入口；`metrics.py` 中的 MetricsState 只是 Observer 内部的
普通状态对象，不是独立 Collector 服务。记录具体事件和聚合数字分开实现，方便
理解统计口径，但二者都从同一事件读取安全元数据。

LoggingObserver.emit 分别通过 B1 复用的 `_safely()` 处理两个分支。日志失败后
仍执行 Metrics 更新，Metrics 失败也不会撤销已输出的日志；备用诊断失败仍隔离。
捕获范围仅为普通 Exception，BaseException 继续传播。Runtime 的 safe_emit 外围
边界继续保护自定义 Observer，业务调用不进入观测故障的捕获范围。

## 作用域与并发判断

当前 AgentRuntime 的事件调用是同步的，emit/record/snapshot 不含 await。
MCP Host 启动一个 Runtime，Evaluation Runner 在当前协程逐个组装和运行 Runtime；
没有发现通过线程池、其他线程或多个并行事件循环交付这些事件的生产路径。
同一 asyncio 线程中，协程不会在一次指标更新或快照中途切换，因此没有增加 Lock、
线程池或异步队列。

`get_default_observer()` 返回模块内唯一默认实例，Runtime 构造器省略 observer
时复用它。Bootstrap 和 MCP Host 的既有接线不需改动。所有默认 Runtime、所有
Run 汇入该进程的累计指标；RunContext 仍是每次 Run 局部状态。

显式构造 LoggingObserver 并注入可创建独立作用域，主要用于测试或嵌入调用方；
不在每次 Run 中创建指标实例。NoOpObserver 同时关闭日志与指标。进程重启清零，
多个 worker 各有自身指标；新增跨线程调用路径时需要重新审查同步假设。

## Runtime 最小增量

保留 B1 所有原事件名称、字段及业务含义。只增加：

```text
llm.attempt_started
attributes: llm_call_id, iteration, attempt（从 1 开始）
```

触发位置在原重试循环内，请求 deepcopy 成功之后、每次实际调用
LLMPort.complete 之前。attempt>1 才增加 retries；attempt_failed 中的
will_retry=True 不提前增加任何额外请求计数。因此等待时取消、等待耗尽预算或
请求准备失败都不会凭空产生第二次 Attempt。

LLM 调用预算、分类器、重试次数、工具分支和所有公共契约不变。B1 并发用例
只调整新增事件的序列断言；原来的 ID 关联、异常、取消、不可变性和安全测试保留。

## 状态、标签与快照

10 项指标的完整口径、桶边界、标签和示例见 [指标目录](metrics_catalog.md)。
标签仅为固定 outcome；不按工具名、ID 或错误代码分组，任意未知工具名称不会
增加指标序列。Run 终态通过活跃 ID 集合过滤重复事件，移除 ID 后释放内存；
该集合不暴露给 Snapshot，也不保存历史 Run。LLM 和 Tool 终态沿用 Runtime
单次交付语义，不建立永久 ID 去重表。

Histogram 保存 count、sum 和固定累计桶，不存原始样本；无上限桶以 +Inf 表示。
Run、LLM、Tool 分别选择适合其延迟范围的固定边界，所有观测值来自 Runtime
单调时钟耗时。输入耗时必须为有限非负数，先校验再更新，避免非法事件部分修改
Run 终态计数。

Snapshot 在同一线程内同步复制状态，返回 frozen dataclass 和复制后的只读映射。
to_dict 再转换成独立 JSON 兼容结构，按 Counter/Gauge/Histogram 分组。
不增加服务层、HTTP 接口、数据库、远程指标或监控基础设施。

## 冻结边界与后续事项

Evaluation V1、MCP Host、ToolDispatcher、ToolRegistry、LLMPort、DeepSeekAdapter、
ToolContext 和 AgentRunResult 保持不变。真实 Judge 校准仍未执行，Evaluation
的 COMPLETE/FROZEN 指既有 Offline V1 工程范围；指标不代替质量评测。

B1 Review 的字符串长度约束和失败事件累计耗时字段命名保持为非阻断后续项，
本轮不修改字段名或扩大协议。Observer 故障允许丢失事件，缺失终态可能留下活跃
计数；V1 不提供自动对账、重放或持久化恢复，也不以修复指标改变业务行为。

验证证据见 [B2 与 Module 8 最终验收](testing/module8_final_acceptance.md)。
