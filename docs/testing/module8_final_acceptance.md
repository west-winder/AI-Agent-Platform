# Stage 3 Module 8 — B2 Metrics V1 与最终工程验收

日期：2026-11-07（Asia/Shanghai）。结论：**Module 8 COMPLETE / FROZEN**。

## 最终状态

| 模块 | 最终状态 |
| --- | --- |
| Evaluation V1 | COMPLETE / FROZEN（既有 Offline V1 范围） |
| Observability B1 | COMPLETE / FROZEN |
| Metrics B2 | COMPLETE / FROZEN |
| Module 8 | COMPLETE / FROZEN |

Stage 3 整体仍待用户另行进行系统性复盘和最终确认，本轮未开始复盘。

## 实际基线与 Git 范围

开始时分支为 main，HEAD 为 `3f2a8839efc4bd49e1e608685511809b9dbe17c8`。
执行 git fetch origin main 后，本地与 origin/main 差异为 0/0。检查全部差异后
确认仅有已验收 B1 的六个文件，没有无关修改或预先暂存的内容。

先复跑 B1 完整相关回归：**947 passed in 16.32s**，再独立 Commit/Push：

- B1 Commit：`ed614ecadee5baaabec7a1f19e25cd37215cf945`
- 标题：`feat(observability): add runtime logging and tracing v1`
- Push：origin/main 从 3f2a883 更新到 ed614ec，成功；之后工作区干净、分支同步。

B2 以该提交为基线。B2 提交标题为 `feat(metrics): complete process-local metrics v1`；
完整 B2 Hash、最终 Push 和同步状态以最终交付消息及 Git 记录为准，本报告随同
B2 提交。仅精确暂存以下 B2 文件，不包含环境、缓存、凭据或无关文件。

| 文件 | 改动 |
| --- | --- |
| backend/agent_runtime/metrics.py | 新增内部聚合状态、固定累计 Histogram、只读快照及 JSON 兼容导出 |
| backend/agent_runtime/observability.py | Observer 内日志/指标两路安全更新、默认进程共享实例和 snapshot |
| backend/agent_runtime/runtime.py | 默认实例接线、请求准备完成后增加 llm.attempt_started；不重写 Agent Loop |
| tests/agent_runtime/test_metrics.py | 新增 Metrics 专项测试 |
| tests/agent_runtime/test_observability.py | 仅兼容新增实际尝试开始事件的并发序列断言 |
| docs/metrics_catalog.md | 10 项指标定义、标签、单位、更新边界、示例和桶设计 |
| docs/metrics_v1.md | B2 架构、同步假设、事件增量、故障及范围 |
| docs/observability_v1.md | 同步 B2 默认作用域和新增事件，保留 B1 语义说明 |
| docs/stage3_status.md | Stage 3 当前路线图与 Module 8 冻结状态 |
| README.md | 同步 Module 8 状态和文档入口 |
| 本报告 | B2 和 Module 8 最终工程验收证据 |

## 10 项指标最终口径

| 指标 | 类型 / 口径 |
| --- | --- |
| agent_runs_started_total | Counter：新活跃 Run 的 run.started +1 |
| agent_runs_active | Gauge：内部活跃 Run ID 集合长度；四种首次终态都移除，重复终态忽略 |
| agent_runs_terminal_total | Counter：首次有效终态按 completed / max_iterations_reached / exception / cancelled 分类 |
| agent_run_duration_seconds | Histogram：上述同一有效终态的完整单调时钟耗时，按 Run outcome 分组 |
| agent_llm_calls_total | Counter：llm.started，每次逻辑调用一次 |
| agent_llm_attempts_total | Counter：llm.attempt_started，每个真正开始的 LLMPort.complete 请求一次 |
| agent_llm_retries_total | Counter：实际请求开始且 attempt>1，等待时取消或预算耗尽不虚增 |
| agent_llm_call_duration_seconds | Histogram：逻辑调用 completed / exception / cancelled 的累计耗时，包含重试等待 |
| agent_tool_calls_total | Counter：工具终态按 success / failure / exception / cancelled 分类；ToolFailure 不变为 Run exception |
| agent_tool_duration_seconds | Histogram：工具终态的完整处理耗时，与工具 Counter 使用相同 outcome |

所有 Histogram 都有 count、sum、固定累计有限桶和 +Inf 桶，不存原始样本。
不把任何关联 ID、工具名、错误代码或异常类型作为指标标签。Snapshot 逐层复制冻结，
to_dict 再返回独立的 JSON 兼容字典。详细中文定义见 [指标目录](../metrics_catalog.md)。

## Metrics 专项覆盖

31 项参数化展开后的用例覆盖：

- Run：实时启动数和 Active Gauge、四种终态、重复 started/terminal、无对应 started 的终态。
- LLM：逻辑调用与实际尝试、真实请求内可见的计数、重试成功与最终失败、等待取消不虚增、完整等待耗时、请求准备失败不计实际请求。
- Tool：success/failure/exception/cancelled、业务失败后继续完成 Run、未知工具归类、工具耗时。
- Histogram：三类分布的 count/sum、边界包含、累计桶、超过最大有限上界的 +Inf 桶。
- 架构与安全：并发 Run、跨 Runtime 默认共享、显式 NoOp/独立作用域、1000 个不同 ID/未知工具不扩张标签、只读快照和导出副本隔离。
- 故障：日志及备用诊断同时失败仍更新 Metrics，Metrics 失败仍输出日志并保留原始业务异常，BaseException 继续传播，非法耗时不部分终止 Run。

B1 的 44 项原观测回归继续保留，包括完整工具处理分支计时、协议验证边界、
原始异常对象、主动取消和预算超时区分、不可变性及日志内容排除。

## 验收命令与结果

```powershell
.\.venv\Scripts\python.exe -m pytest tests/agent_runtime/test_metrics.py -q
.\.venv\Scripts\python.exe -m pytest tests/agent_runtime tests/mcp_host tests/evaluation tests/tools -q --basetemp=D:/AI_Agent_Platform/.pytest_cache/m8b2-final --junitxml=D:/AI_Agent_Platform/.pytest_cache/m8b2-final-tests.xml
.\.venv\Scripts\python.exe -m pip check
uv --cache-dir D:/AI_Agent_Platform/.pytest_cache/m8b2-uv tool run ruff check --select E9,F backend/agent_runtime/metrics.py backend/agent_runtime/observability.py backend/agent_runtime/runtime.py tests/agent_runtime/test_metrics.py tests/agent_runtime/test_observability.py
uv --cache-dir D:/AI_Agent_Platform/.pytest_cache/m8b2-uv tool run ruff format --check backend/agent_runtime/metrics.py tests/agent_runtime/test_metrics.py
.\.venv\Scripts\python.exe -m compileall -q backend/agent_runtime tests/agent_runtime
.\.venv\Scripts\python.exe -m tabnanny backend/agent_runtime/metrics.py backend/agent_runtime/observability.py backend/agent_runtime/runtime.py tests/agent_runtime/test_metrics.py
git diff --check
```

Metrics 专项首轮：**31 passed in 0.67s**。完整组合首轮：**978 passed in 16.15s**。
最终格式化后组合回归：**978 passed in 16.39s，0 failures/errors/skipped**。
从最终 JUnit 的 testcase 记录独立汇总得到以下数量：

| 测试域 | 通过数 |
| --- | ---: |
| Metrics 专项 | 31 |
| 既有 Agent Runtime（含 B1 44 项） | 416 |
| MCP Host（含真实 stdio） | 150 |
| Evaluation V1 | 209 |
| Tools | 172 |
| 合计 | 978 |

| 其他检查 | 结果 |
| --- | --- |
| pip check | No broken requirements found |
| Ruff 静态检查（E9/F） | All checks passed |
| Ruff 新增文件格式检查 | 2 files already formatted |
| compileall / tabnanny | 通过，无错误输出 |
| git diff --check | 通过 |
| 冻结模块相对 ed614ec 的 diff | 无变化 |
| 最终暂存前 fetch / HEAD...origin/main | 同步，0/0 |

Ruff 使用隔离工具环境，版本 0.17.4，不改变项目依赖。格式化仅针对两个新增
Python 文件，静态检查覆盖全部 B2 修改的 Python 文件；B1 的既有格式不整体重写。
测试基于当前本地固定依赖和已恢复的 MCP Server 虚拟环境；沙箱临时目录/子进程
权限限制通过获准的执行权限处理，没有跳过用例或改变测试业务逻辑。

## 冻结保护与遗留事项

相对 B1 基线核对：Evaluation 生产代码/测试、MCP Host、ToolDispatcher、ToolRegistry、
LLMPort、DeepSeekAdapter、ToolContext、AgentRunResult 和 Bootstrap 未修改。
Runtime 只新增请求事件和默认 Observer 引用，既有预算、分类器、最多一次额外重试、
工具执行顺序、错误传播和取消语义保持；B1 事件语义不变。

没有接入 Prometheus、Grafana、OpenTelemetry、数据库、远程服务、多进程汇总或
Dashboard。未调用真实 LLM、Judge 或公网工具；真实 Judge 校准仍是既有后续项，
不将离线工程验收宣称为真实模型质量验收。未重跑无关的 Memory/RAG 全仓测试。

非阻断后续事项：B1 元数据字符串长度约束、llm.attempt_failed 累计耗时字段命名；
以及 V1 明确排除的跨线程、跨进程、故障事件补偿和精确分位数计算。
Observer 故障期间允许指标不完整；缺失终态可能使 Active Gauge 留存到进程重启。
指标不用于业务决策或计费，不以修补观测数据改变 Runtime 行为。

Module 8 在约定 Offline Evaluation + Runtime Observability + Process-local Metrics V1
范围内达到 COMPLETE / FROZEN；后续 Stage 3 系统性复盘等待用户单独发起。
