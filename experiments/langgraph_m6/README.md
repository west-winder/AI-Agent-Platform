# Stage 4 / M6-4 · 科研邮件人工审批实验

**M6 COMPLETE / FROZEN**，**M7 NEXT**。2026-11-27 完成独立 Code Review 后的正式验收。

借助一个使用固定模板的邮件流程，观察 LangGraph 如何暂停、接收人工决策并恢复。Part A 用 `InMemorySaver` 验证审批与路由。Part B 用 `PostgresSaver` 验证检查点跨进程恢复，再用独立的 SQLite 发送服务观察同一次业务操作如何跨故障重试。

## 执行流程

```text
START → DraftEmail → HumanApproval（interrupt）
                       ├─ approve   → SendEmail → END
                       ├─ reject    → Reject    → END
                       ├─ revise    → ReviseEmail → HumanApproval
                       └─ 无效反馈  → HumanApproval
```

| 节点 | 职责 |
| --- | --- |
| `DraftEmail` | 根据 `query` 套用固定邮件模板，初始化版本 1 和本任务的发送幂等键。 |
| `HumanApproval` | 用 `interrupt()` 提供当前收件人、草稿、版本与请求 ID。恢复后校验反馈，更新决策，由条件边选择下一节点。 |
| `ReviseEmail` | 把修改意见追加到草稿，版本加一，清空审批决策和批准版本，再次请求审核。 |
| `SendEmail` | 校验业务状态、审批决策和批准版本，调用模拟发送服务并保存结果。 |
| `Reject` | 将人工反馈写为拒绝原因，以 `rejected` 结束。 |

## State 与人工反馈

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `query`、`recipient` | `str` | 任务需求与模拟收件人，调用方首次启动时提供。 |
| `email_draft`、`draft_version` | `str`、`int` | 当前草稿和它的版本，每次修订推进版本。 |
| `human_feedback`、`decision` | `str`、`approve \| reject \| revise \| None` | 本轮接受的文字意见与审批决策。 |
| `approved_version` | `int \| None` | 仅 `approve` 记录当前版本，修订返回等待审核时清空。 |
| `feedback_error`、`rejection_reason` | `str \| None` | 无效反馈的校验提示与正式拒绝原因。 |
| `idempotency_key` | `str` | `DraftEmail` 生成的每任务 UUID 键。修订和发送重试沿用已持久化的原值。 |
| `status` | `str` | 业务状态。等待审核为 `awaiting_approval`，路由过程中可观察到 `approved`、`revising` 或 `rejecting`，业务终态为 `sent` 或 `rejected`。 |
| `send_result` | `dict \| None` | 模拟发送成功后保存 `message_id`。 |

`thread_id` 放在 `configurable`。运行时暂停可从 `GraphOutput.interrupts` 或 `StateSnapshot.interrupts` 读取，`StateSnapshot.next` 显示接下来可调度的节点。在本图中，`next == ()` 表示执行结束，业务结果由 `status` 给出。

提交 `Command(resume=feedback)` 时复用原 `thread_id`，并从当前请求复制 `request_id` 和 `draft_version`。`action` 可为 `approve`、`reject` 或 `revise`，后两种要求非空 `feedback`。请求 ID 包含任务标识与草稿版本。只更新旧反馈中的版本号仍会被请求 ID 校验拦截。无效反馈会产生新 Interrupt，并在请求中带回 `feedback_error`，调用方可以继续提交合法反馈。

## 暂停、恢复与副作用

首次到达 `interrupt()` 会结束本次调用并返回审批请求。收到 `Command(resume=...)` 时，LangGraph 从 `HumanApproval` 的入口重跑，生成请求的代码会再执行一次。此时 `interrupt()` 返回传入的反馈，已完成的 `DraftEmail` 保持在检查点历史中。测试使用实际 `tasks` 事件和请求构造次数验证这一行为。[LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)

| 存储 | 保存内容 |
| --- | --- |
| LangGraph Checkpoint | 草稿、审批决策、批准版本、幂等键、Interrupt 和后续可调度节点。 |
| SQLite `deliveries` | 每个幂等键对应一条模拟发送记录，包含首次提交的收件人、草稿和消息 ID。 |
| SQLite `send_requests` | 已处理发送请求的 PID、幂等键和结果。一次重放或冲突各添一条请求记录。 |

SQLite 使用 `BEGIN IMMEDIATE` 和唯一键约束，将查找、去重、新增发送记录和请求计数放在同一个事务里。相同键和内容返回首次提交的消息 ID。相同键配合不同的收件人或草稿会触发 `IdempotencyConflict`。

Part B 调用统一使用 `durability="sync"`，LangGraph 在启动下一步前持久化上一步更新。B3 在 SQLite 提交之后、`SendEmail` 返回更新之前用 `os._exit(73)` 终止实验子进程。新进程在同一 `thread_id` 读取已批准的检查点，调用 `graph.invoke(None, config, durability="sync")` 重跑待执行的 `SendEmail`，随后将原消息 ID 写入 Graph State。这个恢复入口以已持久化的审批为依据，无需提交新的 `Command`。[LangGraph Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)

## 运行与复验

从仓库根目录运行，复用 M5 的 Python 3.12 环境和依赖锁。首次使用时按 M5 的环境步骤安装。本实验使用 Python 自带的 `sqlite3`。[M5 环境设置](../langgraph_m5_persistence/README.md#运行方式)

```powershell
$m6Python = './experiments/langgraph_m5_persistence/.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'

# Part A 演示与验收
& $m6Python -m experiments.langgraph_m6.demo
& $m6Python -m pytest -q experiments/langgraph_m6/tests/test_part_a.py

# SQLite 幂等测试也可离线执行
& $m6Python -m pytest -q experiments/langgraph_m6/tests -m 'not postgres'
```

Part B 按 M5 规范准备本机 `agent_platform_m5_lab` 实验库。连接配置沿用 `M5_POSTGRES_URI`，M5 的连接函数会校验回环地址和数据库名。场景使用独立 UUID Thread，在退出时清理这些 Thread 的检查点。Demo 每次使用独立临时 SQLite 文件，pytest 使用 `tmp_path`。

```powershell
$env:M5_POSTGRES_URI = Read-Host -MaskInput '输入本机 M5 实验库连接字符串'
try {
    & $m6Python -m experiments.langgraph_m6.demo_part_b
    & $m6Python -m pytest -v -s experiments/langgraph_m6/tests -m postgres
    & $m6Python -m pytest -q experiments/langgraph_m6/tests
    & $m6Python -m pip check
} finally {
    Remove-Item Env:M5_POSTGRES_URI
}
```

`demo_part_b` 会串行启动并等待独立的 Python 子进程退出。B1 对照首个进程保存的等待审核快照与第二个进程读回的完整快照。B2 观察 A 获批期间 B 的快照保持一致。B3 对照硬退出后与恢复后的 SQLite 行数、消息 ID 和请求 PID，并显示失败边界 `SendEmail` 的任务结果和 Pending Writes。缺少实验库连接配置时，pytest 会跳过数据库用例。完整验收要求 3 项数据库测试实际执行。

## 代码阅读指南

1. 从 [Part A Demo](demo.py) 的首次 `invoke` 开始。沿 [Workflow](workflow.py) 的 `draft_email`、`human_approval` 读到 `interrupt()`，对照输出中的第一版草稿和审批请求。
2. 跟进 Demo 中的 `revise` → `approve`。在 `validate_feedback` 查看版本和请求 ID 检查，再依次看 `route_approval`、`revise_email` 和 `send_email`，核对第二版怎样得到自己的审批。无效反馈如何重新暂停可对照 [A6 测试](tests/test_part_a.py)。
3. 从 [Part B Demo](demo_part_b.py) 的 `recover_send_in_processes` 跟踪进程退出和恢复。沿 `worker` 看首次批准触发的 `crash_after_send`，再看 `recover` 分支的 `invoke(None, ...)`。
4. 沿 `SendEmail` 的调用进入 [SQLite 服务](mail_store.py) 的 `send`，找到事务何时提交。对读 [B3 测试](tests/test_part_b.py) 中的待运行节点、请求数、发送记录数和消息 ID，能同时定位流程恢复边界和业务去重边界。

## 实测结果与限制

2026-11-27 在 LangGraph 1.2.22、PostgreSQL 18.4 和 SQLite 3.45.3 上完成最终回归。Part A 15 项通过，另有 2 项离线 SQLite 测试及 3 项 PostgreSQL 集成测试通过。完整 M6 共 20 项通过，0 失败、0 跳过。详见 [正式验收报告](ACCEPTANCE.md)和 [最终 pytest 证据](evidence/pytest_final.txt)。

实验调用方按同一 Thread 串行提交反馈，审批 ID 用于关联草稿，审批身份由调用方负责。B3 覆盖 Python 进程在模拟发送提交后的退出，恢复依赖同一份 SQLite 文件和可用的 PostgreSQL 检查点。单条送达记录的结论仅适用于本模拟服务的幂等语义。真实 SMTP 或任意邮件服务的 Exactly-once 需要独立验证。
