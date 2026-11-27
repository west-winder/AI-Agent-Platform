# M6-4 实验验收报告

2026-11-27，M6-4 通过独立 Code Review。正式回归通过 A1–A6、B1–B3，Stage 4 进度为 **M6 COMPLETE / FROZEN**、**M7 NEXT**。

## 正式验收

| 范围 | 最终实测 |
| --- | --- |
| M6 全量 | 20 通过，0 失败、0 跳过，耗时 8.53 秒。包括 Part A 15 项、SQLite 2 项和 PostgreSQL 集成 3 项。 |
| M5 共享持久化回归 | 19 通过，0 失败、0 跳过，耗时 11.44 秒。 |
| 依赖与实验数据 | `pip check` 通过。隔离实验库的三张 Checkpoint 数据表和两张 M5 订单表均为 0 行。 |

- B1 中 PID `21596` 暂停后正常退出。PID `22908` 在 Thread `m6-c59b00dc9ecb485aac3297d69b31952c-review` 读回检查点 `1f1ea45f-0d9c-6d9c-8001-51c6a5425f6b`，批准后正常完成一次模拟发送。
- B2 在新 Saver 读回两个 Thread 的独立结果。A 获批发送时 B 仍等待审核，随后 B 被拒绝，B 的发送请求数为 0。
- B3 中 PID `13204` 在发送事务提交后以退出码 `73` 结束。PID `28304` 从 Thread `m6-30cfd45aab2d4190afb268a762fddab6-crash` 的已批准检查点 `1f1ea45f-38af-6348-8002-340de94e3242` 恢复，重跑 `SendEmail`。前后使用键 `m6-email:d7684a00f2df45d5aba0d0162cce5afd`，两次请求只产生一条发送记录，最终消息 ID 为 `mock-mail:c143d39d1b5248dba0d861b4a160d895`。

[最终 pytest 原始证据](evidence/pytest_final.txt)

## 设计与交付范围

科研邮件草稿根据固定模板生成。`HumanApproval` 通过动态 Interrupt 公开草稿与当前版本，校验请求 ID 和版本后用条件边进入发送、拒绝或修订。无效反馈重开当前草稿的审核，修订开启新版本的审核。`SendEmail` 在调用服务之前复验批准状态。Part B 用 M5 的官方 `PostgresSaver` 连接函数保存流程状态，用独立 SQLite 事务维护发送记录和请求证据。

| 新增文件 | 用途 |
| --- | --- |
| `state.py`、`workflow.py` | 输入、状态、反馈校验、节点和路由。 |
| `mail_store.py` | 原子去重、冲突校验及持久化发送审计。 |
| `demo.py`、`demo_part_b.py` | 离线审批演示、PostgreSQL 跨进程验收和故障注入。 |
| `pytest.ini`、`tests/conftest.py`、`tests/test_part_a.py`、`tests/test_part_b.py`、`tests/test_mail_store.py` | 测试环境、离线验收、数据库验收与 SQLite 幂等测试。 |
| `.gitignore` | 排除本地实验临时目录。 |
| `README.md`、`ACCEPTANCE.md`、`evidence/pytest_m6.txt`、`evidence/pytest_final.txt` | 操作手册、代码阅读指南、验收及正式回归证据。 |

## 逐项验收

| 项目 | 结果 | 观察与断言 |
| --- | --- | --- |
| A1 主动暂停 | 通过 | 调用正常返回一个 Interrupt，带有收件人、第一版邮件和版本。可读回相同状态，下一节点为 `HumanApproval`，发送调用数为 0。 |
| A2 审批通过 | 通过 | 相同 Thread 用 `Command(resume=...)` 恢复。当前版本获批，模拟发送一次，`next=[]` 且业务状态为 `sent`。发送节点单独拒绝批准版本不一致的状态。 |
| A3 审批拒绝 | 通过 | `Reject` 保存拒绝原因，`next=[]` 且业务状态为 `rejected`，发送调用数为 0。 |
| A4 修订后重审 | 通过 | 修改意见以固定格式追加，版本从 1 变为 2，清空决策和批准版本。第二版产生新 Interrupt 和请求 ID，获批后才发送第二版。 |
| A5 节点重跑 | 通过 | 恢复前后请求构造代码执行两次。恢复时的 `tasks` 事件依次启动 `HumanApproval` 和 `SendEmail`，最终发送一次。 |
| A6 无效与过期反馈 | 通过 | 覆盖未知 action、缺少版本或请求 ID、错误版本、布尔版本、错误请求 ID、无效意见与旧版本反馈。相应校验错误回传，新 Interrupt 可以接收合法反馈。 |
| B1 跨进程审批恢复 | 通过 | 首进程正常退出。第二个 PID 读回同一 Thread 和完整审批快照，批准后保存发送结果。 |
| B2 Thread 隔离 | 通过 | A、B 以独立草稿、收件人、请求 ID 和幂等键等待审批。A 发送期间 B 的快照保持一致。重连后读回 A `sent`、B `rejected`，B 的请求与发送记录均为 0。 |
| B3 提交后故障恢复 | 通过 | 故障 PID 提交发送事务后硬退出。恢复 PID 从已批准的同步检查点重跑 `SendEmail`。2 次请求，共 1 条实际模拟发送记录，复用相同幂等键和消息 ID。另验证冲突拒绝和 4 个并发发送请求只产生 1 条记录。 |

## 独立 Review 前的跨进程与故障证据

以下 B1、B3 数据取自同一次完整 pytest 运行。原始日志包含 `thread_id`、审批请求、Checkpoint、SQLite 请求 PID 和持久化邮件记录。[完整实测日志](evidence/pytest_m6.txt)

### B1 等待审批的恢复

Thread 为 `m6-a064778b76094206b81dbeff39d71a32-review`。审批前 SQLite 请求数和记录数均为 0。

| 进程 | 退出码 | 检查点 | 执行边界 |
| --- | ---: | --- | --- |
| `30496` | 0 | `1f1ea41e-461e-6948-8001-5196bce2a51b` | Step 1 等待 `HumanApproval`。 |
| `29356` | 0 | `1f1ea41e-461e-6948-8001-5196bce2a51b` → `1f1ea41e-5583-6313-8003-594f694c50e6` | Step 1 读回原快照，批准后在 Step 3 结束。 |

最终状态为 `sent`，消息 ID 为 `mock-mail:a0322e4f40fe47ccafb19a0af0f06044`。

### B2 两个 Thread

数据库 Demo 中，`m6-1475703d1a7f4c96a4fc0077ff193fd2-b` 在 A 获批前后都保持 `awaiting_approval`，检查点始终为 `1f1ea41e-ad9a-6fbf-8001-2a5c26e22390`。重新打开 Saver 后，A 的终态检查点为 `1f1ea41e-adce-6548-8003-f7864c852db0`（`sent`），B 的终态检查点为 `1f1ea41e-ade5-660d-8003-c7c1a1d5cdc2`（`rejected`）。

### B3 已提交发送事务的恢复

Thread 为 `m6-3078989aca1d4788ba19e67679384a09-crash`。PID `6600` 创建任务并暂停后正常退出。

| 阶段 | PID / 退出码 | 检查点或 SQLite 观察 |
| --- | --- | --- |
| 人工批准后注入故障 | `29124` / `73` | Step 2 为 `1f1ea41e-7355-617f-8002-e25ce42fd4d0`，`status=approved`、`approved_version=1`，下一节点为 `SendEmail`。SQLite 已有 1 次 `sent` 请求和 1 条发送记录。 |
| 新进程继续执行 | `812` / `0` | 从相同 Step 2 调用 `invoke(None, ...)`，到 Step 3 `1f1ea41e-808c-67f0-8003-0365e3659981` 正常结束。SQLite 新增 1 次 `replayed` 请求，发送记录仍为 1 条。 |

故障边界的 `send_result` 与 `SendEmail` 任务结果皆为 `None`，该任务的 Pending Writes 为空，审批 Interrupt 列表也为空。两个发送请求都使用 `m6-email:735c3661051640a8aa7d449b02185414`，最终 `send_result` 与故障进程提交时均为 `mock-mail:fa6f16d0f6a2429c8b6def02faa80745`。

## 独立 Review 前的测试统计

Python 3.12.9、LangGraph 1.2.22、LangGraph Checkpoint 4.3.2、PostgresSaver 3.2.1、Psycopg 3.3.9、pytest 9.2.0。数据库验收使用在工作区临时初始化的 PostgreSQL 18.4 实例和 M5 约定的实验库名。SQLite 版本为 3.45.3。

| 范围 | 实测 |
| --- | --- |
| Part A | 15 通过。 |
| SQLite 服务 | 2 通过。 |
| PostgreSQL 集成 | 3 通过，包含 2 个真正跨进程的场景。 |
| M6 全量 | 20 通过，0 失败、0 跳过。9.78 秒。离线和 PostgreSQL Demo 均运行成功。 |
| M5 全量回归 | 19 通过，0 跳过。13.21 秒。核对实验库三张 Checkpoint 数据表及两张 M5 订单表均为 0 行。 |
| M4-8 离线回归 | 9 通过。6.05 秒。1 条既有的 `MCPAdapter` Beta 警告。 |
| 自研 AgentRuntime 契约与执行回归 | 114 通过。2.17 秒。 |
| 依赖检查 | M5 和 M4-6 环境的 `pip check` 均通过。 |
| 补丁空白检查 | `git diff --check` 和 14 个新增文件的逐文件差异检查均通过。 |

## 已知限制与 Git 状态

本实验验证进程退出时的发送幂等，恢复依赖同一份 SQLite 文件及可读的 PostgreSQL 检查点。覆盖范围是单 Thread 顺序提交审批和 SQLite 的并发发送去重。审批请求 ID 用于相关性检查，真实审批授权及真实 SMTP 的 Exactly-once 需要独立验证。

正式收口从与 `origin/main` 同步的 `main` 开始。当时暂存区为空，工作区包含此前送审的 14 个 M6 文件。正式交付包含 `experiments/langgraph_m6/` 和根目录 README 中的 Stage 4 进度更新。临时 PostgreSQL 实例已停止并清理。
