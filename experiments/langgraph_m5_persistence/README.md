# Stage 4 / M5 — Checkpoint、持久化与恢复实验

Part A 使用真实 LangGraph Runtime 和 `InMemorySaver`，通过确定性的论文分析流程验证跨 Run 状态、Thread 隔离、快照和历史查询，以及顺序和并行 Super-step 的检查点边界。Part B 接入 `PostgresSaver`，验证跨进程持久化、Pending Writes、Resume 和同步 Durability，并通过订单实验验证事务提交后故障恢复的业务幂等性。

## Graph 结构

```text
START → InitializeRun → ParsePaper
                         ├── AnalyzeContent ──┐
                         └── CheckReferences ─┤
                                              ↓
                                       GenerateReport → END
```

`ParsePaper` 归一化空白。两个分支分别生成模拟摘要和提取 `[ref:标识]`，写入不同的 State 字段。列表边 `add_edge(["AnalyzeContent", "CheckReferences"], "GenerateReport")` 让 Runtime 等齐两个分支，然后在下一步生成一份报告。各节点只返回自己的局部更新，业务逻辑均为本地字符串处理。

## State Contract

`PaperInput` 只接受 `run_id` 和 `raw_paper_content`。`PaperState` 由输入和可选的 `PaperUpdate` 字段组成。派生字段在相应节点执行时产生，因此较早的快照可以缺少这些字段。

| 生命周期 | 字段 | 写入时机 |
| --- | --- | --- |
| Run | `run_id: str`、`raw_paper_content: str` | 由每次 `invoke` 的输入更新。调用方为每个新 Run 提供唯一 `run_id`。 |
| Run | `paper_content: str` | `InitializeRun` 清空，`ParsePaper` 写入规范化后的论文。保留 `raw_paper_content` 可以在快照中同时观察输入和解析边界。 |
| Run | `analysis: str`、`references: list[str]`、`report: str` | `InitializeRun` 分别重置为 `""`、`[]`、`""`，随后由对应节点写入。 |
| Thread | `completed_runs: int`、`report_history: list[str]` | `GenerateReport` 以 `0` 和 `[]` 为首轮默认值，完成时返回计数加一及追加一份报告的新列表。 |

`thread_id` 通过 `configurable` 传入。新一轮输入写入后，`InitializeRun` 才清空上一轮派生字段，并保留 Thread 字段。报告节点单独拥有历史字段的写入权，通过显式更新追加一次，旧列表原样保留在历史快照中。

## Checkpointer 接入和查询

`build_paper_graph(checkpointer=None)` 将公开 `Checkpointer` 参数传给 `StateGraph.compile`。省略参数得到独立运行的顶层图。传入 `InMemorySaver()` 后，Runtime 按 Thread 保存状态。将来可在同一参数位注入其他 saver。

```python
from langgraph.checkpoint.memory import InMemorySaver

from paper_graph import build_paper_graph

graph = build_paper_graph(InMemorySaver())
config = {"configurable": {"thread_id": "A"}}
graph.invoke({"run_id": "A-1", "raw_paper_content": "石墨烯 导电 [ref:a1]"}, config)

latest = graph.get_state(config)
history = list(graph.get_state_history(config))
selected = graph.get_state(history[1].config)
```

- `get_state(config)` 返回最新 `StateSnapshot`。`values` 是该边界的 State，`next` 是下一步要执行的节点，正常结束时为 `()`。`config` 包含 `thread_id`、根命名空间 `checkpoint_ns=""` 和实际的 `checkpoint_id`。`metadata` 提供 `source` 和 `step` 等信息。
- `get_state_history(config)` 返回最新在前的迭代器。`parent_config` 指向前一快照，可以检查相邻快照的关系。把历史快照的 `config` 传回 `get_state` 即可按其中的真实 ID 查询。
- `tasks` 提供该边界调度的任务。查询已完成的历史时，任务条目可能带有各自的 `result`，而 `values` 仍反映这些任务执行前的完整状态。下一快照的 `values` 才反映同一 Super-step 的合并结果。[LangGraph Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)

## 运行方式

所有运行入口使用 M5 实验目录自己的 Python 3.12 `.venv`。从仓库根目录进入实验目录，首次使用时创建环境并安装完整锁定依赖。已有 M5 环境可直接激活。

```powershell
Set-Location experiments/langgraph_m5_persistence
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = '1'
python -m pip install -r requirements.lock.txt
```

### 仅复跑 Part A

```powershell
python demo.py
python -m pytest -v tests/test_persistence.py
```

### 非数据库测试

在 M5 环境中运行 Part A 及 Part B 的内存恢复和连接目标校验。

```powershell
python -m pytest -v -m 'not postgres'
```

需要验证 PostgreSQL 时，按下文的[完整验收命令](#实验库和完整验收)配置实验连接。Demo 和测试禁用 LangSmith tracing，Part A 测试还会拦截 socket 连接。

## Part A 实测结果

Part A 首次验收复用了 M3 的独立 `.venv`，当时使用 Python 3.12.9、LangGraph 1.2.22、LangGraph Checkpoint 4.3.2、LangChain Core 1.8.1 和 pytest 9.2.0。下方证据保留了当时的运行结果。

Demo 中，无 Checkpointer 的两个调用各得到 `completed_runs=1` 和一份报告。启用 `InMemorySaver` 后，按 A-1、B-1、A-2 调用。A 的最终 `completed_runs=2`，A-2 的引用为空，历史包含 A-1 和 A-2。B 的最终 `completed_runs=1`，只保留 B-1 的报告。两个 Thread 的最终 `next` 均为 `()`。[Demo 原始输出](evidence/demo_part_a.txt)

| Demo 中的边界 | `next` | `values` 中观察到的状态 |
| --- | --- | --- |
| A-1 step 1 | `ParsePaper` | 已初始化的本轮字段。 |
| A-1 step 2 | `AnalyzeContent`、`CheckReferences` | 有解析后的论文，摘要和引用仍为空。 |
| A-1 step 3 | `GenerateReport` | 摘要和引用同时可见，报告仍为空。 |
| A-2 step 6 | `InitializeRun` | 已换入 A-2 的 ID 和原文，派生字段仍为 A-1 的值。 |
| A-2 step 7 | `ParsePaper` | 派生字段已清空，上一份报告仍在 `report_history`。 |
| A-2 step 10 | `()` | A-2 已完成，累计两份报告。 |

A1～A8 共 **8 项通过（0.72 秒）**，覆盖无 saver、同 Thread 延续、不同 Thread 隔离、中间快照验证初始化、最终 StateSnapshot、历史顺序及 ID 查询、并行合并边界、saver 实例隔离。A7 使用 `tasks`、`metadata.step`、`parent_config` 和合并前后的 `values` 验证，并逐个检查快照中两份非空分支结果始终同时出现。[pytest 原始输出](evidence/pytest_part_a.txt)

`pip check` 返回 `No broken requirements found.`。[依赖检查原始输出](evidence/pip-check_part_a.txt)

## Part A 能力边界

本实验验证同一 Thread 的顺序调用、不同 Thread 的隔离，以及同一 Run 内的并行 Super-step。`InMemorySaver` 的数据存在实例内存中。A8 在同一进程创建新实例，观测空历史及首轮计数从 1 开始。历史查询只读取快照。`run_id` 用于标记 Run，唯一性由调用方负责。

## Part B — PostgreSQL 持久化、恢复与业务幂等

### 实验库和完整验收

Part B 使用 `agent_platform_m5_lab` 独立库。本机使用 PostgreSQL 18.4 的现有服务和角色建库。初始化连接维护库 `postgres`，用该角色已有的 `CREATE DATABASE` 权限执行 `CREATE DATABASE agent_platform_m5_lab TEMPLATE template0`。实验连接由 `M5_POSTGRES_URI` 提供，接受 PostgreSQL URI 或 libpq 连接字符串。程序校验配置中的数据库名和回环地址，再用 `current_database()` 核对实际数据库。

激活前文的 M5 `.venv` 后设置实验连接，执行全量测试及跨进程专项。锁文件保留了 Part A 的核心版本，新增 `langgraph-checkpoint-postgres` 3.2.1、Psycopg 与二进制驱动 3.3.9 及其依赖。

```powershell
$env:M5_POSTGRES_URI = Read-Host -MaskInput '输入本机 M5 实验库连接字符串'
python demo_part_b.py
python -m pytest -v
python -m pytest -v -s -m cross_process
python -m pip check
Remove-Item Env:M5_POSTGRES_URI
```

Demo 和数据库测试在入口执行 `PostgresSaver.setup()` 和订单表初始化。已有表可复用。`postgres` marker 标记数据库测试，缺少 `M5_POSTGRES_URI` 时这些测试会明确跳过。完整验收要求使用已配置的实验库，19 项测试全部执行，且无跳过。数据库连接限定在本机。

### B1 — PostgresSaver 生命周期

`open_postgres_saver()` 用 Psycopg 创建同步连接，以 `autocommit=True`、`prepare_threshold=0` 和 `dict_row` 构造官方 `PostgresSaver`。将 saver 注入原来的 `build_paper_graph`，离开上下文时关闭连接。恢复进程直接重连并读取已有检查点。订单写入用另一条连接和独立事务。[PostgresSaver API](https://reference.langchain.com/python/langgraph.checkpoint.postgres/PostgresSaver)

新 Saver 读回 A 的原检查点 ID 和完整历史。B1-2 用两个独立 Python 进程验证同样的行为，P1 正常退出后，P2 读回同一 ID 和第一份报告，再提交 `paper-2` 新输入。A 的 `completed_runs` 从 1 到 2，保留两份报告和原有历史 ID。交替调用的 Thread B 保留自己的检查点和一份报告。

### B2 — Pending Writes 与 Resume

同一份论文图通过 `node_overrides` 注入记录和故障。在 `CheckReferences` 等待 `Event` 时，调用方消费 `graph.stream` 的更新。同步 Runtime 启用 `updates` 和 `custom` 两种模式，在观察到 `AnalyzeContent` 已返回的更新之后释放引用分支。`CheckReferences` 随即抛出 `ReferenceFailure`，异常传播到流的调用方。超时只用于防止死锁。[LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)

失败边界在 Demo 中的 step 2，完整 `values` 的 `analysis` 和 `references` 仍为空。同一快照的分析任务带有 `task.result`，`saver.get_tuple(checkpoint_config).pending_writes` 中可找到该任务的 `analysis` 写入。最新状态的 `next` 只有 `CheckReferences`。解除故障后调用 `graph.invoke(None, config=config, durability="sync")`，Runtime 复用分析写入，补跑失败分支，在 step 4 完成报告，`completed_runs=1` 且只有一份报告。传入新输入会进入 `InitializeRun` 开启新一轮。[LangGraph Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)

| 节点 | InMemory 总次数 | PostgreSQL 故障进程 P1 | PostgreSQL 恢复进程 P2 |
| --- | ---: | ---: | ---: |
| `InitializeRun` | 1 | 1 | 0 |
| `ParsePaper` | 1 | 1 | 0 |
| `AnalyzeContent` | 1 | 1 | 0 |
| `CheckReferences` | 2 | 1 | 1 |
| `GenerateReport` | 1 | 0 | 1 |

进程各自将节点调用和 PID 写入同一个隔离的 JSONL 审计文件。B2-2 的 P1 捕获已传播的预期异常，读取 Pending Writes 后正常退出，P2 使用新 Saver 恢复相同 Run。

### B3 — 订单提交与幂等性

`START → GenerateOrder → SaveOrder → END`。State 保存 `run_id`、`item`、`quantity`、`idempotency_key`、`order_id` 和 `save_status`。`GenerateOrder` 从 `run_id` 生成稳定键，并持久化 `pending` 状态。同步持久化确保进入 `SaveOrder` 前的检查点已经保存。`SaveOrder` 读取该键，在独立事务中提交订单，再返回 `order_id` 和 `saved` 状态。

三个场景都在订单事务提交后、`SaveOrder` 返回前触发故障。恢复前查询已看到一条订单，检查点仍为 `pending` 和 `order_id=None`，下一任务为 `SaveOrder`，且没有它的成功结果写入。恢复会再次执行 `SaveOrder`，总共两次。

| 场景 | 恢复前订单数 | 恢复后订单数 | 结果 |
| --- | ---: | ---: | --- |
| B3-1 无幂等保护 | 1 | 2 | 预期的重复写入，第二次返回新订单 ID。 |
| B3-2 有幂等保护 | 1 | 1 | 恢复返回原订单 ID，Graph 正常结束。 |
| B3-3 有幂等保护，跨进程硬退出 | 1 | 1 | P1 硬退出后，P2 从持久化检查点恢复并返回原订单 ID。 |

对照表 `m5_orders_unprotected` 和幂等表 `m5_orders_idempotent` 分离。幂等表对 `idempotency_key` 设唯一约束。`INSERT … ON CONFLICT … DO UPDATE … WHERE … RETURNING` 原子地处理冲突并返回已有订单 ID。只有 `item` 和 `quantity` 等于首次提交值时才复用，参数冲突会显式报错。业务幂等由订单表约束和应用写入逻辑共同提供。[PostgreSQL INSERT](https://www.postgresql.org/docs/18/sql-insert.html)

B3-3 在 P1 的提交回调记录订单 ID 后调用 `os._exit(73)`，直接终止该实验子进程，跳过正常退出清理。父进程确认退出码，使用独立连接读取已提交订单，再启动 PID 不同的 P2。子进程超时为 45 秒，超时时由 `subprocess.run` 终止并回收它启动的子进程。已验证的故障窗口是 Python 子进程退出，数据库服务继续运行。

### 实测与数据清理

完整验收 **19 项通过（16.97 秒）**，覆盖 A1～A8 回归和 11 项 Part B 测试。三个跨进程场景另行执行，均通过，证据保留 PID、检查点、Pending Writes、订单 ID 和审计记录。`pip check` 返回 `No broken requirements found.`。

- [Demo 原始输出](evidence/demo_part_b.txt)
- [全量 pytest 原始输出](evidence/pytest_part_b.txt)
- [跨进程 pytest 原始输出](evidence/cross_process_part_b.txt)
- [依赖检查原始输出](evidence/pip-check_part_b.txt)

每个测试和 Demo 场景生成独立 Thread ID。场景结束时，`PostgresSaver.delete_thread` 清理本场景的检查点，订单表按本场景的幂等键删除记录。临时审计文件由 Demo 临时目录或 pytest 的 `tmp_path` 管理。若手动中止整个运行导致残留，可根据已知的实验 `thread_id` 和 `idempotency_key` 使用相同清理入口。验收后查询确认三张 Checkpoint 数据表和两张订单表均为零行。实验库和表结构保留供复跑，连接配置和虚拟环境由 `.gitignore` 排除。
