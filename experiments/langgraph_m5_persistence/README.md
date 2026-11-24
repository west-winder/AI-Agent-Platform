# Stage 4 / M5-4 Part A — Checkpoint 基础综合实验

用真实 LangGraph Runtime 和 `InMemorySaver` 跑一个确定性的论文分析流程，观察跨 Run 状态、Thread 隔离、历史查询，以及顺序和并行 Super-step 的检查点边界。

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

从仓库根目录进入实验目录。交付实测复用了 M3 的独立 `.venv`。Python 3.12.9，LangGraph 1.2.22，LangGraph Checkpoint 4.3.2，LangChain Core 1.8.1，pytest 9.2.0。

```powershell
Set-Location experiments/langgraph_m5_persistence
$paperPython = '../langgraph_m3/.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
& $paperPython demo.py
& $paperPython -m pytest -v
& $paperPython -m pip check
```

也可在本目录用 Python 3.12 创建独立环境，然后执行上面的三条运行命令。

```powershell
py -3.12 -m venv .venv
$paperPython = './.venv/Scripts/python.exe'
& $paperPython -m pip install -r requirements.lock.txt
```

锁文件来自实测环境的 `pip freeze`。安装需要可用的软件包来源。Demo 和测试禁用 LangSmith tracing，测试额外拦截 socket 连接，运行无需外部服务或 API Key。

## 实测结果

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

## 能力边界与 Part B

本实验验证同一 Thread 的顺序调用、不同 Thread 的隔离，以及同一 Run 内的并行 Super-step。`InMemorySaver` 的数据存在实例内存中。A8 在同一进程创建新实例，观测空历史及首轮计数从 1 开始。历史查询只读取快照。`run_id` 用于标记 Run，唯一性由调用方负责。

Part B 待 Review 后开展。后续实验项包括 PostgreSQL Checkpointer 接入、Pending Writes 故障恢复，以及幂等订单场景验证。具体范围以 Part B 的任务为准。
