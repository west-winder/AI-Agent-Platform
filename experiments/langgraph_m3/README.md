# Stage 4 / M3 最小 Runtime 实验

2026-11-16（Asia/Shanghai）。纯本地模拟论文；只验证 M3，不接入正式模块，不进入 M4。

仓库检查：根 README / docs/stage3_status.md 标记 Stage 0～3 COMPLETE / FROZEN；
Stage 4 只有路线图。项目用 Python 3.12、根 requirements.txt 固定依赖，无现成 experiments
目录或 LangGraph 实现，未发现适用的 AGENTS.md。本目录为新建隔离实验，正式依赖、
README 和业务代码均未修改。首次实验交付时未执行 git add / commit / push，文件保持
untracked；后续按用户授权进行独立的 Git 收口。

## 文件

- `paper_review.py`：主图、引用子图、Wrapper、固定输入。
- `superstep_demo.py`：A → B/C、B → D 的同步实验。
- `demo.py`：运行完整示例并打印结果。
- `tests/conftest.py`、`tests/test_runtime.py`：禁止测试联网、12 项真实 Runtime 测试。
- `requirements.txt`：实验直接依赖；`requirements.lock.txt`：本机完整解析版本。
- `pytest.ini`、`.gitignore`：独立测试配置和环境/缓存忽略规则。
- `evidence/demo.txt`、`evidence/pytest.txt`、`evidence/pip-check.txt`：实际执行原始输出。
- `README.md`：本说明。`.venv/` 和 `.tmp/` 为本地环境与临时目录，不交付源码。

## 运行（PowerShell）

本机实测 Python **3.12.9**、pip **24.3.1**、LangGraph **1.2.22**、pytest **9.2.0**。
已有虚拟环境可直接从下方的环境变量设置和运行命令开始，无需 activate。
从干净环境重建时执行完整命令：

```powershell
Set-Location F:\AI_Agent_Platform\experiments\langgraph_m3
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
$env:PYTHONUTF8 = '1'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe superstep_demo.py
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m pip check
```

安装需要联网，运行和测试不需要服务或 API Key。测试用 socket guard 拦截外部网络连接。
本机初次 ensurepip 因沙箱临时目录权限失败，将 TEMP/TMP 指向实验 .tmp 后成功；
pip 首次联网受沙箱限制，获自动审批后安装成功，没有修改项目原环境。
本轮没有更新根 requirements；以后是否集成依赖或跟踪文件由后续需求决定。

保存证据的命令（在上述目录和环境变量下）：

```powershell
New-Item -ItemType Directory -Force evidence | Out-Null
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements.lock.txt
.\.venv\Scripts\python.exe demo.py 2>&1 | Tee-Object -FilePath evidence/demo.txt
.\.venv\Scripts\python.exe -m pytest -v 2>&1 | Tee-Object -FilePath evidence/pytest.txt
.\.venv\Scripts\python.exe -m pip check 2>&1 | Tee-Object -FilePath evidence/pip-check.txt
```

## 主图、Send 和 Reducer

`StateGraph(PaperState)` 注册节点和边后 compile：

```text
START → ParsePaper → DispatchTasks → Send(AnalyzeContent × N / CheckReferences × N)
                                   → GenerateReport → END
```

ParsePaper 复制模拟章节并检查唯一 ID。DispatchTasks 记录计划数量；其条件边
`send_chapters` 返回 2N 个真实 Send，每条只携带当前章节的 chapter_id/content。
Python 循环只创建消息；执行节点的是 LangGraph Runtime。N=0 时直接生成空报告。

`content_results` 和 `reference_results` 使用 `Annotated[list[...], operator.add]`：
每个 worker 返回仅含自己结果的单元素列表，Runtime 用字段 reducer 合并更新。
Reducer 是字段机制，没有 Reducer Node，也不需要任务自行读旧列表再拼接。
`events` 同样使用 reducer 方便检查执行证据。**合并后的 events 列表不是完成时间线**，
不要从其排列推断并发完成顺序。报告按 chapter_id 字符串排序，本例使用补零 ID。

## Subgraph 与 Wrapper

引用子图为独立 `StateGraph(ReferenceState)`：
`START → ExtractReferences → ValidateReferences → GenerateReferenceResult → END`。
内部保存 references、failure_reason、trace 和 output。Wrapper 显式构造章节输入，
调用已 compile 的子图，等待返回后只把 output 包成主图 reference_results 更新；
内部中间字段不会写进主图，trace 只作为实验观测记录。

`[ref:missing]` 模拟引用不存在；没有引用也正常返回 failed。业务检查失败不抛异常，
所以同章节内容分析仍保留。未模拟基础设施异常恢复或真实文献验证。

## 完整汇聚的依据与适用边界

本拓扑所有 Send worker 都在父图同一个 super-step（实测 step 3）。虽然引用子图
内部有三个步骤，Wrapper 完成整次子图 invoke 后才结束这个父图任务。
两个 worker 节点分别用普通边指向同一个 GenerateReport；Runtime 等当前步所有任务
完成、合并更新后，在下一步（step 4）激活一次报告节点。

这不是根据图形推断：测试读取 Runtime 的 `tasks` 开始事件，核对 N=1/3/5 对应
2/6/10 个实际 worker、各自输入以及报告任务的完整输入；再读取 `updates` 检查仅有
一次报告输出。额外测试阻塞 ch02 引用 Wrapper，直到观察者收到全部内容分析更新才
释放它，最终仍只输出一次完整报告。Barrier 测试还验证三个 Send 内容任务确实重叠执行。
GenerateReport 内完整性检查只暴露缺失/重复结果，不负责等待、重试或补跑。

保证范围是这里的无循环、无重试、无 checkpoint 的固定拓扑；如以后将某条父图分支
拆成更深的独立节点或加入循环，不能沿用“普通边自然等齐所有路径”的假设，需重新设计汇聚。

## Super-step 独立实验

B 等到 C 启动后返回；C 阻塞在 Event 上。主线程从 stream 收到 B 的更新后，
确认 C 仍阻塞、D 尚未开始，再释放 C。D 检查 C 已结束且 State 中能读到 B、C 两份更新。
图只有 **B → D，没有 C → D**，因此能独立观察全局 super-step 边界。
10 秒 wait / Barrier 超时只作死锁保护，不以时间差判断执行顺序，无 sleep。

实际日志：

```text
step=1 A
step=2 C:blocked
step=2 B:return
step=None observer:B update received; C blocked; D not started
step=2 C:return
step=3 D:start (sees B,C)
```

`None` 是主线程观察者事件，不属于图节点。B 的更新可以先流出，但下一步 D 要等
B/C 所在步骤完成；更新被观察到和下游节点开始执行是两回事。

## 实测结果与阅读顺序

最终 pytest：**12 passed in 0.68s**，无失败/跳过；pip check：`No broken requirements found.`
完整输出见 evidence；N=3 报告包含三个章节，ch02 引用失败原因为“模拟引用不存在：missing”，
内容分析仍是 ok。还验证了空论文、空内容、重复/空 ID、同一 compiled graph 多次独立 invoke。
未运行冻结模块的全仓测试，因为本实验不导入或修改它们。

建议依次阅读：

1. `paper_review.py` 的 State 定义：先看字段和 Annotated reducer。
2. `build_paper_graph` 与 `send_chapters`：看图连接与动态任务输入。
3. `build_reference_graph` 与 `check_references_wrapper`：看两个 State 的边界。
4. `generate_report` 与测试 `test_send_tasks_isolation_reducers_and_report`、
   `test_report_waits_for_blocked_reference_wrapper`：看汇聚机制和实际验证依据。
5. `superstep_demo.py` 的 `run_superstep_demo`：看同步点和 step 边界。

实现依据为当前官方文档：[Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)、
[Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)、
[Control flow](https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow)。
