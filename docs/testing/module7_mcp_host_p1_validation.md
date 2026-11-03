# Module 7 · 两项 P1 定点修复验证

日期：2026-11-03（Asia/Shanghai）。本文件记录两项 P1 修复阶段（当时尚未提交）的验证。修复已通过最终定点 Review，Module 7 当前为 **COMPLETE / FROZEN**；最新验收与收口依据见 [最终验收报告](module7_mcp_host_final_acceptance.md)。

## 1. 基线与范围

分支 main，HEAD `b21f5e2148b5ad0a022cfa86e78b9b5521ffae22`。本轮开始时已有 Phase 2 未提交实现与文档，均保留；本轮没有安装或升级依赖。

本轮生产修改仅为：

- `backend/mcp_host/schema_adapter.py`：根据原始 alias 是否出现，修正动态模型声明字段的存在状态。
- `backend/mcp_host/manager.py`：跟踪资源进入/退出阶段、区分传播异常与新增退出异常、保留安全诊断及取消优先级。
- `backend/mcp_host/bootstrap.py`：OwnerError 作为基础设施失败传播；取消与 CleanupError 叠加时保留取消和清理诊断。

Tool Adapter 保持原有 handler 和 dump 调用，新增测试验证其实际转发参数。没有更改 Registry、Dispatcher、RegisteredTool、AgentRuntime、命名空间、重试/重连策略或独立 Web Page Fetch Server。

## 2. P1-1 根因与修复

Pydantic 把额外属性名也写入 model_fields_set。额外键 `field_0` 与声明字段内部名相同时，即使原始 alias `optional_text` 缺失，原来的 exclude_unset 也会认为该字段已提供，将占位 None 序列化为 optional_text:null。

修复不是更换前缀：原始 Draft 2020-12 校验和关键字白名单保持不变，before validator 改为 wrap validator；先校验原始 dict，调用 Pydantic 装配后，按原始 alias 的存在性移除声明字段的错误“已设置”标记。额外值仍在 model_extra 中，原生 `model_dump(mode='python', by_alias=True, exclude_unset=True)` 会继续保留它们。显式 null 只有在原始 Schema 允许时通过，default 仍只是元数据。

新增 21 组 Schema 矩阵（7 种输入 × additionalProperties 省略/true/false）覆盖：单/多内部名碰撞、多个可选字段、显式字段、缺失、合法/非法 null、额外字段禁止、default 不注入。每组以原始 Draft202012Validator 为对照，成功分支同时断言 dump 与原始输入相等、键集合相等、dump 仍通过原始 Schema。新增 Tool Adapter 测试检查实际发送给 call_tool 的 payload 未被补齐。

先增加测试复现旧问题：Schema 为 **6 failed、68 passed**；修复后该文件为 **74 passed**。没有修改断言或扩大输入接受范围来使测试通过。

## 3. P1-2 根因与修复

SDK 的 stdio 读取任务属于其 TaskGroup。非法 UTF-8 会让读取任务失败，异常可能在 transport 的 __aexit__ 才以 ExceptionGroup 传播。原 owner 将 AsyncExitStack 外层的所有普通异常都当成 cleanup_failed，导致已结束的通信故障被 Bootstrap 升级为全局 CleanupError。

当前通过标准上下文协议跟踪退出，不改变 SDK，不调用生产私有方法：

1. 进入失败记录 transport_enter/session_enter；initialize 代码块错误单独归为 initialization。
2. AsyncExitStack 登记带跟踪的 Session/transport 退出回调，传入原来的异常参数，保留原有传播/取消行为。
3. 对比退出异常叶子与输入异常对象的身份；单纯重抛或包装输入错误不算新增清理失败。
4. SDK 1.26 transport TaskGroup 中已知的 UnicodeDecodeError、AnyIO 管道资源错误和 ConnectionError 叶子记录 communication；与之混合的未知退出叶子仍记录 cleanup。
5. Session exit、新增未知 transport exit、stderr close 的实际异常仍置 cleanup_failed，保留 CleanupError。
6. stderr 打开等 owner 基础设施异常记录 owner，以 OwnerError 传播，不伪造清理失败，也不使用部分可用掩盖基础设施故障。

Connection.diagnostics 与 CleanupError.diagnostics 仅含 phase/kind/error_type，不保存远端自由文本或 traceback。用于区分异常身份的临时对象只在 owner 执行期间保留，不作为 Host 诊断输出。

## 4. 取消与清理失败优先级

**对外 CancelledError 优先；清理失败仍可观察。**

- 调用方在 connect/关闭等待期间取消，finish_cleanup 继续 shield 已保存的清理任务，不取消 owner。
- 若清理同时失败，CancelledError.__cause__ 为安全 CleanupError，后者包含 diagnostics；连接的 cleanup_failed/diagnostics 也保留。
- 若取消已从 Bootstrap/lifespan 代码块传出，而 finally 关闭失败，仍重新传播原取消并链接 CleanupError。
- 没有取消时继续直接抛 CleanupError；未知退出错误、混合异常组中的真实退出错误不被当作普通通信失败忽略。

新增测试覆盖启动取消+Session exit 异常、关闭中重复取消+Session exit 异常、lifespan 取消+关闭异常。它们检查 CancelledError、CleanupError cause、阶段诊断、owner 完成与同任务退出。另测试单独 Session/transport 退出异常、混合通信/清理组、异常透传以及 owner 自身错误。Bootstrap 也验证 OwnerError 不能被部分可用策略吞掉。

## 5. 真实 stdio 双 Server 证据

Server A 使用真实主环境 Python 子进程，向 stdout 写非法 UTF-8；Server B 使用独立 Server 解释器启动现有 Web Page Fetch FakeFetcher 测试入口。不依赖公网、不改生产 Server。

两种时序均断言：Host 成功启动；A 初始化失败且 Connection.state=failed、cleanup_failed=false；A 不注册工具；本地 get_weather 保留；B 正常注册并经真实 Dispatcher/stdio 调用成功；A 已退出，关闭后所有 owner 完成。

| 时序 | 本轮真实退出码 A/B | 结果 |
| --- | --- | --- |
| 不控制 SDK 读取时序 | **[1, 0]** | SDK 读取任务失败时终止 A；通信故障隔离，B 成功 |
| A 自然退出后 SDK 再读取管道中真实字节 | **[0, 0]** | A 自然退出，解码故障仍隔离，B 成功 |

第二种仅在测试的 SDK 进程创建观察钩子中等待 A 的真实 process.wait，再把原 Process 返回给官方 stdio 实现；没有替换读取流、字节、协议或资源退出函数。它用于确定性覆盖“进程已正常退出”的情形；第一种保持原始时序，保留 SDK 非零终止的事实，不把两者混淆为相同保证。

退出码 1 是 SDK 因通信任务异常关闭 Process 的实际行为，不是 Host 使用私有函数强杀，也不是资源泄漏判据。不宣称所有非法 UTF-8 Server 都自然退出为 0。

最终运行还保留原有强制回收/孙进程用例；五轮正常 stdio 启停后句柄观测为 **[259,259,259,259,259]**。这是本机有限轮次证据，不是所有故障场景的无泄漏证明。

## 6. 测试命令与结果

环境沿用初次 Phase 2 的 Python 3.12.9、Pydantic 2.13.4、MCP 1.26.0、jsonschema 4.26.0；未变更依赖。异步/真实进程测试使用获准沙箱外执行能力，避免本机 Windows 受限沙箱的 asyncio 限制。

分层阶段实际运行：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/mcp_host/test_schema_adapter.py -q
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/mcp_host/test_manager.py tests/mcp_host/test_stdio_integration.py::test_invalid_utf8_server_does_not_block_healthy_server -q -o faulthandler_timeout=20
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/mcp_host/test_schema_adapter.py tests/mcp_host/test_manager.py tests/mcp_host/test_tool_adapter.py tests/mcp_host/test_bootstrap.py tests/mcp_host/test_lifespan.py tests/mcp_host/test_stdio_integration.py -q -o faulthandler_timeout=25 --junitxml=.venv/module7-p1-targeted.xml
```

对应阶段结果：Schema 修复后 74 passed；初次 Manager+双 Server 为 9 passed；随后相关六层为 132 passed。后续增加自然退出时序和 OwnerError 的 Bootstrap 断言，最终以以下完整命令为准：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/mcp_host tests/tools -q -rP -o faulthandler_timeout=30 --junitxml=docs/testing/module7_mcp_host_p1_tests.xml
```

最终 **322 passed，0 failures/errors/skipped**；原始记录见 [P1 专项及 Tool 回归 JUnit](module7_mcp_host_p1_tests.xml)。

| 范围 | 通过数 |
| --- | ---: |
| Schema Adapter | 74 |
| Manager | 15 |
| Bootstrap | 13 |
| Lifespan | 5 |
| Tool Adapter | 16 |
| 真实 stdio | 11 |
| Config | 16 |
| MCP Host 小计 | **150** |
| Tool System（Dispatcher/Registry/布局/Weather） | **172** |
| 合计 | **322** |

没有新增 skip，没有修改历史 Memory/RAG 测试，没有重新运行耗时全仓或公网验收；初次全仓和公网证据保留为历史记录，不能视为本轮重跑。

## 7. 文档、风险与 Git

ARCHITECTURE.md 已明确 main 的 Phase 1 基线与当前 lifespan，更新 wrap validator、字段存在性修复、异常阶段/叶子归属与取消优先级。READING_GUIDE.md 同步真实资源退出流程和诊断读取方式；初次验证报告增加本轮报告链接，保留历史数字。

仍存边界：异常归属针对固定 SDK 1.26 的已知 TaskGroup 通信类型；未知退出异常保守报清理失败，不能用通信异常白名单覆盖所有将来 SDK 行为。Pydantic 升级需复跑碰撞/序列化矩阵。OS 强杀、Job 分配失败、系统进程创建永久阻塞等已有风险没有在本轮扩大解决；取消不意味着远端工作必然撤销。

P1 修复阶段修改上述三份生产文件、相关测试/fixture、架构/导读和验证文档；新增本报告与 JUnit。未触碰核心契约、独立 Server 或旧路由；git diff --check 通过。该阶段结束时分支/HEAD 不变，Phase 2 与修复文件尚未提交。随后已通过最终定点 Review 和专项验收，最终收口以最新报告与 Git 记录为准。
