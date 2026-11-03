# Module 7 · MCP Host Integration · 最终验收

日期：2026-11-03（Asia/Shanghai）。状态：**COMPLETE / FROZEN**。

最终 Review 与验收基线：main，HEAD `b21f5e2148b5ad0a022cfa86e78b9b5521ffae22`，其上包含此前未提交的完整 Phase 2 实现、两项 P1 修复和证据。本轮没有新增功能、安装依赖或重构稳定组件；通过验收后按用户授权统一提交并推送 origin/main，实际提交标识以 Git 记录为准。

## 1. 最终定点 Review

未发现新的 P0/P1，准许验收和 Git 收口。

- P1-1：wrap validator 仍以原始 Draft 2020-12 Schema 校验输入；字段装配后依据原始 alias 是否存在修正声明字段的 model_fields_set。额外属性保留在 model_extra，内部名碰撞不再补出缺失参数；合法 null、额外属性策略和 default 元数据语义保持正确。Tool Adapter 继续按原 Dispatcher 契约转发 lossless dump，命名空间不变。
- P1-2：退出回调在原 owner task 中执行，区分输入异常的传播与新增退出异常；已知 SDK TaskGroup 通信叶子和真正清理异常分开记录。混合组中的真实退出错误仍置 cleanup_failed 并保留 CleanupError；单个通信故障清理完成后不阻断其它 Server。
- 取消/清理组合：对外 CancelledError 优先，安全 CleanupError cause 和阶段诊断保留；没有跨任务调用 SDK __aexit__，没有新增重连或重试。

复核依据：[架构](../mcp_host_integration/ARCHITECTURE.md)、[导读](../mcp_host_integration/READING_GUIDE.md)、[P1 报告与测试](module7_mcp_host_p1_validation.md)。

## 2. 最新专项验收

实际执行用户指定命令：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/mcp_host tests/tools -q -rP -o faulthandler_timeout=30
git diff --check
.\.venv\Scripts\python.exe -B -m pip check
```

结果：**322 passed in 27.08s，0 failures/errors/skipped**；其中 MCP Host 150、Tool System 172。git diff --check 无输出，pip check 为 No broken requirements found。

本轮运行没有重写历史 JUnit；[P1 JUnit](module7_mcp_host_p1_tests.xml) 保留上次同一生产代码的 322 项逐项证据，最新命令结果以本节记录为准。

最新真实 stdio 证据：

- 非法 UTF-8 A 与健康 Web Page Fetch 测试 Server B 共存：A failed、cleanup_failed=false，不注册工具；本地工具保留，B 经 Dispatcher 调用成功。
- 未控制 SDK 读取时序时实际退出码为 **[1,0]**；等待 A 先自然退出、再读取真实管道字节的确定性测试为 **[0,0]**。两种时序所有 owner 均结束，不将 SDK 终止退出码 1 宣称为自然退出。
- Session/transport 真实退出异常模拟仍能识别 CleanupError；启动取消、重复取消、lifespan 取消叠加清理失败均保留取消和安全诊断。
- 正常生命周期、强制回收/孙进程等已有专项继续通过；五轮句柄观测 **[259,259,259,259,259]**。

自动专项不访问公网。之前生产 Web Page Fetch 公网成功与独立 Server 143 项回归保留为历史证据，未在本轮重复执行；详见 [初次验证报告](module7_mcp_host_validation.md) 与 [公网 JSON](module7_mcp_host_public.json)。

## 3. 完成能力与冻结边界

已完成：MCP Host 基础架构、MCPClientManager/Session 生命周期、有限 JSON Schema → Pydantic 动态适配、MCP Tool Adapter、Host Bootstrap、Server 级原子注册、Partial Availability、FastAPI lifespan、真实 stdio 调用，以及错误处理/资源清理/专项评测。

首版仅使用 stdio 与静态 Server 配置；不支持动态管理、自动重连、热更新，不把平台暴露为 MCP Server。旧 `/chat` 和 `/chat/stream` 未切换至 AgentRuntime；Runtime 通过 app.state 注入。没有修改 Registry、Dispatcher、AgentRuntime 核心契约或独立 Server。

Stage 3 继续 **IN PROGRESS**：

| Module | 状态 |
| --- | --- |
| M1 Tool Contract | COMPLETE |
| M2 Function Calling | COMPLETE |
| M3 Registry / Dispatcher | COMPLETE |
| M4 Minimal Agent Loop | COMPLETE |
| M5 State / Context | COMPLETE |
| M6 Error / Retry / Limits | COMPLETE |
| M7 MCP Integration | **COMPLETE / FROZEN** |
| M8 Evaluation / Observability | **NEXT** |

Module 8 仅更新路线图状态，没有开始实施。

## 4. 已知限制与历史问题

有限 Schema 子集、固定 SDK 的异常归属边界、未知退出异常保守报清理失败、取消不保证远端撤销、OS 强杀/Job 分配失败/系统创建永久阻塞等风险继续保留。句柄观测不是全路径永久无泄漏证明；建议首版单 worker，升级 SDK/Pydantic 后复跑专项矩阵。

历史主项目全量有 4 个 Memory reader fixture errors，根目录默认收集有独立 Server 的 5 项导入问题，另有组合回归 SDK 冷启动超时观测。它们在旧报告中如实保留，没有修改相关测试，也不计入本次 322 项专项验收。本轮按授权不重复运行耗时全仓测试。

## 5. Git 收口范围

纳入完整 Module 7：backend/mcp_host、静态 Server 配置、main lifespan 接线、两项直接依赖、tests/mcp_host、架构/导读/各阶段验证证据，以及 README 状态同步。排除 .venv、缓存、临时解析/基线快照、.env、数据库和无关文件。

提交说明：`feat(mcp): integrate MCP host into agent platform`；目标 origin/main，不 force push、不创建 Tag/Release。完成后核对本地 HEAD、远端 main 与工作区状态，在最终交付中报告实际结果。Module 7 到此冻结，等待用户开启下一阶段学习。
