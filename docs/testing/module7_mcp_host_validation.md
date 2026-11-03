# Stage 3 Module 7 · MCP Host Phase 2 验证报告

日期：2026-11-03，Asia/Shanghai。本文件记录初次实现阶段（当时尚未提交）的历史验证。Module 7 当前已 **COMPLETE / FROZEN**，最新定点 Review、验收与 Git 收口依据见 [最终验收报告](module7_mcp_host_final_acceptance.md)。

本文保留初次 Phase 2 实现的验收记录。最终 Review 随后发现的两项 P1 已进行定点修复；最新结果、异常优先级和新增资源证据见 [P1 修复验证报告](module7_mcp_host_p1_validation.md)，不应以本文的历史 117 项覆盖范围替代修复后验证。

基线为 `main@b21f5e2148b5ad0a022cfa86e78b9b5521ffae22`。开始时仅有 Phase 1 新增的未跟踪 `docs/mcp_host_integration/ARCHITECTURE.md`。本报告不把既有套件错误隐藏成“全仓通过”。

## 1. 实现与文件范围

| 文件 | 实际职责 |
| --- | --- |
| backend/mcp_host/__init__.py | Host 包边界 |
| backend/mcp_host/config.py | 静态配置、全局歧义校验、Server 级非法配置隔离、根路径解析 |
| backend/mcp_host/manager.py | owner task、Session initialize、ready/stop、串行租约、连接健康、清理与清理失败状态 |
| backend/mcp_host/schema_adapter.py | 有限白名单、Draft 2020-12、原始 JSON 校验、动态 Pydantic 字段/alias/hook、资源限制 |
| backend/mcp_host/tool_adapter.py | ToolBinding、命名空间 RegisteredTool、远程调用、总预算、结构化结果及固定安全错误 |
| backend/mcp_host/safe_logging.py | 定向屏蔽 SDK 自由文本，退出后恢复过滤状态 |
| backend/mcp_host/bootstrap.py | 完整分页、整 Server 候选组装、部分可用、最终 Registry/Runtime 与 lifespan |
| backend/config/mcp_servers.json | 默认启用独立 Web Page Fetch，使用独立解释器 |
| backend/main.py | 仅增加 lifespan 导入和 FastAPI 参数，原数据库初始化和七个路由注册保留 |
| requirements.txt | 新增直接依赖 mcp==1.26.0、jsonschema==4.26.0 |
| tests/mcp_host/ | 7 个分层测试模块、conftest、包隔离、协议/Selector 支持脚本及单独公网验收脚本 |
| docs/mcp_host_integration/ARCHITECTURE.md | 命名空间修订、实际实现差异与生命周期说明 |
| docs/mcp_host_integration/READING_GUIDE.md | 完整调用链和源码阅读顺序 |
| README.md | Module 7 进度与文档链接 |

另交付本报告、[专项 JUnit](module7_mcp_host_tests.xml) 和 [公网验收 JSON](module7_mcp_host_public.json)。没有修改 ToolRegistry、ToolDispatcher、RegisteredTool、AgentRuntime、DeepSeekAdapter、Memory、RAG 或独立 MCP Server。

本地名为 `mcp_web_page_fetch__fetch_web_page`，远程调用名始终为 `fetch_web_page`。应用 lifespan 发布 `app.state.agent_runtime`，旧 `/chat`、`/chat/stream` 仍使用原有路径，没有新增 API。

## 2. 环境与依赖

Windows，Python 3.12.9。主环境 FastAPI 0.141.1、Starlette 1.6.0、AnyIO 4.14.2、Pydantic 2.13.4、OpenAI 3.1.0、Uvicorn 0.52.4；传统 httpx/httpcore 为 0.28.1/1.0.9，httpx2/httpcore2 为 2.10.0/2.10.0。

依赖步骤实际执行：

1. 主环境完整 requirements 加两项新依赖的 pip dry-run，保存解析报告于忽略的 `.venv/mcp-phase2-resolve.json`；现有 requirements 均已满足。
2. 从已安装分发元数据生成 `.venv/mcp-baseline-constraints.txt`，冻结全部既有版本。
3. 在 `.venv/mcp-phase2-check` 新建独立 venv，以该约束安装 MCP/jsonschema/FastAPI/OpenAI/httpx2 及完整传递依赖，执行 pip check 通过。此环境验证集成共享依赖闭包，没有复制整个 ML 环境。
4. 主环境同样用冻结约束安装 MCP/jsonschema；pip check 通过。对比安装前后完整分发版本，**changed_existing=[]**。

新增包及版本：mcp 1.26.0、jsonschema 4.26.0、attrs 26.1.0、cffi 2.1.1、cryptography 50.0.2、httpx-sse 0.4.3、jsonschema-specifications 2025.9.1、pycparser 3.12、pydantic-settings 2.16.2、PyJWT 2.15.1、pywin32 312、referencing 0.37.0、rpds-py 2026.9.1、sse-starlette 3.6.0。

没有整体升级、替换 HTTP 栈或修改独立 Server .venv。Server 仍有独立 AnyIO 4.16.0 / Starlette 1.10.0；它通过 stdio 与主环境隔离，不导入 backend。

核心命令（从仓库根运行，独立 Server 命令除外）：

```powershell
.\.venv\Scripts\python.exe -m pip install --dry-run --report .venv/mcp-phase2-resolve.json -r requirements.txt mcp==1.26.0 jsonschema==4.26.0
.\.venv\Scripts\python.exe -m venv .venv/mcp-phase2-check
.\.venv\mcp-phase2-check\Scripts\python.exe -m pip install -c .venv/mcp-baseline-constraints.txt mcp==1.26.0 jsonschema==4.26.0 fastapi==0.141.1 openai==3.1.0 httpx2==2.10.0
.\.venv\mcp-phase2-check\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip install -c .venv/mcp-baseline-constraints.txt mcp==1.26.0 jsonschema==4.26.0
.\.venv\Scripts\python.exe -m pip check
```

## 3. 分层测试

最终专项命令：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/mcp_host -q -rP -o faulthandler_timeout=30 --junitxml=docs/testing/module7_mcp_host_tests.xml
```

最终：**117 passed in 22.78s，0 failures/errors/skipped**。

| 层 | 数量 | 主要验证 |
| --- | ---: | --- |
| Config | 16 | 正常/非法配置、重复 ID/JSON key、相对路径、disabled、预算类型/边界、环境隔离 |
| Manager | 8 | owner 同任务进入退出、锁、关闭拒绝新租约、失败/超时、重复取消、不可用、重复关闭、关闭超时及退出异常 |
| Schema Adapter | 53 | 原始 jsonschema 对照、缺失/null、additionalProperties、整数/boolean/数字字符串、enum/const、Unicode 长度、非法约束、默认值、alias/hook、非 JSON/深度/大小/循环 |
| Tool Adapter | 15 | 远程原名、ToolContext 不转发、固定错误、无重试、超时包含锁等待、无效结果、取消/含取消异常组 |
| Bootstrap | 12 | 多 Server、原子注册、同名隔离、名称超限/非法、冲突、完整分页/空 cursor、坏后页、全部失败、基础设施失败、发现上限 |
| Lifespan | 4 | ASGI startup/shutdown 消息、部分可用、在途租约排空、启动失败、启动取消、生产 main 接线和无新增路由 |
| 真实 stdio | 9 | 完整 Agent Loop、两页缓存、崩溃、超时、坏输出、畸形协议、Job 强制回收/孙进程、失败隔离/真实启动取消、句柄观测 |

Mock 与真实 stdio 分开：Config/Schema 是同步纯单元；fake_sdk 模拟连接以精确控制任务、退出及错误；真实 stdio 用正式 SDK 创建 Python 子进程并传输 JSON-RPC。自动化抓取成功使用现有 FakeFetcher，不依赖公网，不替代生产网络验收。

## 4. 回归范围与实际结果

```powershell
# 定向 Tool/Runtime/Frontend/日志/MCP 回归
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/tools tests/agent_runtime tests/frontend tests/logging tests/mcp_host -q -o faulthandler_timeout=30 --junitxml=.venv/module7-regression.xml

# 主项目全部测试
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests -q -o faulthandler_timeout=40 --junitxml=.venv/module7-main-full.xml

# 仓库根默认收集
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q -o faulthandler_timeout=40 --junitxml=.venv/module7-root-final.xml

# 独立 Server；cwd = mcp_servers/web_page_fetch
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests -q -o faulthandler_timeout=25 --junitxml=../../.venv/module7-server-regression.xml
```

| 范围 | 实际结果 | 边界 |
| --- | --- | --- |
| 最终主项目 tests/ | **1033 passed、35 skipped、4 errors，54.35s；0 assertion failures** | 四项缺 reader fixture，不能称全量通过 |
| 独立 Server 自己的环境 | **143 passed，7.20s** | 没有修改其生产代码/测试/依赖 |
| 定向组合回归 | **674 passed、1 failed，46.99s** | 原有 DeepSeek MockTransport 的 1 秒预算间歇超时，详情如下 |
| 原始根目录 pytest 最终 | **5 collection errors，22.38s** | 独立 Server 包从根目录不可导入；在其专用 cwd/env 跑全部通过 |

四项 errors 均在未改动的 `tests/memory/test_memory_reader_v2_e2e.py`：test_langgraph_lexical（231）、test_langgraph_semantic（359）、test_no_relevant_memory（482）、test_no_memory（580）。错误为 fixture 'reader' not found。35 项 skip 由原有测试自身控制，没有新增 skip 绕过失败；本机 Windows 的专项没有 skip。

定向组合中的 `test_real_sdk_mock_transport_and_sdk_response_model` 在 OpenAI SDK `_platform = await asyncify(get_platform)()` 的线程等待耗尽其现有 1 秒预算，随后旧测试的 translate_error=Mock() 引出 TypeError。该文件和 DeepSeekAdapter 均未改。将 HEAD 的 backend/测试通过 git archive 解包到忽略的 `.venv/module7-baseline` 后，单独执行相同测试为 **1 passed in 2.28s**；最终主项目全量运行该用例也通过。保留这一间歇性观测，不提高预算、不预热/monkeypatch 生产 SDK、不将组合失败写成成功，也不声称已经确定其根因。

根目录默认收集的五个错误是独立 Server 的 test_connection_security、test_contracts、test_fetcher、test_html_text、test_security 导入 web_page_fetch 失败。本次新增 tests/mcp_host/__init__.py 消除了新增测试与旧 test_bootstrap 同名导致的收集冲突；其余五项不通过修改独立 Server 导入布局来掩盖。

## 5. 真实 stdio 与公网验收

自动化全链路确实经过：FakeLLM → AgentRuntime → Registry → Dispatcher → 动态参数模型 → MCP Tool Adapter → Manager 租约 → 官方 SDK stdio → 真实独立测试 Server → structuredContent → ToolSuccess → Runtime → FakeLLM 最终回答。检查五字段中文 Fixture 内容、远程安全失败、失败后同 Session 再次成功及进程退出。

两页真实测试 Server 的每次响应包含 tools/list 累积计数；连续调用最后一页工具三次后计数始终为 2，remote 始终为 last。无需读/写 SDK 私有缓存。运行期进程退出码 7 场景保留工具注册，后续调用安全报不可用，未重启；坏 outputSchema 结果返回 MCP_INVALID_RESULT；挂起调用返回 MCP_TIMEOUT。

单独生产公网命令：

```powershell
.\.venv\Scripts\python.exe -B tests/mcp_host/manual_web_fetch.py
```

本轮结果见 [公网 JSON](module7_mcp_host_public.json)：

- 使用默认静态配置和生产 `python -B -m web_page_fetch`，没有替换 Fetcher/Resolver/Transport，也没有放宽 SSRF。
- FakeLLM 两次 complete，最终回答“公网网页读取成功”；没有真实 LLM API 请求。
- example.com 返回 title=Example Domain、正文 156 字符，五个结构化字段正确，content_type=text/html。
- 非法 file scheme、localhost 带 query、私有 IPv4 三次均经 Dispatcher 返回安全 MCP_TOOL_ERROR；同 Session 随后再次公网调用成功。
- 协议请求计数：initialize=1、tools/list=1、tools/call=5；无调用阶段重新发现。
- Server returncode=0、强制终止=0、全部 owner task 结束。

该证据验证生产公网成功链，不等同穷尽 DNS/TLS/SSRF 攻击面；独立 Server 的更细网络安全边界仍参见其现有架构/验收文档。

## 6. 生命周期与资源专项

- fake_sdk 对所有 enter/exit 记录 asyncio.current_task，断言 Session/stdio 的进入和退出只有同一个 owner；逆序为 Session exit → transport exit，stderr sink 在资源退出后关闭。
- 真 stdio 正常 EOF 关闭 exit=0，不调用 SDK 强制终止；坏路径 Server 启动失败后另一真实 Server 仍可调用。
- initialize 无响应的真实子进程：取消启动调用方后，Manager 通过 stop 取消并收集初始化请求、在原 owner 内关闭上下文；子进程最终退出。
- fake_sdk 重复取消、ASGI 启动取消均向上传播 CancelledError，清理不会在借用者任务执行。
- 在途租约关闭期间新租约被拒绝；既有租约释放后完成关闭。
- 假退出挂起超过观测预算明确抛 CleanupError，cleanup_failed=true、owner 未被取消；释放测试事件后 owner 可以完成。模拟 SDK exit 异常也会显式报告并继续执行 transport 清理。
- Windows stubborn Server 启动一个 120 秒孙进程，收到 EOF 后仍不退出。测试确认 SDK 已分配 Job Object、触发其强制回收，Server 已退出，孙进程 PID 不存在或其退出码已不为 STILL_ACTIVE。测试 finally 只允许兜底清理该测试创建的 PID；生产没有私有终止逻辑。
- 五轮真实 stdio 初始化/关闭后 GC 观测本测试进程句柄数：**[259, 259, 259, 259, 259]**。这是五轮本机总句柄观测，不是所有路径永不泄漏的证明。
- 单独 Selector fallback 探针以 asyncio.SelectorEventLoop 启动官方 SDK，结果 exit=0、`selector_success_and_shutdown`。它验证当前 Windows/SDK 组合，不代替所有 Uvicorn worker/reload 模式现场测试。

Selector 命令（建议外部进程设置 25 秒上限，本轮已设置）：

```powershell
.\.venv\Scripts\python.exe -B tests/mcp_host/support/selector_probe.py
```

## 7. 开发中发现并修正的问题

1. 初始化默认 read timeout 误用了测试的短调用预算：改用启动预算，tools/call 单独覆盖调用预算。
2. 启动取消只设置 stop 仍可能等待整个 initialize：增加 initialize 请求/stop 的竞争等待，SDK 上下文仍由 owner 进入/退出。
3. SDK 默认跳过畸形协议 Exception 后可能返回后续合法结果：Manager 消息处理器标记协议健康失败，当前租约安全报协议错误，未来租约不可用；日志 canary 不泄漏。
4. AsyncExitStack 在进入时保存 exit 回调，最初测试事后替换类 __aexit__ 未能注入失败：改为 fixture 自身可控退出事件/异常；保留原失败断言，实际验证 CleanupError，没有降低标准。
5. 测试同名收集：只为新增测试目录建立包边界，不重命名/改动旧测试。

## 8. 已知限制与 Review 重点

- 全仓不是全绿：保留四项 Memory fixture 错误、根目录独立包五项收集问题和组合运行 SDK 冷启动超时观测。没有为本模块擅自修复旧稳定组件。
- 首版仅平面有限 Schema；format/$ref/组合/数组/嵌套对象/数值边界约束等拒绝整 Server 注册。服务资源上限外的 JSON 显式拒绝；不宣称无限输入语义等价。
- SDK 公开 stdio API 不返回 Process。Job Object 分配失败、OS 强杀 Host、进程创建系统调用永久阻塞、解释器崩溃的进程树/句柄回收没有全场景保证。不引入自建 Transport 或生产私有 SDK 方法。
- 关闭是有界观测而非硬性强杀保证：10 秒排空 + 10 秒 owner 退出；超时记录并抛 CleanupError，保留 owner 供继续完成。应用/OS 随后强制停止时不宣称能完成后台清理。
- 未覆盖真实 OS 强杀 Host、Job 分配失败现场、所有 Uvicorn worker/reload 组合、创建阶段永久阻塞；未做长时间压力/全路径句柄审计。
- Server 工具列表变化不刷新，崩溃不注销/重连；每个 ASGI worker 持有自己的 Server，首版建议单 worker。
- 取消/超时只保证本地边界，不能确认远端业务已撤销；所有 MCP 错误 retryable=false，无自动重试。
- stderr 丢弃，SDK 自由文本日志定向过滤；这是首版隐私选择，会牺牲底层诊断细节。没有新建指标/管理 API。
- 本次没有把旧聊天切换到 Agent Loop，需由后续明确授权的接口工作使用 app.state.agent_runtime；不自动新增 /agent/chat。

## 9. Git 与交付状态

仅修改 README.md、requirements.txt、backend/main.py；新增 backend/mcp_host、backend/config/mcp_servers.json、tests/mcp_host，以及架构/导读/本报告/测试证据。Phase 1 文档从原未跟踪文件继续完善。既有核心模块和独立 Server 无 diff；git diff --check 通过。HEAD 与分支保持基线，不执行 Commit/Push/Tag/Release。

阅读入口：[架构](../mcp_host_integration/ARCHITECTURE.md)、[代码导读](../mcp_host_integration/READING_GUIDE.md)。以上为初次 Phase 2 当时的记录；后续 P1 修复与最终冻结状态以对应报告为准。
