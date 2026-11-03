# Module 7 · MCP Host Integration · 架构与冻结边界

日期：2026-11-03（Asia/Shanghai）。状态：**COMPLETE / FROZEN**。最终定点 Review 和专项验收通过，收口记录见第 16 节。Phase 1 核验历史保留在第 10、13 节，Phase 2 实际交付与差异见第 14、15 节。

基线：`F:\AI_Agent_Platform`，HEAD `b21f5e2148b5ad0a022cfa86e78b9b5521ffae22`，Phase 1 开始时工作树干净；Phase 2 开始时仅有上阶段新增的本架构文档。宿舍使用相同仓库相对布局，根目录为 `D:\AI_Agent_Platform`。本文第 3 节接口已按职责实现，精确函数签名以同目录阅读导读所链接代码为准。

## 1. 范围与结论

平台是 MCP Host，内部持有 MCP Client；不把平台暴露为 MCP Server。首版仅支持 stdio、静态配置、初始化完整分页发现、运行期内存注册、Server 级原子注册、部分可用启动。

保持 ToolRegistry、ToolDispatcher、AgentRuntime 及其核心契约不变。使用现成的 `build_agent_runtime(registry=...)` 注入入口。FastAPI lifespan 创建并持有集成实例，注入 `app.state.agent_runtime`，不增加 API，也不改 `/chat`、`/chat/stream`。因此本阶段设计的 Runtime 接线不会使现有聊天路由自动改用 Agent Loop。

不实现动态管理、热更新、自动重连、运行期注销、自动重试、RAG Tool、通用 JSON Schema 转换或独立 Server 重写。

核验后的两项必要细化：

1. AsyncExitStack 可以用于 SDK，但必须在同一个资源持有任务内按相反顺序退出。每个 Server 使用独立持有任务，避免把所有 SDK TaskGroup 的取消作用域嵌套到 FastAPI lifespan 任务中。
2. 先完整调用当前 Session 的 `list_tools()`，再允许适配器调用发现过的远程名称。SDK 会缓存每页工具的 outputSchema（包括 None），从而避免 `call_tool()` 成功后隐式再发现。无需私有缓存写入、SDK 补丁或改用低层调用。

## 2. 已有代码契约

| 组件与位置 | 已确认契约 | 集成方式 |
| --- | --- | --- |
| `backend/tools/infrastructure/contracts.py` | frozen RegisteredTool(name, description, args_schema: type[BaseModel], handler)；ToolHandler 为 async (BaseModel, ToolContext) → dict[str, Any] | 适配器生成普通 RegisteredTool；handler 必须是 async def 闭包，不能直接传异步可调用对象实例 |
| 同上 ToolContext | user_id / conversation_id / SQLAlchemy Session 均可空 | 只用于可信本地上下文，不序列化发给 MCP Server |
| `registry.py` | register/get/list_tools；重复名和非 async handler 被拒绝；没有事务、注销、批量注册 | 仅对未发布的新 Registry 组装候选集 |
| `dispatcher.py` | model_validate(arguments)；ValidationError → INVALID_ARGUMENTS；只捕获 ToolExecutionError；handler 返回 dict 后用 json.dumps(allow_nan=False) 检查 | Adapter 必须提前规范化 MCP 错误和无效结果，避免 TypeError 泄漏到 Runtime |
| `errors.py` | ToolExecutionError(code, message, retryable=False) | 使用现有错误，不新增 MCP 专属 Runtime 分支 |
| `backend/tools/bootstrap.py` | build_tool_registry() 新建 Registry，注册 get_weather | 本地工具作为优先保留的种子集合 |
| `backend/agent_runtime/bootstrap.py` | build_agent_runtime(*, llm=None, registry=None, client=None, ...) | 注入最终 Registry；原函数自行构造对应 Dispatcher，无需调用方另造一个 |
| `runtime.py` | 从 args_schema.model_json_schema() 生成 LLMToolDefinition；每轮按顺序 dispatch；工具执行不在 LLM 重试边界 | Schema 模型还必须向 LLM 暴露与原始 Schema 一致的参数定义 |
| `backend/main.py` | Phase 1 基线：模块加载时配置日志、create_all，无 lifespan；Phase 2 当前：FastAPI(title=..., lifespan=lifespan) | 已接入 Host lifespan，保留原 DB 初始化和路由 |

`build_agent_runtime` 默认路径可能导入既有 llm_service，使用其 Client；Client 归原有调用方所有，Host 不擅自关闭该共享对象。测试注入 LLMPort 可以完全绕开旧服务和真实凭据。基础设施、配置解析器本身或本地工具组装失败不属于“远程 Server 失败”，应使启动失败并清理已启动 MCP 资源。

## 3. 组件、接口与数据流

建议新增 `backend/mcp_host/`，不放进通用 infrastructure：

| 组件 | 职责 | 明确不承担 |
| --- | --- | --- |
| config | 读取静态配置、校验 Server ID/路径/预算 | 执行远程工具 |
| MCPClientManager | 建立 stdio、创建并 initialize ClientSession、连接状态、有效 Session 租约、关闭资源 | tools/list、Schema 转换、注册、tools/call 业务编排 |
| Host Bootstrap | 调度初始化、分页发现、适配、冲突检查、原子候选组装、最终 Registry 与 Runtime 构建 | 协议传输实现 |
| Schema Adapter | 有限 Schema 审核与动态 Pydantic 模型生成 | 接受未知校验关键字 |
| MCP Tool Adapter | 绑定 server_id/remote_name、tools/call、结果检查、错误转换 | Session 建立、重连、重新发现 |

接口约定（实际实现使用 Connection / Host / ToolBinding 数据类）：

- `await manager.connect(config) -> None`：启动独立 owner 任务；等待 initialize 成功或失败。须先将 owner 登记为待清理资源，再等待 ready，避免启动调用方取消时漏掉连接。
- `async with manager.session(server_id) as session`：只向受信任的 Bootstrap/Adapter 提供有效 Session；检查 ready/closing/unavailable，取得 Server 级锁，再复查状态。租约禁止退出或替换 Session。
- `await manager.close_server(server_id)` / `await manager.aclose()`：幂等，发送停止信号并等待 owner 在原任务内清理，不由调用方直接执行 SDK __aexit__。
- `manager.mark_unavailable(server_id)`：幂等、单向状态变化，阻止新租约；不操作 Registry。
- `discover_tools(session, limits) -> tuple[types.Tool, ...]`：Bootstrap 所有，完整分页，无全局注册副作用。
- `build_args_model(input_schema) -> type[BaseModel]`：通过白名单审核后生成模型。
- `adapt_tool(server_id, remote_tool, args_model, manager, call_timeout) -> RegisteredTool`：返回标准异步 handler。
- `async with bootstrap_host(configs, *, llm=None) as host`：创建 Manager，组装 registry/runtime/statuses；失败和退出均关闭 Manager。注入 llm 仅用于测试，生产沿用已有构造器默认行为。

初始化流：静态配置 → 本地工具种子 → 各 Server connect/initialize → 当前 Session 全分页 tools/list → Schema/元数据审核 → 临时候选组 → Server 原子纳入 → 最终 Registry → build_agent_runtime(registry=...) → app.state。

调用流：AgentRuntime → ToolDispatcher.model_validate → MCP async handler → Session 租约 → session.call_tool(remote_name, arguments) → SDK outputSchema 校验 → Adapter 结果/JSON 检查 → dict → ToolSuccess。任意 MCP 失败通过 ToolExecutionError → ToolFailure；外部取消继续向上传播。

## 4. 静态配置与命名

建议 `backend/config/mcp_servers.json`：顶层 servers 有序数组，每项 server_id、enabled、command、args、cwd、env、startup_timeout_seconds、call_timeout_seconds。不引入新配置库；使用现有 Pydantic 校验。禁用项不启动。

Web Page Fetch 配置解析结果应为：

```text
server_id = web_page_fetch
command   = <repo>/mcp_servers/web_page_fetch/.venv/Scripts/python.exe
args      = ["-B", "-m", "web_page_fetch"]
cwd       = <repo>/mcp_servers/web_page_fetch
env       = {"PYTHONUTF8": "1"}
startup_timeout_seconds = 15
call_timeout_seconds    = 40
```

相对路径以仓库根解析，不能依赖启动终端 cwd；支持受信任部署配置显式指定解释器。Host 使用根 `.venv`，子 Server 使用独立 `.venv`，不通过 sys.path 拼接借用 Server 依赖。command/args 直接传 SDK，不拼 shell 命令。env 仅传显式项；SDK 会与其默认环境白名单合并，不转发整个 os.environ 或主项目 .env。配置不是 LLM 输入。

server_id 必须唯一；顶层 JSON 语法、重复 ID 等全局歧义使启动失败。结构可独立归属到某 Server 的路径、预算、连接或工具错误按该 Server 失败隔离。配置中无 Server 或全部禁用时仅构建本地工具。

按 Phase 2 用户补充授权，本地 RegisteredTool.name 使用 `mcp_{server_id}__{remote_tool_name}`，例如 `mcp_web_page_fetch__fetch_web_page`。ToolBinding 分别保存 server_id 和 remote_tool_name；tools/call 只发送 `fetch_web_page` 原名。server_id 和最终注册名均须完整匹配 `[A-Za-z0-9_-]{1,64}`；禁止截断，超限或非法字符使整个 Server 注册失败。不同 Server 的相同远程名称可以共存，仍检查与本地工具的名称冲突。description 缺失转空字符串，其余元数据不作为权限声明。工具 description/output 属于不可信数据，不改变 Host 指令。

## 5. Server 级原子注册与部分可用

按静态配置顺序处理，首版不并行初始化，结果与机器调度无关。本地工具名优先；较早成功 Server 优先；较晚冲突 Server 整体失败。

每个 Server 执行：

1. 建连并 initialize，确认 tools capability 存在。
2. 全分页发现并核验同 Server 内名称唯一。空集合可以作为成功但无工具的 Server。
3. 在局部内存构造全部 RegisteredTool；任意输入 Schema 不支持、outputSchema 非法、名称不合规、重名、handler 无效均丢弃整组。
4. 新建一个临时 ToolRegistry，把本地及之前已接受的工具重新 register，再 register 本组全部候选。不发布、不修改旧 Registry。
5. 全部成功才替换 Bootstrap 持有的候选 Registry。失败就丢弃临时实例，并 close_server；之前候选保持不变。
6. 全部 Server 处理后构建 Runtime，最后一次性发布 host/registry/runtime/statuses 到 app.state，期间没有请求可见的半成品。

如果最终 Runtime 构建或 app.state 发布前失败，关闭全部成功连接后传播异常。Server 初始化失败仅记录固定分类及 server_id，不将异常对象/自由文本写入 LLM 结果。所有 MCP Server 失败仍保留 get_weather，并允许基础设施正常的应用启动。资源清理无法完成时不能谎报成功清理：保留显式 cleanup_failed 状态并尝试继续清理其他 Server；若启动阶段连资源归属都无法保证，应视为 Host 基础设施故障，不能套用部分可用掩盖。

运行期崩溃：RegisteredTool 不注销；Manager 拒绝新调用或由本次调用发现断连并标记 unavailable，后续安全返回 MCP_SERVER_UNAVAILABLE。没有重启、重新 initialize 或重新注册。Host 销毁时整个内存 Registry 随实例销毁；每个 ASGI worker 各有一套子进程，首版部署建议单 worker，reload 会重建完整实例。

## 6. SDK 1.26.0 接口与隐式发现

核验源码位于 `mcp_servers/web_page_fetch/.venv/Lib/site-packages/mcp/`；分发版本由 `mcp-1.26.0.dist-info/METADATA` 确认。官方版本源码：[ClientSession](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/src/mcp/client/session.py)、[stdio_client](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/src/mcp/client/stdio/__init__.py)、[BaseSession](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/src/mcp/shared/session.py)、[Windows utilities](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/src/mcp/os/win32/utilities.py)。以下具体行为主要以本机固定版本逐段阅读和探针为证，不依据最新版文档推断。

- `stdio_client(StdioServerParameters(...), errlog=...)` 为异步上下文管理器，yield `(read_stream, write_stream)`；没有公开返回 Process 的接口。
- `ClientSession(read_stream, write_stream, read_timeout_seconds=timedelta(...))`；进入后需显式 `await initialize()`，后者包括发送 initialized 通知。
- `list_tools(*, params=types.PaginatedRequestParams(cursor=...)) -> ListToolsResult`，含 tools/nextCursor。旧 cursor 参数仍可用但已弃用；SDK 不自动遍历全部页。
- 首次 `list_tools()`，之后只在 nextCursor **不是 None** 时继续。cursor 是不透明字符串（包括空字符串），不能用 truthiness 判断结束。重复 cursor、重复工具名、超过 100 页/1000 工具或 15 秒 Server 初始化总预算均拒绝该 Server；页数和工具总量计入全分页过程。
- 每次 list_tools 都将当页 name → outputSchema 加入内部缓存，不清空旧页；完整发现后再调用，包含无 outputSchema 的工具也不触发再发现。
- `call_tool(name, arguments=None, read_timeout_seconds=None, progress_callback=None, *, meta=None) -> CallToolResult`。成功时调用 `_validate_tool_result`；缓存没有 name 时会在 **tools/call 返回之后** 额外 list_tools 一次，不保证完整分页。错误结果 isError=true 不走该校验。
- 有 outputSchema 时 SDK 使用 jsonschema 校验 structuredContent；缺失或不符合时抛 RuntimeError。不能按异常文本前缀解析 SDK 错误。Adapter 将 call_tool 边界的此类 RuntimeError 统一安全映射为 MCP_INVALID_RESULT；其它无明确识别的异常归 MCP_PROTOCOL_ERROR。
- 只用初始化已发现的 remote_name 和同一个 Session；不替换 Session、不直接改私有缓存、不响应 list_changed 触发刷新。内存探针、真实 stdio 都已验证热缓存没有额外 list。

## 7. 有限输入 Schema 与动态 Pydantic 模型

关键约束：Pydantic Model 既要满足 Dispatcher.model_validate，也要满足 Runtime.model_json_schema，且参数 dump 不能扩大或缩小输入 Schema 接受范围。单靠严格类型注解仍有 JSON Schema integer/enum 等细微差异，因此采用“**白名单审核 + 原始 JSON 校验 + 动态字段模型**”。不使用通用 Schema→Python 类型转换器。

首版允许子集：根为 object，properties 为有限个平面属性；属性为 JSON 标量（string、boolean、integer、number、null），type 可为单类型或这些标量的非空无重复数组。属性也可仅用非空 enum 或 const 限制标量。允许 type/enum/const 同时出现并按交集校验。禁止无约束属性 `{}`，额外未知字段由 additionalProperties 决定。

| 关键字 | 首版语义 |
| --- | --- |
| 根 type/properties/required | type 必须 object；properties 省略视为空；required 是唯一属性名数组且每项必须已声明。required 不含某字段只表示可缺失，不表示接受 null |
| additionalProperties | 省略或 true：保留未知 JSON 字段；false：拒绝；Schema 对象形式拒绝注册。根默认值是 true，不能沿用 Pydantic 默认 ignore |
| 属性 type | 对原始 JSON 校验，不将数字字符串转数值，不将 bool 当 integer；integer 接受数学整数如 1.0，不接受 1.5；number 必须有限 |
| enum/const | 使用 JSON Schema 相等语义，true 不等同 1；1 与 1.0 等值；候选值限定 JSON 标量，拒绝非有限数、空 enum 和 JSON 等值重复 enum |
| minLength/maxLength | 非负整数且不为 bool；按 Unicode 字符数，min ≤ max；只允许用于明确限定 string 的属性，避免产生首版不打算支持的条件约束 |
| title/description/default/examples | 作为 JSON 元数据保留；default 不赋值、不插入参数、不使 required 失效 |
| $schema | 可省略或精确为 Draft 2020-12 URI；其它方言拒绝，防止语义漂移 |

根只允许上述适用关键字及元数据；属性只允许 type/enum/const/minLength/maxLength/元数据。所有未列关键字都拒绝，包括 format、pattern、minimum/maximum/multipleOf、$ref/$defs、allOf/anyOf/oneOf/not、嵌套 properties、数组 items、依赖和条件关键字、unevaluatedProperties，以及无法确定为纯元数据的扩展。不得把未知验证约束当作描述忽略。此子集覆盖当前 url:string Schema；若后续 Server 扩展超出子集，以注册失败显式暴露，不扩大实现。

实现约定：

1. 深拷贝 Schema，先执行关键字/结构白名单审核，再用显式 `Draft202012Validator.check_schema` 检查合法性并预建 validator。不启用 format 自动校验。
2. 动态 create_model，字段使用内部安全名与原始属性名 alias（如 f_0 → model_dump），避免 Pydantic 保留名、前导下划线和点号引发字段丢失。只允许 alias 输入，不接受内部字段名作为声明字段。
3. 属性使用 Any 存储，model_validator(mode='wrap') 先对原始 dict 执行严格 JSON 树检查及已审核 Schema 校验，再调用 Pydantic 字段装配。校验失败转安全 ValueError，供 Dispatcher 统一变成 INVALID_ARGUMENTS。装配后按原始输入中是否存在属性 alias 归一化声明字段的 model_fields_set，移除由同名额外键引入的错误“已提供”标记；model_extra 不改动。不是把整个参数塞进单个 RootModel：动态字段仍真实存在，只由 Schema validator 而非 Pydantic 强制转换负责验证语义。
4. JSON 树检查只接收字符串键 dict、list 和 JSON 标量，拒绝 tuple、bytes、自定义实例、NaN、Infinity 等 Python 非 JSON 值；这也覆盖 additionalProperties=true 的任意 JSON 额外值。资源边界为深度 32、20,000 节点、ASCII 转义 JSON 1 MiB；Schema 最多 256 个声明属性，每个 enum 最多 256 项。这些是服务资源限制，超出时显式拒绝，不声称对无限大小 JSON 保持等价。输入、Schema 和结果共用 JSON 树保护。
5. 必填字段默认值为 Pydantic 必填标记；非必填用内部 None 默认占位，validate_default=False。占位值不代表允许显式 null，显式 null 是否有效先由原始 Schema 判定。
6. `extra='allow'` 对应 true/省略，`extra='forbid'` 对应 false。handler 仅使用 `model_dump(mode='python', by_alias=True, exclude_unset=True)`，禁止 exclude_none（会吞掉合法 null），禁止普通全量 dump（会补占位 None）。exclude_unset 必须与第 3 步基于原始 alias 的存在性修正共同使用；单独使用它不能避免额外键 field_0 等与内部名碰撞。Any 字段和额外字段保持原始 JSON 值，再作 JSON 检查。
7. 模型的 JSON Schema hook 返回冻结保存的原始 Schema 的深拷贝，避免自动导出 Any/default=None/internal alias 等改变 LLM 所见定义；每次返回独立快照。不改 Runtime。

原始校验与模型字段装配是必要的最小适配细化；不支持的约束仍在注册阶段拒绝，而非把所有 Schema 交给 jsonschema 就宣称通用支持。Phase 2 已实现并测试 alias/hook、nullable、额外字段与原值保留矩阵，结果见验证报告。

## 8. 调用、结果与错误

每个 Server 首版使用一个 asyncio.Lock 串行化 Session 租约；不同 Server 可并发。SDK 按 request ID 路由响应且本轮两个并发调用成功，但这不等于证明所有 Server/SDK 状态操作均并发安全。锁不跨 Runtime LLM 调用持有。40 秒调用总预算包含等待锁、写请求、等结果和适配；SDK read timeout 单独不能覆盖等待锁/写入，因此外层 asyncio.timeout 必须覆盖完整 handler，且只在其自身过期时映射超时。

Adapter 顺序：验证本地远程绑定 → 预算内取得 Session → call_tool（传 timedelta 预算）→ 优先检查 isError → 检查 structuredContent 是 dict → 检查标准 JSON 可序列化且无非有限数 → 返回字典。忽略 content 中的文本/图片及 _meta，不解析自由文本为成功数据，不返回完整 CallToolResult。SDK 已按 outputSchema 校验的结果也要检查 dict/JSON 边界；注册时预先检查 outputSchema 合法性，不把 outputSchema 转成输入模型。

Web Page Fetch 实际返回固定五字段：title:string（≤500）、url:string、content:string（≤12000）、truncated:boolean、content_type:const text/html，五项必填、额外字段禁止。保持独立 Server 自带 outputSchema，不导入其业务模型到 Host。其它工具也必须返回 structuredContent 字典；无 outputSchema 时仍执行 dict/JSON 检查。

| 条件 | ToolExecutionError.code | LLM 安全消息 |
| --- | --- | --- |
| isError=true | MCP_TOOL_ERROR | 远程工具执行失败 |
| Host 自身截止时间耗尽；SDK McpError.error.code=408 | MCP_TIMEOUT | 远程工具调用超时 |
| 关闭/不可用状态；连接关闭代码；BrokenResourceError/ClosedResourceError/EOF 等已知断连 | MCP_SERVER_UNAVAILABLE | 远程工具服务不可用 |
| JSON-RPC 错误、结果结构解码失败、未分类 SDK/协议异常 | MCP_PROTOCOL_ERROR | 远程工具通信失败 |
| 非字典/缺 structuredContent、非标准 JSON、SDK 结果校验 RuntimeError | MCP_INVALID_RESULT | 远程工具返回结果不符合约定 |

全部 retryable=false，不自动重试。McpError 使用结构化 error.code，408 是 SDK 本地 read timeout 使用的 httpx 状态值，也可能由远端发送；统一按超时分类但不宣称能辨别来源。CONNECTION_CLOSED 使用 SDK 常量，不硬猜数值。协议解码的 ValidationError 与结果 Schema 的 RuntimeError 在 call_tool 边界区分，不按敏感异常文本匹配。

显式重抛 asyncio.CancelledError；ExceptionGroup 先分离/传播取消及进程级 BaseException，再分类普通异常。不能捕获 BaseException 后返回业务错误。超时或取消不保证远端操作已经停止，首版也不补发重试；必须由下一请求与关闭测试证明 Session 可用性，若明确断连才标记 unavailable。

SDK 的 read timeout 实现会移除本地响应等待项，没有自动发送远端取消通知的保证。Host 外层取消同样不能被解释成远端执行撤销成功。

日志仅允许固定事件及 server_id 等安全元数据；不记录参数、原始异常字符串、完整 URL、stderr、traceback 或远端自由文本。stdio_client 默认 errlog=sys.stderr，实际改为 owner 管理的 os.devnull 句柄，避免敏感 stderr 落盘或混入平台日志；safe_logging.SDKLogGuard 按 SDK 来源定向过滤其自由文本，不清空平台 root logger，全部 owner 结束后移除过滤。stdout 仅协议流；MCP 日志通知默认丢弃。初始化状态不新增公开 endpoint。当前仅记录初始化失败固定事件，未额外建设指标系统。

## 9. 生命周期、任务归属与失败隔离

状态：configured → connecting → ready → closing → closed；初始化错误 → failed；运行期断连 → unavailable → closing/closed。registered 与连接状态分别保存，unavailable 不改变 Registry。

每个 Server owner 是显式保存句柄的独立 asyncio.Task，不能把未处理 owner 异常交给同一 AnyIO TaskGroup 取消其他 owner。每个 owner 内：打开 stderr sink → 进入 AsyncExitStack → stdio_client → ClientSession → initialize → 发布 ready → 等 stop → 逆序退出 Session → stdio → 关闭 sink。owner 的普通异常由 Manager 消费并记录状态；ready Future 必须在所有退出路径完成或设置失败，避免调用方无限等待。

Bootstrap 在 ready 后使用 Session 租约进行发现与适配；某 Server 失败时只发送该 owner 的 stop 信号并等待其退出。SDK 上下文的进入/退出始终发生在该 owner 中，不能 `create_task(stack.aclose())`、不能把 stack.pop_all 转交到 lifespan 任务、不能用 wait_for 包裹 enter_async_context（会换任务进入）。不要在已进入的 SDK 取消作用域内部临时套 AnyIO shield 再 aclose，破坏作用域栈的退出顺序。

owner 在资源进入后检查 stop；initialize 请求在短生命周期子任务中执行，owner 等待请求完成或 stop，停止时取消并收集该请求任务。仅请求等待使用子任务，stdio 与 Session 上下文均仍在 owner 内进入/退出。Session 默认 read_timeout_seconds 使用启动预算，Adapter 每次 call_tool 显式覆盖调用预算。Bootstrap 的 15 秒总预算覆盖 connect、initialize、发现、适配；外部等待 ready 超时只通知停止，不把 SDK 上下文退出移到等待者任务。进程创建阶段若 OS/SDK 永不返回，stop 不能主动打断它，归入 cleanup_failed 边界，不能宣称严格墙钟回收保证。

正常关闭：先阻止新租约，有限等待现有 handler（不持有锁等待自身），再 stop 所有 owner 并逐个收集清理结果。FastAPI lifespan 必须在 try/finally 覆盖 bootstrap 及 yield，启动尚未发布状态时也能遍历已登记 owner；取消调用方时用 asyncio.shield 等待保存的 owner 完成，并最终重新抛出原取消。不要把 owner.task.cancel() 作为日常关闭办法。重复取消、卡死或强制退出仍需单独验收，shield 不是进程存活保证。

SDK stdio 正常退出先关闭 stdin，等待 2 秒，再按平台执行强制进程树清理；Session 则先取消自身接收 TaskGroup。异步资源必须逆序退出。SDK Windows 使用 pywin32 Job Object，创建/分配失败会降级；正常路径下的 Job handle 关闭与异常创建过程并非由公开 Process API 完整暴露。不能因代码写了 finally 就声称没有遗留进程/句柄风险，也不能调用 SDK 私有 terminate 函数作为生产依赖。

P1 修复后的异常归属：AsyncExitStack 通过带跟踪的标准异步退出回调管理 Session/stdio，通过同步回调关闭 stderr；所有回调仍在原 owner 内执行。记录 LifecycleDiagnostic(phase, kind, error_type)，不记录远端异常文本或 traceback。

| 来源 | 处理 |
| --- | --- |
| transport/session enter、initialize 代码块 | initialization，Server failed；成功退出其它已进入资源时 cleanup_failed=false |
| Session exit 新增异常 | cleanup，cleanup_failed=true，保留 CleanupError |
| transport exit 重新抛出/包装输入的同一异常对象 | 不当作新增清理失败；按异常叶子的身份区分传播与新增 |
| transport TaskGroup exit 中 SDK 1.26 已知读取/写入通信异常 | UnicodeDecodeError、AnyIO BrokenResourceError/ClosedResourceError/EndOfStream、ConnectionError 叶子记录 communication；仅通信失败且其它退出完成时允许 Server 级隔离 |
| transport exit 其它新增异常，包括混合组里的未知叶子；stderr close 异常 | 保守记录 cleanup，不能因为同组含通信异常而忽略真正退出失败 |
| owner 自身基础设施异常（例如 stderr 无法打开） | owner；与 cleanup 分开，以 OwnerError 使 Host 启动失败；不伪造 cleanup_failed |

退出回调始终重新传播 SDK 异常，保留 AsyncExitStack/TaskGroup 的原有传播和取消语义；owner 最后依据已归属的叶子决定状态，不再以“外层 catch Exception”等同清理失败。初始化失败的通信异常仍不注册该 Server 的工具，清理完成后 Bootstrap 继续其它 Server。

取消与清理失败叠加时，**对外 CancelledError 优先**。finish_cleanup、connect 和 Bootstrap 的 finally 都保留这一优先级；若同时有清理失败，将安全的 CleanupError 链接为 CancelledError.__cause__，其 diagnostics 快照和 Connection.diagnostics 保留清理阶段/错误类型，cleanup_failed=true。没有取消时仍抛 CleanupError。重复取消不取消 owner；不把取消改成业务 ToolFailure，也不静默丢弃清理失败。

非法 UTF-8 读取失败会触发 SDK 自身 TaskGroup/Process 关闭；子进程可能被 SDK 终止为非零退出码。这不等同资源回收失败，也不宣称是自然退出码 0。真实测试分别覆盖未控制读取时序与子进程已以 0 退出、管道仍含真实非法字节的时序；二者都要求进程结束、owner 完成、其它 Server 和本地工具可用。

Phase 2 设 10 秒关闭观测预算：超时记录 cleanup_failed，并继续通知其余 owner 关闭；保留 owner 引用与后台完成观测。该预算是告警/升级边界，不应通过 wait_for 强行取消正在退出 SDK 的任务。若真实卡死测试表明公开 SDK 无法实现可接受的进程回收，应暂停上线并提出最小 SDK/进程监管修正，不在本模块悄悄引入自建 Transport。OS 强杀 Host、解释器崩溃及 job 分配失败下的孙进程回收不能保证。

## 10. Phase 1 依赖兼容性核验（历史记录）

本轮未安装、升级或修改依赖。主环境与 Server 环境 Python 均为 3.12.9、Pydantic 均为 2.13.4。两个环境各自执行 `python -B -m pip check`，均为 `No broken requirements found`。这只是现有环境检查，不等于 MCP 已在主环境安装或整个解析集合已验收。

| 项目 | 主项目实际环境 | MCP 1.26.0 要求 / 结论 |
| --- | --- | --- |
| Python / Pydantic | 3.12.9 / 2.13.4 | ≥3.10 / ≥2.11,<3，满足 |
| AnyIO | 4.14.2 | ≥4.5，满足；OpenAI 要求 ≥4.10,<5，交集存在 |
| FastAPI / Starlette | 0.141.1 / 1.6.0 | SDK 要求 Starlette ≥0.27，FastAPI 元数据要求 ≥0.46，当前满足 |
| Uvicorn | 0.52.4 | SDK ≥0.31.1，满足；不必照搬 Server 的 0.54.0 |
| HTTPX / HTTPcore | 0.28.1 / 1.0.9 | MCP 使用 httpx ≥0.27.1；当前传统栈已存在 |
| HTTPX2 / HTTPcore2 | 2.10.0 / 2.10.0 | OpenAI 3.1.0 使用 httpx2 ≥2.7,<3；与传统栈包名/import 不同，可共存，不替换、不别名化 |
| MCP 本体 | 未安装 | Phase 2 增加 mcp==1.26.0 |
| MCP 直接依赖未安装项 | httpx-sse、jsonschema、pydantic-settings、PyJWT、pywin32、sse-starlette | Phase 2 在隔离环境解析完整传递依赖；不能根据 Server 环境版本直接升级主环境 |

python-multipart 0.0.32、typing-extensions 4.16.0、typing-inspection 0.4.4 均满足当前 MCP 元数据下限。Server 环境 AnyIO 4.16.0 / Starlette 1.10.0 与主环境不相同，本轮 stdio 探针运行在 Server 环境，不能宣称等价验证了主环境的集成。

Phase 2 对 `mcp==1.26.0` 及 Schema Adapter 直接使用的 `jsonschema` 建立显式依赖约束；jsonschema 建议以已验 Server 的 4.26.0 为首个解析候选，检查完整解析结果和 pip check 后再定。无需将 Server 的 httpcore pin 搬入主 requirements；Host stdio 不执行 Fetcher HTTP 逻辑。生产代码 MCP 错误处理独立于现有 httpx2/httpcore2 的 LLM/Weather 重试分类。

## 11. Phase 2 实施依据与验收矩阵

建议文件结构：

```text
backend/mcp_host/
  __init__.py
  config.py          # 配置模型、仓库根路径解析
  manager.py         # owner / session lease / 关闭 / 状态
  schema_adapter.py  # 白名单、原始校验、动态字段模型与 Schema hook
  tool_adapter.py    # async handler / call / 结果与安全错误转换
  bootstrap.py       # 发现分页、候选 Registry、Runtime 注入
backend/config/mcp_servers.json
tests/mcp_host/
  test_config.py
  test_manager.py
  test_schema_adapter.py
  test_tool_adapter.py
  test_bootstrap.py
  test_lifespan.py
  test_stdio_integration.py
  support/           # 分页/坏 Schema/挂起/崩溃测试 Server
docs/testing/module7_mcp_host_validation.md
```

必要现有文件变更仅为 `requirements.txt`（批准后安装所需依赖）与 `backend/main.py`（lifespan 接线）；不改 Runtime/Registry/Dispatcher、旧路由及独立 Server。需要时更新 README 的进度说明。实施顺序：

1. 在隔离环境解析依赖，确认传统与 2 系 HTTP 栈共存、SDK 导入及 Windows 子进程能力；固定候选版本并保存证据，不整体升级。
2. 实现配置与 Manager，先通过 owner 同任务退出、局部失败、启动取消和无遗留进程测试，再写业务适配。明确 ready/stop/锁竞争的状态机。
3. 实现 Schema 白名单与动态模型，测试输入接受范围、字段原值、alias、缺失与 null、LLM Schema hook；使用 jsonschema 作为受支持子集的对照 oracle。
4. 实现工具 Adapter，分别覆盖 SDK 超时、协议错误、断连、isError、无效 structuredContent、取消、日志脱敏，不读取远端自由文本作为错误分类。
5. 实现 Bootstrap 完整分页、缓存保证、冲突和候选 Registry 原子组装，构建并注入现有 Runtime。
6. 加 FastAPI lifespan；启动与关闭测试注入 FakeLLM/测试 DB，避免导入 main 时触碰真实 DB。至少一次测试真实 ASGI lifespan startup/shutdown 消息而非只手动调用构造器。
7. 完成真实 stdio 集成、必要主项目回归与手动生产公网验收，记录精确命令、版本、进程退出、结果和限制。整批实施无需逐文件确认；任何超出本设计范围的结构冲突应先提出最小修正。

分层验收：

- Schema 单元：required/optional × 缺失/null/有效/错误类型；additionalProperties 省略/true/false/对象；enum [1] 不接受 true 而接受 1.0；const；字符串 Unicode 长度上下边界；default 不注入；整数 1.0；NaN/Infinity；不支持关键字逐个拒绝；保留名/别名碰撞；dump 与原 JSON 等价，model_json_schema 返回独立且准确的 Schema。
- Manager 单元/异步集成：阶段性启动失败、锁等待取消、关闭竞态、重复关闭、owner 异常被消费、失败 ready 不悬挂、租约内外状态复查、cleanup_failed 可观测；多 Server 故障互不取消。
- Adapter 单元：固定错误码、敏感 canary 不泄漏、无任何重试、SDK 408/连接关闭常量、取消和含取消异常组；SDK output 校验失败与 Pydantic 解码失败分别映射；非法结果在进入 Dispatcher 结果检查前被转换。
- Bootstrap 集成：分页包括空页但有 nextCursor、空 cursor、重复 cursor、重复工具名、第二页坏 Schema；一个 Server 最后一工具失败仍零注册；与本地/前序 Server 冲突；全部远程失败本地可用；最终组装失败全部清理；list_changed 不刷新；请求阶段 tools/list 计数不增加。
- Runtime/Lifespan：FakeLLM 发出 fetch_web_page 调用，经真实 Dispatcher 返回 ToolSuccess 并继续 Loop；失败生成 ToolFailure；已有本地工具仍可用；启动中断及 shutdown 清理；没有新增 API。回归 tests/tools、tests/agent_runtime 和既有聊天测试，保存实际范围，不能把未跑全仓写成全仓通过。

真实 stdio 端到端方案（不依赖外网的自动测试 + 单独公网手动验收）：

1. Host 测试进程使用集成后的主环境；子进程使用独立 Server 解释器。复用现有 `tests/support/fixture_server.py` 验证中文五字段成功、NETWORK_ERROR、INTERNAL_ERROR、失败后再次成功。
2. 用测试专用 Server 提供两页工具，最后一页调用成功；在协议发送边界仅观察/计数 methods，断言启动分页后所有 tools/call 不新增 tools/list。不要靠直接读私有缓存证明行为。
3. 同时启动两个 Server，使 B 初始化/适配失败并关闭；A 与本地工具继续可用。另做 B 在 ready 后进程崩溃：工具仍在 Registry，调用失败且没有重新启动。
4. 在进程创建、initialize、分页、Runtime 构建、lifespan yield 前后注入中断；记录本次启动的 PID/returncode、EOF/强制退出计数、临时句柄关闭。测试可观察 SDK 私有创建/终止函数，但生产不可依赖它们。加入忽略 stdin EOF 的子进程、孙进程及 Windows Selector/Proactor 运行方式测试。
5. 手动验收生产入口 `python -B -m web_page_fetch`：initialize → 完整发现 → Dispatcher 请求 https://example.com/ → 五字段/非空正文；非法 scheme、localhost、私网返回安全失败；同 Session 再次成功；关闭无遗留进程。生产不放宽 SSRF。记录公网瞬时失败，不能用 FakeFetcher 结果替代生产 Fetcher 证明。

## 12. Phase 1 风险与验证计划（历史记录；实际证据见第 14 节及报告）

| 风险 | 影响与最小控制 | 验收/剩余项 |
| --- | --- | --- |
| SDK TaskGroup 任务归属与作用域顺序 | 跨任务退出可报 cancel-scope 错误；每 Server owner 内 stack，调用任务只租约/stop | 同任务正常退出已实测；异常进入、重复取消仍需 Phase 2 矩阵 |
| Windows 进程/Job handle 清理 | Job 分配降级、FallbackProcess、异常创建或 OS 强杀可能遗留进程/句柄 | 正常 EOF 已实测；强制回收、孙进程、句柄增长和 Uvicorn Selector 模式未测 |
| Session 并发与关闭竞争 | 多个 Runtime 可并发；串行化每 Server 租约，关闭先拒绝新租约 | SDK 两个并发请求已实测成功；锁与在途 shutdown 为待实现测试 |
| Schema 接受范围漂移 | Pydantic coercion、bool/int Literal 等价、占位默认值、extra ignore | 原始 JSON 校验 + Any 存储 + exclude_unset；完整动态模型未实现，不宣称已验收 |
| 隐式工具发现 | 冷缓存 call_tool 成功后额外 list；遗漏后页会破坏初始化发现约定 | 完整 SDK list 分页缓存、同 Session 和固定名称；冷热缓存探针均已验证 |
| 部分可用与原子性 | 局部 register 后无回滚；清理失败可能拖累启动 | 每组候选新建 Registry；局部 owner 关闭实测，真实 Bootstrap 尚未实现 |
| 输出校验及错误泄漏 | SDK RuntimeError/解析日志可能含完整远端内容 | 预检 outputSchema、安全错误边界、stderr/SDK logger 定向过滤；恶意协议 canary 测试待做 |
| 启动/调用预算 | SDK read timeout 只覆盖等待响应，取消不保证远端停止 | Host 全调用预算、启动总预算、停止 owner；慢写/挂起进程待测 |
| 依赖共存 | 传统 HTTPX 与 HTTPX2 混淆、无上界传递依赖漂移 | 当前元数据约束可满足；尚未安装/完整解析目标主环境，Phase 2 第一门槛 |
| 基础设施失败与部署 | create_all 在 import 阶段；多 worker 多子进程；共享 LLM Client 归属 | 保留现有行为，区分基础设施与 Server 失败，首版单 worker；不扩大 DB/LLM 改造 |

## 13. Phase 1 核验记录与实际阅读清单（历史记录）

未安装/升级依赖，未执行生产功能实现，未 Commit/Push。没有运行主项目全量测试；本轮只做针对架构事实的探针。

实际阅读：

- 主仓库：requirements.txt、.gitignore；backend/tools/infrastructure 的 contracts.py、errors.py、registry.py、dispatcher.py；backend/tools/bootstrap.py；backend/agent_runtime/bootstrap.py、runtime.py；backend/main.py。
- 对照测试：tests/tools/test_dispatcher.py、tests/agent_runtime/test_bootstrap.py；搜索了 build_agent_runtime、生命周期入口及 httpx2/httpcore2 使用点；阅读 docs/testing/module6_v1_validation.md 的既有验收边界。
- 独立 Server：README.md、DEPENDENCIES.json、requirements.txt；web_page_fetch/server.py、contracts.py、__main__.py；tests/test_stdio.py、tests/support/fixture_server.py。
- 本地 SDK：mcp/client/session.py、client/stdio/__init__.py、shared/session.py、os/win32/utilities.py、mcp-1.26.0.dist-info/METADATA；主环境 FastAPI/OpenAI/httpx2/httpcore2 的安装元数据；搜索 Starlette routing 的 lifespan 执行位置。最初尝试的 client/stdio/win32.py 不存在，已改读实际 os/win32/utilities.py。

本轮探针均通过 PowerShell here-string 经 `python -B -` 执行，不保留实现脚本，不访问生产凭据；真实子进程仅启动既有 FakeFetcher 测试入口，不发出 HTTP 网络请求。Windows 受限沙箱的 asyncio 探针未能推进，已中止相应 exec 会话；获准沙箱外执行后结果如下：

| 核验 | 实际输出/结论 |
| --- | --- |
| Git | HEAD 与用户提供值一致；初始 status --short 无输出 |
| 两环境 pip check | 各自 No broken requirements found |
| 当前 Server inputSchema | properties.url={title:Url,type:string}，required=[url]，type=object；未声明 additionalProperties，因此允许额外字段 |
| 真实 stdio Session | AsyncExitStack 在 owner 内进入 stdio → Session；initialize、发现、一次调用和两个并发调用成功；方法序列恰为 tools/list、tools/call、tools/call、tools/call |
| 正常关闭 | process_returncodes=[0]，强制终止调用次数=0 |
| 两 Server 局部关闭 | 模拟 Bootstrap 拒绝 B 后，B returncode=0，A 调用成功；不是已实现了正式 Bootstrap |
| 启动取消 | 调用方取消向上传播；finally 通过 stop+shield 等待 owner；最终两个进程 returncode=[0,0] |
| 分页缓存内存探针 | 官方 ClientSession、替换 send_request 返回两页；序列 list(None)、list(p2)、call(second)，没有额外 list |
| 冷缓存内存探针 | 同一官方类新 Session，序列 call(first)、list(None)，确认额外发现发生在 call 之后 |
| 原始 Schema + 动态模型探针 | {} 接受且 dump={}；非必填 s 的 null、整数、空串被拒绝；s=ok 与额外 x=false 原样保留；enum=[1] 拒绝 true，接受且保留 1.0 |

内存 Schema 探针仅覆盖关键机制，没有完整实现白名单/alias/Schema hook。双 Server 测试是资源拓扑验证，不是完整 FastAPI 或正式 Manager 的测试。没有执行本轮生产公网抓取、主环境安装 MCP、强杀/孙进程/句柄泄漏验证；既有独立 Server 公网验收保留在其目录文档中，不冒充本轮结果。

以上为 Phase 1 当时的交付边界；用户随后已授权 Phase 2。

## 14. Phase 2 实际交付与最小修正

- 实现：backend/mcp_host 的 config、manager、schema_adapter、tool_adapter、bootstrap，以及 SDK 来源日志过滤 safe_logging。配置位于 backend/config/mcp_servers.json；main.py 只增加 lifespan 导入/参数，不改变 create_all、路由或旧聊天路径。
- Host 是 async context manager，配置和本地工具基础设施失败向上传播；远程失败按 Server 丢弃候选并关闭；最终调用原有 build_agent_runtime。app.state 保留 mcp_host 供关闭后诊断，shutdown 时撤下 agent_runtime/tool_registry。
- 注册名按用户补充要求使用命名空间；这是对 Phase 1 第 4 节的明确修订，未改变 RegisteredTool 契约。
- initialize 与 stop 竞争等待，修复短调用预算误用于握手的问题；上下文任务归属不变。finish_cleanup 屏蔽调用方重复取消直至清理任务完成，再传播取消。
- Manager 的 SDK message_handler 将畸形协议 Exception 标记为 protocol_failed/unavailable。当前租约退出时报安全协议错误，后续租约报不可用。否则 SDK 默认会跳过该 Exception，允许后面的合法结果继续成功；本修正属于连接健康管理，不执行工具业务。
- Schema 使用显式 jsonschema 4.26.0；不调用 JSON Schema/SDK 私有转换接口。JSON 树与 Schema 数量限制已明确。模型生成、alias、JSON Schema hook 及原值 dump 均有测试。
- os.devnull 取代临时 stderr 文件：同样是非公开受控 sink，减少磁盘敏感数据与临时文件清理面；使用 with 保证句柄退出。
- 关闭使用 10 秒在途租约排空预算，加 10 秒 owner 退出观测预算，两阶段最多约 20 秒；并发关闭不同 Server。超时不强行跨任务退出 SDK，不取消 owner；cleanup_failed 显式报 CleanupError，owner 仍保留引用并自行完成。异常 SDK 退出同样显式报告，继续清理其它连接。
- 静态配置可归属到 Server 的非法配置记录 invalid_config；全局 JSON/重复键/ID 歧义拒绝启动。直接注入 ServerConfig 是测试/受信任代码入口；生产通过 load_config 解析仓库根相对路径。
- SDK 公共接口没有暴露 Process；生产没有依赖其私有创建/终止函数。测试观察这些函数仅用于进程退出与强制回收证据。强制 Job Object 回收与 Selector fallback 已有本机探针，不延伸为所有 Windows 环境保证。

初次 Phase 2 测试结果、依赖新增清单、真实网络证据、全仓既有问题及未覆盖风险见 [Phase 2 验证报告](../testing/module7_mcp_host_validation.md)。源码阅读见 [READING_GUIDE.md](READING_GUIDE.md)。初次实现与后续 P1 修复均已通过最终定点 Review；当前冻结状态以第 16 节为准。

## 15. 两项 P1 定点修复

Schema wrap validator 依据原始 alias 存在性修正 model_fields_set，保留原生 model_dump 契约，不换内部字段前缀、不改 Dispatcher/RegisteredTool。Manager 跟踪进入/退出阶段及新增异常叶子，分开初始化、通信、清理和 owner 基础设施异常；Bootstrap 继续隔离清理完成的通信失败，并明确取消与清理异常的优先级。命名空间、静态配置、无重试/重连策略及独立 Web Page Fetch 均不变。

本轮测试与资源证据见 [P1 修复验证报告](../testing/module7_mcp_host_p1_validation.md)。第 10、12、13 节与初次 Phase 2 验收数字是历史记录，不代表修复后的新增覆盖范围。

## 16. 最终验收与冻结状态

2026-11-03：两项 P1 定点复核未发现新的 P0/P1；以当前代码运行 tests/mcp_host 与 tests/tools，结果为 **150 + 172 = 322 passed，0 failures/errors/skipped**。pip check 与 git diff --check 通过。非法 UTF-8 双 Server 隔离、真实 stdio、Session 生命周期、取消/清理异常组合和句柄观测的最新证据均有效，详见 [最终验收报告](../testing/module7_mcp_host_final_acceptance.md)。

Module 7 正式 **COMPLETE / FROZEN**，已具备 MCP Host 基础架构、连接与 Session 管理、有限 Schema 动态适配、工具适配、Bootstrap、整 Server 原子注册、部分可用、FastAPI lifespan、真实 stdio 及错误/资源清理专项验证。

冻结边界：stdio、静态配置；没有动态管理、自动重连、热更新或平台 MCP Server 能力；旧 /chat 与 /chat/stream 未切换至 AgentRuntime。Stage 3 仍 IN PROGRESS，Module 8 Evaluation / Observability 仅标为 NEXT，不在本次启动。历史全仓问题与 OS/SDK 清理限制继续保留，不扩展为本次专项失败，也不宣称已解决。
