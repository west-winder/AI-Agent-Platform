# MCP Host 代码阅读导读

Module 7 当前状态：**COMPLETE / FROZEN**。最终定点 Review 与 322 项专项/Tool 回归通过，见 [最终验收报告](../testing/module7_mcp_host_final_acceptance.md)。Stage 3 仍进行中，Module 8 为 NEXT。

建议先读 [ARCHITECTURE.md](ARCHITECTURE.md) 第 1–9、14 节，再按下列调用链读代码。旧 `/chat` 和 `/chat/stream` 没有切换到 AgentRuntime；本次只将可用 Runtime 注入 app.state，没有新增 API。

## 1. 启动与组装

入口是 [backend/main.py](../../backend/main.py) 的 FastAPI lifespan 参数，对应 [bootstrap.py](../../backend/mcp_host/bootstrap.py) 的 lifespan。

`lifespan → bootstrap_host → load_config → build_tool_registry → manager.connect → discover_tools → build_args_model → adapt_tool → 新 ToolRegistry → build_agent_runtime(registry=...)`。

[config.py](../../backend/mcp_host/config.py) 读取 [静态配置](../../backend/config/mcp_servers.json)。相对 command/cwd 以仓库根解析，独立 Server 使用自己的解释器。显式 env 交给 SDK 的白名单环境合并逻辑，主项目 .env 不整体转发。

Bootstrap 依次处理各 Server。完整发现和适配都成功后才把整组候选加入一个新的 Registry；任何一项失败就丢弃新实例并关闭该 Server。先前成功的候选和本地 get_weather 保留。最后组装 Runtime 并发布 app.state；调用方不能看到注册到一半的实例。

`Host.statuses` 记录初始化结果；运行期健康状态查看 `host.manager.connections[server_id].state`，两者不是同一个生命周期。

## 2. Session 生命周期

[manager.py](../../backend/mcp_host/manager.py) 的 Connection 保存 ready Future、stop Event、单 Server 锁、owner/closing task 和状态。

每个 Server 有独立 owner：

```text
owner task
  AsyncExitStack
    open stderr sink，登记 close 回调
    enter stdio_client，登记带跟踪的 exit 回调
    enter ClientSession，登记带跟踪的 exit 回调
    initialize 请求与 stop 竞争
    ready
    等待 stop
    exit ClientSession
    exit stdio_client
    close sink
```

initialize 请求可以运行于子任务；SDK 上下文的进入/退出始终在 owner 中。借用方仅通过 `async with manager.session(id)` 访问 Session，不负责退出。锁前后都检查可用状态，单 Server 串行，不同 Server 可以并发。

关闭先拒绝新租约，再限时排空在途租约，发送 stop，等待原 owner 清理。`finish_cleanup` 使用 shield 保留清理任务，重复取消不会把 SDK 退出转移到另一个任务。超时保留 cleanup_failed 并报错，不把无法证明已回收的进程写成成功。

`_enter_resource` 跟踪退出回调的输入异常与新增异常。重抛代码块异常不等于清理失败；SDK stdio TaskGroup 中已知的解码/管道通信错误也独立归类。Session/transport/stderr 的真实新增退出错误保留 cleanup_failed 与 CleanupError；混合异常组逐叶处理，不能整体忽略。Connection.diagnostics 只记录阶段、类别和错误类型。未知退出错误仍保守报清理失败，owner 自身基础设施错误以 OwnerError 区分。

取消与清理错误同时发生时，外层仍收到 CancelledError；其 __cause__ 指向带 diagnostics 的安全 CleanupError，连接也保留 cleanup_failed。没有取消时直接抛 CleanupError。SDK 因通信任务失败终止子进程的非零退出码不能单独作为清理失败判据，必须与进程/owner 是否完成及退出阶段诊断一起看。

Manager 只管理连接和协议健康，没有 tools/list、Schema 转换或 tools/call 的业务编排。

## 3. 参数模型为何使用 Any

[schema_adapter.py](../../backend/mcp_host/schema_adapter.py) 先审核有限关键字，再构建 Draft202012Validator。动态字段使用安全内部名和原始属性 alias，字段值用 Any 保存。

`model_validate(raw_dict)` 的 wrap validator 先验证标准 JSON 树和原始 Schema，再装配字段，因而不会发生数字字符串转数字、bool 变 integer、或 enum 的 Python 等值误判。Any 只负责保留已经校验的值，不意味着跳过校验。

装配后按原始输入键是否包含属性 alias 修正声明字段的 model_fields_set。Pydantic 会把额外键也放进该集合，所以仅用 exclude_unset 仍可能把与 field_0 等内部名碰撞的额外键误判成声明字段已提供。修正存在性标记后，额外值继续留在 model_extra，不换前缀、不删除合法额外参数，也不新增输入默认值。

非必填字段的内部 None 只是占位；显式 null 仍按原始 Schema 判断。调用时必须使用：

```python
model.model_dump(mode="python", by_alias=True, exclude_unset=True)
```

原始 alias 的存在性修正与该 dump 组合使用，保证缺失字段仍缺失，合法 null 仍保留，额外属性在 additionalProperties 允许时原样转发，default 元数据不会注入参数。JSON Schema hook 向 Runtime 返回原始 Schema 深拷贝；内部 Any/None 不会改变 LLM 所见契约。

## 4. 工具调用与错误

[tool_adapter.py](../../backend/mcp_host/tool_adapter.py) 的 ToolBinding 分开保存 server_id 和 remote_tool_name。

本地：`mcp_web_page_fetch__fetch_web_page`。远程：`fetch_web_page`。

`AgentRuntime → 原有 ToolDispatcher → 动态模型校验 → async handler → Manager 租约 → session.call_tool(remote_name, payload)`。

Handler 的总预算包含锁等待、写入、等待响应和适配。成功仅使用 structuredContent 字典；文本、图片和自由文本错误不转成业务成功结果。SDK outputSchema 校验之外，还检查 JSON 树和标准序列化范围。

错误通过已有 ToolExecutionError 转成 Dispatcher 的 ToolFailure；远端消息、traceback 和参数不进入 LLM 错误信息。MCP 工具不自动重试。调用取消继续向上传播，不宣称远端执行必然停止。

## 5. 为什么调用时不会再次发现

Bootstrap 对同一 Session 调用 SDK list_tools 并遍历所有 nextCursor；SDK 每页缓存工具 outputSchema，包括 None。Adapter 只能绑定已发现的远程名称，不替换 Session，也不更新私有缓存。

真实两页测试 Server 的成功结果会返回自己的 list 计数；多次调用最后一页工具后计数仍为 2，证明没有遗漏后页或调用阶段隐式再发现。list_changed 通知不会触发刷新。

## 6. 推荐测试阅读顺序

1. [Schema 对照矩阵](../../tests/mcp_host/test_schema_adapter.py)：以原始 jsonschema 为语义对照。
2. [Manager 测试](../../tests/mcp_host/test_manager.py)：任务归属、串行租约、取消和清理失败。
3. [工具适配测试](../../tests/mcp_host/test_tool_adapter.py)：远程名称、固定错误、不重试和取消。
4. [Bootstrap 测试](../../tests/mcp_host/test_bootstrap.py)：分页、命名空间、冲突、整 Server 原子性。
5. [Lifespan 测试](../../tests/mcp_host/test_lifespan.py)：真实 ASGI startup/shutdown 消息、部分可用与启动中断。
6. [真实 stdio 测试](../../tests/mcp_host/test_stdio_integration.py)：FakeLLM 全链路、崩溃、畸形协议、强制回收、孙进程及句柄观测。
7. [手动公网验收](../../tests/mcp_host/manual_web_fetch.py)：使用生产 Fetcher，单独执行；不属于自动 pytest。

测试中使用 SDK 私有创建/终止方法仅为观察。生产代码无需知道 SDK Process 对象的结构。完整命令和已知限制见 [验证报告](../testing/module7_mcp_host_validation.md)。

两项 P1 的新增碰撞矩阵、非法 UTF-8 双 Server 与取消/清理组合证据见 [P1 修复验证报告](../testing/module7_mcp_host_p1_validation.md)。
