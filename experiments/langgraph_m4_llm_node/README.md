# Stage 4 / M4-3 · LangGraph LLM Node

两个实验共用 `START → LLM → END` 拓扑。E1 展示如何把模型回复提取为普通文本并写入 State。E2 展示如何把完整回复追加到对话。两个实验都在图外创建模型，图内同步调用一次。

## 运行

在仓库根目录打开 PowerShell，从独立锁文件创建实验环境。

```powershell
Set-Location experiments/langgraph_m4_llm_node
py -3.12 -m venv .venv
$m43Python = '.\.venv\Scripts\python.exe'
& $m43Python -m pip install -r requirements.lock.txt
& $m43Python -m pip check
& $m43Python -X utf8 demo.py
& $m43Python -X utf8 -m pytest -v
```

本次实测使用 Windows、Python 3.12.9、pip 24.3.1、LangGraph 1.2.22、LangChain Core 1.8.1 和 ChatDeepSeek 1.2.0。`requirements.txt` 列出直接依赖，`requirements.lock.txt` 沿用 M4-2 的 50 个锁定版本。默认测试关闭 LangSmith 上报，通过 socket guard 拦截外部连接和域名解析，允许 Windows asyncio 初始化所需的数字回环地址。

## 从构造到执行

`build_basic_qa_graph(model)` 和 `build_messages_graph(model)` 接收一个已经创建的 `BaseChatModel` 对象。内部定义的 `llm_node(state)` 引用外层的 `model`，因此外层函数结束后，节点仍能访问那个对象。这就是本例中的闭包。

Python 中的 `return llm_node` 会返回函数对象，函数体要等到调用 `llm_node(...)` 时才运行。这里直接把 `llm_node` 传给 `add_node("LLM", llm_node)`，工厂返回 `builder.compile()` 生成的图。调用 `graph.invoke(initial_state)` 后，LangGraph 才沿边调度 `LLM`，把本轮的 State 作为 `state` 参数传入。节点从中取出模型输入，调用 `model.invoke(...)`，并把更新交回 LangGraph。

| 实验 | `graph.invoke()` 的输入 | `model.invoke()` 的输入 | 节点返回 | 最终 State |
| --- | --- | --- | --- | --- |
| E1 Basic QA | `{"question": str}` | 问题字符串 | `{"answer": response.content}` | 原 `question` 和新的 `answer` |
| E2 MessagesState | `{"messages": [HumanMessage(...), ...]}` | 当前完整消息列表 | `{"messages": [response]}` | 合并后的完整消息列表 |

### E1 · 部分 State 更新

`BasicQAState` 用 `TypedDict` 声明 `question` 和 `answer` 都是 `str`，`NotRequired` 表示输入可以暂时没有 `answer`。模型返回 `AIMessage` 后，节点检查 `content` 确实是字符串，再只返回 `answer` 更新。LangGraph 按字段合并，保留原问题，已有 `answer` 时写入新值。节点只需要返回自己负责更新的字段。 [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

### E2 · 消息与 `add_messages`

`AIMessage` 是带有助手角色的消息对象，可以保留 `content`、`id`、`response_metadata` 和 `usage_metadata` 等字段。E1 的 `answer` 只携带文本，E2 把完整的 `AIMessage` 放回 State，便于在后续调用中继续传递角色和历史。

`MessagesState` 已在 `messages` 字段上配置 `add_messages`。节点只提交新回复，LangGraph 根据消息 ID 合并。新 ID 追加到列表末尾，相同 ID 替换原位置的消息。初始 `HumanMessage` 与新 `AIMessage` 使用不同 ID 时，两条都会保留。 [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

### State 隔离与模型共享

同一张编译图连续两次 `invoke()`，每次都用自己的输入初始化 State。要继续上一轮对话，调用方可以把上一次返回的消息列表和本轮问题一起传入。闭包则一直持有同一个模型对象。Demo 让两张图共享一个 `FakeListChatModel`，先后取得两条预设回复，展示模型对象可以在多个调用间保有自己的内部状态。

两张图各只有一个节点，未配置图重试。测试使用 Spy 检查构造阶段的调用次数为零、每次 `invoke()` 只调用一次模型，以及模型异常原样传给调用方。

## 与 Stage 3 的职责对应

| 本实验 | Stage 3 |
| --- | --- |
| 图工厂接收已创建的模型并组装节点 | `build_agent_runtime()` 接收 `LLMPort` 或组装 `DeepSeekAdapter`，再交给 `AgentRuntime` |
| `BaseChatModel.invoke()` 完成一次模型交互，真实测试用 `ChatDeepSeek` | `LLMPort.complete()` 接收 `LLMRequest` 并返回 `LLMResponse`，由 `DeepSeekAdapter` 对接供应商 |
| `graph.invoke()` 调度节点，按 State 字段及 reducer 规则合并结果 | `AgentRuntime.run()` 启动运行，`ContextBuilder` 准备模型输入，Runtime 自己更新历史并处理循环 |

## 真实 DeepSeek Smoke Test

```powershell
& $m43Python -X utf8 -m pytest --deepseek-smoke -m deepseek_smoke -v -s
```

测试复用 M4-2 的配置读取器。进程环境变量优先，仓库根 `.env` 提供后备配置。`DEEPSEEK_API_KEY` 和 `DEFAULT_MODEL` 缺少任一项时跳过，`DEEPSEEK_BASE_URL` 为空时使用 DeepSeek 官方端点。显式启用后，用 E2 发送固定问候提示。测试关闭 SDK 重试和思考模式，设定 20 秒 SDK 请求超时和 64 个生成 Token 的上限。 [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/) · [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)

测试检查合并后的消息包含原始 `HumanMessage` 和一条具备非空文本的 `AIMessage`。成功输出只记录类型和文本长度。请求失败会使测试失败，并记录异常类型和可用的 HTTP 状态码。

## 实测记录 · 2026-11-23

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| M4-3 默认测试 | 10 passed，1 skipped（真实联调开关关闭） | [pytest-offline.txt](evidence/pytest-offline.txt) |
| Demo | E1 输出问题与字符串答案，E2 依序输出 `HumanMessage`、`AIMessage` | [demo.txt](evidence/demo.txt) |
| 独立环境 | `pip check` 通过，`pip freeze` 与 50 项锁定版本一致 | [pip-check.txt](evidence/pip-check.txt) |
| M3 回归 | 12 passed | [pytest-regression-m3.txt](evidence/pytest-regression-m3.txt) |
| M4-2 回归 | 22 passed，4 skipped（真实联调开关关闭） | [pytest-regression-m4-2.txt](evidence/pytest-regression-m4-2.txt) |
| M4-3 显式联调，进程中清空 Key | 1 skipped（缺少 Key） | [pytest-deepseek-no-key.txt](evidence/pytest-deepseek-no-key.txt) |
| M4-3 显式联调，使用现有配置 | 1 passed，保留原始人类消息，助手回复有 6 个字符 | [pytest-deepseek.txt](evidence/pytest-deepseek.txt) |
