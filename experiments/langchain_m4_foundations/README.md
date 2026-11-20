# Stage 4 / M4-2 · LangChain Foundations

六个可独立运行的实验。从 Prompt、Chat Model 和 Runnable 入手，经由工具调用与执行，最后把模型输出交给 Pydantic 校验。全部使用实验目录中的固定输入与模拟数据，真实模型由独立 Smoke Test 启用。

## 环境与运行

在仓库根目录打开 PowerShell，然后执行以下命令。安装从实验的完整锁文件读取版本。已有实验虚拟环境时，进入该目录、设置 `$m4Python` 后即可运行下方的脚本。

```powershell
Set-Location experiments/langchain_m4_foundations
py -3.12 -m venv .venv
$m4Python = '.\.venv\Scripts\python.exe'
& $m4Python -m pip install -r requirements.lock.txt
& $m4Python -m pip check
```

实测环境为 Windows x64、Python **3.12.9** 和 pip **24.3.1**。核心依赖为 `langchain-core` **1.8.1**、`langchain-deepseek` **1.2.0**、`langchain-openai` **1.8.1**、`langgraph` **1.2.22**、`langgraph-prebuilt` **1.1.2**、Pydantic **2.14.3**、pytest **9.2.0**。完整版本见 [requirements.lock.txt](requirements.lock.txt)。

```powershell
& $m4Python -X utf8 e1_prompt_model_parser.py
& $m4Python -X utf8 e2_runnable_lcel.py
& $m4Python -X utf8 e3_async_streaming.py
& $m4Python -X utf8 e4_tool_calling.py
& $m4Python -X utf8 e5_tool_node.py
& $m4Python -X utf8 e6_structured_output.py
& $m4Python -X utf8 -m pytest -v
```

默认 pytest 实际运行 E1–E6 的离线断言。测试 fixture 关闭当前上下文的 LangSmith 上报，拦截外部连接和 DNS 查询，允许数字回环地址供 Windows 事件循环初始化。

## E1 · ChatPromptTemplate、Chat Model 与 StrOutputParser

数据流为 `dict[str, str] → ChatPromptValue → AIMessage → str`。`ChatPromptTemplate.from_messages()` 保留 System 和 Human 角色并填充 `language`、`topic`。`ChatPromptValue.to_messages()` 展开消息以便检查。运行时缺少 `topic` 会抛出 `KeyError`。

`prompt | model | StrOutputParser()` 把前一步的结果作为下一步的输入。固定响应是 `LCEL connects runnable components.`。本次锁定的 Core 1.8.1 中，`type(result).__name__` 是 `TextAccessor`，表示具体实现类型。`isinstance(result, str)` 是 `True`，表示它符合字符串类型契约。接口确实要求内置 `str` 时，可以用 `str(result)`。

## E2 · Runnable、RunnableLambda 与 LCEL

`Runnable` 为组件提供 `invoke()` 等统一调用方法。`RunnableLambda` 把普通函数包装为 Runnable。`|` 将两个组件组装成 `RunnableSequence`，按声明顺序执行。

`split_words` 接收 `str`，返回 `list[str]`。这个列表就是 `summarize_words` 的参数，后者返回 `dict`。示例的中间结果是 `['Prompt', 'Model', 'Parser']`，最终输出为 `{'count': 3, 'joined': 'Prompt/Model/Parser'}`。测试还检查空白输入及逐个调用 `sequence.steps` 的结果。

## E3 · 异步与流式 Chat Model

| API | 如何消费 | 示例中的结果 |
| --- | --- | --- |
| `await model.ainvoke(question)` | 等待一次完整响应 | `AIMessage`，内容为 `Compose and run.` |
| `model.stream(question)` | 用普通 `for` 逐块读取 | 每次得到 `AIMessageChunk` |
| `model.astream(question)` | 用 `async for` 逐块读取 | 异步得到 `AIMessageChunk` |

固定模型逐字符产生 Chunk。把非空 Chunk 序列用 `+` 合并后，对象类型仍是 `AIMessageChunk`，其文本与完整响应一致。真实流的块大小由服务决定，某些块只携带元数据。Smoke Test 实际消费异步 Chunk 并检查合并文本。

## E4 · Tool 定义与调用提议

`@tool` 通过函数签名生成参数 Schema，并使用 docstring 作为描述。`get_weather.name`、`.description` 和 `.get_input_schema().model_json_schema()` 分别展示工具元数据。直接 `get_weather.invoke({"city": "Beijing"})` 得到固定结果 `sunny`。

`ChatDeepSeek.bind_tools([get_weather])` 把工具定义放进模型请求。模型的 `AIMessage.tool_calls` 包含 `name`、`args` 和 `id`，示例提出 `get_weather(city="Beijing")`，调用 ID 为 `weather-1`。离线测试在 HTTP 边界检查发出的工具定义和收到的调用结构，同时断言 Handler 调用次数为零。

## E5 · ToolNode 执行与 ToolMessage

向 `ToolNode([get_weather])` 传入 `{"messages": [AIMessage(...)]}`。独立调用时显式传入 `runtime=Runtime()`，对应图执行时由 LangGraph 注入的参数。返回字典的 `messages` 含本次工具结果。北京与上海分别返回 `sunny` 和 `cloudy`，并用各自的 `tool_call_id` 和输入配对。

| 场景 | 实测结果 |
| --- | --- |
| 模型请求缺少必填的 `city` | 默认生成同 ID 的 `ToolMessage(status="error")`，内容提示 `Field required` |
| 参数形状合法，工具收到未知城市 | 进入 Handler 后触发 `ValueError`，默认向调用者传播 |
| 未知城市，且显式配置 `handle_tool_errors=ValueError` | 该异常转成同 ID 的 `ToolMessage(status="error")` |

显式设置 `handle_tool_errors` 会替换默认异常处理策略。`tool_call_id` 表示请求与结果的关联，成功状态由实际工具执行决定。

## E6 · Structured Output 与 Pydantic

`PaperDecision` 定义 `paper_title: str`、`needs_review: bool`、`reason: str`。`model_json_schema()` 查看要求，`model_validate()` 校验本地对象，字段通过属性读取。实验使用严格类型，字符串 `"yes"` 会在布尔字段触发 `ValidationError`。

`with_structured_output(PaperDecision, method="json_mode", include_raw=True)` 让 ChatDeepSeek 在请求中设置 `response_format={"type": "json_object"}`。Prompt 同时包含 JSON 格式说明和示例。服务端按 JSON Object 模式生成内容，客户端解析后再按 Pydantic Schema 校验。即使内容能被 JSON 解析，字段仍可能校验失败。 [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/)

| 结果键 | 解析成功 | 格式或 Schema 解析失败 |
| --- | --- | --- |
| `raw` | `AIMessage` | 保留原 `AIMessage` |
| `parsed` | `PaperDecision` | `None` |
| `parsing_error` | `None` | 实测为 `OutputParserException` |

适配器 1.2.0 的离线契约测试确认 `function_calling` 通过工具 Schema 请求结构化字段，`json_schema` 在该适配器上也走同一工具路径。本次 `deepseek-flash` 的真实结构化输出验证使用 `json_mode`。本次另有真实天气 Function Calling 验证，结构化 Schema 的 Function Calling 路径仅做了离线验证。 [ChatDeepSeek 版本源码](https://github.com/langchain-ai/langchain/blob/langchain-deepseek%3D%3D1.2.0/libs/partners/deepseek/langchain_deepseek/chat_models.py)

## 与 Stage 3 的职责对照

| 这里的组件 | Stage 3 中对应的学习概念 |
| --- | --- |
| `ChatPromptTemplate` 和 Messages | `ContextBuilder` 负责构造模型可见消息。Stage 3 的 ContextBuilder 还处理历史快照和上下文预算。 |
| `BaseChatModel`、`ChatDeepSeek` | `LLMPort` 与 `DeepSeekAdapter` 的模型接口和供应商转换。 |
| `RunnableSequence` | 在局部先后步骤之间传递数据。Stage 3 的完整编排由 `AgentRuntime` 管理。 |
| `@tool` | 定义工具 Schema 和 Handler，对应注册所需的 Tool Contract。 |
| `bind_tools()` 和 `AIMessage.tool_calls` | 向模型传递工具定义，再获得待分发的调用请求。 |
| `ToolNode` 和 `ToolMessage` | 类似 Dispatcher 的选择、执行和结果封装，使用调用 ID 关联请求。 |
| Agent Loop | 组织模型、工具和历史的多轮交互，决定何时继续或结束。本实验在工具执行结果处结束。 |

## 真实 DeepSeek Smoke Test

```powershell
& $m4Python -X utf8 -m pytest --deepseek-smoke -m deepseek_smoke -v -s
```

配置优先从进程读取，再从仓库根 `.env` 只读补充。`DEFAULT_MODEL` 选模型，`DEEPSEEK_BASE_URL` 指定端点，`DEEPSEEK_API_KEY` 用于认证。缺 Key 或模型时对应测试为 SKIP。未启用参数时四项也为 SKIP。启用后，请求失败、Tool Call 缺失或字段错误、结构化解析失败都会显示为 FAIL。

四项测试各一次请求，关闭 SDK 重试，单次超时 20 秒，每次最多 256 个生成 Token。E3 另外限定整个流消费最多 25 秒。Chat Completions 请求使用 DeepSeek 的非思考模式配置。测试只检查响应形状、非空文本和指定参数，证据保留类型、计数、状态或脱敏错误类别。 [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)

## 实测记录 · 2026-11-19

- **离线 PASS**。22 passed，4 skipped。四项跳过来自默认关闭的联网开关。真实执行了 `ainvoke()`、`stream()`、`astream()` 及 Chunk 汇总。 [完整 pytest 输出](evidence/pytest-offline.txt)
- **DeepSeek PASS**。4 passed，0 failed，0 skipped。E3 收到 9 个 Chunk，E4 返回正确的天气调用，E6 返回 `AIMessage` 和 `PaperDecision`，且 `parsing_error=None`。 [脱敏联调输出](evidence/pytest-deepseek.txt)
- **依赖 PASS**。独立环境 `pip check` 通过。新建第二个环境从锁定 wheel 按 `--no-index` 安装，50 个版本与锁文件一致，并再次通过全部 22 项离线测试。 [pip check](evidence/pip-check.txt) · [锁文件复现](evidence/lock-repro.txt)
- **M3 回归 PASS**。在 M3 自己的虚拟环境重跑原有测试，12 passed。
- **Windows 回环差异**。在 Codex 受管 Shell 中，纯标准库的 `asyncio.run(asyncio.sleep(0))` 在 `socket._fallback_socketpair()` 内的 `accept()` 处阻塞，4 秒诊断计时器结束进程，探针未加载 pytest 或 LangChain。同一 Python 在支持本地回环的 Windows 进程中立即完成。E3 测试和示例在该环境通过，离线测试拦截外部连接的断言也通过。复现受管 Shell 阻塞时，可在支持回环的普通 PowerShell 中运行 E3。
- **未验证**。真实结构化 Function Calling、DeepSeek Beta strict 和其他模型或网关。当前真实结论对应仓库配置中的 `deepseek-flash` 和 Chat Completions。

## 阅读顺序

先看 E1 的 `PROMPT` 与 `explain()`。接着看 E2 的 `build_sequence()` 和 E3 的 Chunk 消费。然后连读 E4 的 `request_weather()` 与 E5 的 `execute_calls()`，对照请求 ID 如何进入结果。最后看 E6 的 `PaperDecision` 与 `review_paper()`，并阅读 [E6 错误测试](tests/offline/test_e6_structured_output.py)。六个脚本的实际输出见 [示例记录](evidence/demos.txt)。
