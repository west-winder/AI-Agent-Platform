# Stage 4 / M4-8 · LangGraph 综合集成实验

```text
START → LoadMemory → LLM ── 有工具请求且还有预算 → ToolNode → LLM
                      └── 无工具请求或达到上限 → Finish → ExtractMemory → SaveMemory → END
```

调用方注入 Agent 模型、工具列表、记忆服务和提取模型。`model.bind_tools()` 与 `ToolNode` 使用同一组工具。入口为 `await graph.ainvoke({"user_id": 1, "messages": [HumanMessage(...)]})`，可以附带调用方提供的同用户历史，最后一条输入必须是本轮用户消息。 [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

## 运行

在仓库根目录打开 PowerShell。Server 解释器复用 `mcp_servers/web_page_fetch/.venv`。首次使用时按 Server README 建立该独立环境。 [Server README](../../mcp_servers/web_page_fetch/README.md#安装与运行)

```powershell
$m48Dir = 'experiments/langgraph_m4_integrated_agent'
py -3.12 -m venv "$m48Dir/.venv"
$m48Python = "$m48Dir/.venv/Scripts/python.exe"
& $m48Python -m pip install -r "$m48Dir/requirements.lock.txt"
& $m48Python -B -X utf8 -m experiments.langgraph_m4_integrated_agent.demo
& $m48Python -B -X utf8 -m pytest -v $m48Dir
```

本次实测复用与本目录锁文件一致的 M4-6 客户端解释器。Client 为 Python 3.12.9、LangGraph 1.2.22、LangChain 1.5.2、FastMCP 4.2.5 和 MCP SDK 2.3.0。Server 使用独立的 MCP SDK 1.26.0，stdio 协议协商到 `2025-11-25`。异步测试允许 Windows 所需的数字回环并阻断外部连接。 [依赖证据](evidence/pip-check.txt)

## 工具与状态

| 工具 | 输入与结果 |
| --- | --- |
| `retrieve_knowledge` | 调用 Stage 2 的 `retrieve(query, chunk_store, top_k=5)` 契约。本次注入 Fake Retriever，使用真实 `RagRetrievedChunk` 对象。`ToolMessage` 中保留文本、来源和 Chunk ID。空命中返回 `{"documents": []}`，执行异常向上抛出。 |
| `fetch_web_page` | 用 M4-6 验证过的 Beta `MCPAdapter` 与 Server 已有的 stdio Fixture。连接上下文覆盖整个 `ainvoke()`。成功时保留原 Call ID 和结构化结果，工具失败返回错误消息，连接中断抛出 `MCPError`。 |
| `get_weather` | 本地固定天气，北京晴、上海多云。未知城市生成工具错误。 |

公共 State 在 `MessagesState` 上增加 `user_id`、`memory_context`、`llm_calls`、`final_answer`、`stop_reason`、`memory_saved`。每次 Agent LLM 调用临时拼接一条包含当前记忆的 SystemMessage，State 只记录模型消息与工具结果。提取决策通过内部字段交给保存节点，工具的业务数据留在 `ToolMessage`。 [MCPAdapter 参考](https://reference.langchain.com/python/langchain/mcp/adapter/MCPAdapter)

## 预算与记忆

`llm_calls` 只统计 Agent 模型，最多三次。第三次仍请求工具时，Finish 保留已有结果，按每个原始 `tool_call_id` 追加「因预算限制未执行」记录，并生成确定性的结束说明。独立的 `recursion_limit=12` 限制图的 super-step。 [LangGraph Recursion limit](https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit)

Finish 后的提取模型有自己的一次结构化调用，只读取本轮用户原话。`SaveMemory` 校验保存决策和原文片段，成功写入后才设置 `memory_saved=True`。无候选时跳过写入，写入异常原样传播。每次调用有独立 State，用户记忆通过共享的 Fake 服务按 `user_id` 读写。

## 实测与限制

- **9 passed，1 warning**，警告来自 `MCPAdapter` 的 Beta 导入提示。测试核对了工具回传序列和 ID、两种 MCP 失败、RAG 空检索与异常、预算、用户隔离和写入校验。断连用例复用 M4-6 的测试子进程入口。 [pytest 输出](evidence/pytest.txt)
- Demo 的三条工具结果都由 `ToolNode` 执行，其中 MCP 进行了真实的本地 stdio 往返，网页正文来自 Fixture。RAG 使用预设检索结果，天气使用固定表，记忆存于进程内，两个模型使用脚本回复。 [Demo 输出](evidence/demo.txt)
- `pip check` 通过，实测客户端的 91 个依赖版本与独立锁文件一致。真实数据库、真实模型和公网抓取均在本次验证范围之外。记忆相关性采用简单词项匹配，候选必须是用户原话片段。 [依赖检查](evidence/pip-check.txt)
