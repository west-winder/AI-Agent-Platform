# Stage 4 / M4-4 · 最小 LangGraph Agent Loop

用一个模拟天气工具观察模型如何提出工具请求，LangGraph 如何回传结果，以及循环何时结束。

```text
START → LLM
        ├─ 有工具请求且 llm_calls < 3 → ToolNode → LLM
        └─ 无工具请求或已达到上限      → Finish → END
```

## 运行

从仓库根目录打开 PowerShell。

```powershell
Set-Location experiments/langgraph_m4_agent_loop
py -3.12 -m venv .venv
$m44Python = '.\.venv\Scripts\python.exe'
& $m44Python -m pip install -r requirements.lock.txt
& $m44Python -X utf8 demo.py
& $m44Python -X utf8 -m pytest -v
& $m44Python -m pip check
```

Demo 和测试使用支持 `bind_tools()` 的 `ScriptedChatModel`，它消费预设回复，并记录实际收到的消息和工具定义。测试关闭 LangSmith 上报，拦截外部连接和域名解析，保留 Windows 数字回环。独立环境使用 Python 3.12.9、LangGraph 1.2.22、LangGraph Prebuilt 1.1.2 和 LangChain Core 1.8.1。

## 一轮执行

`build_agent_graph(model)` 接收支持工具绑定的 `BaseChatModel`，在构造时调用 `bind_tools([get_weather])`。`graph.invoke({"messages": [...]})` 为每次运行创建 State。节点通过闭包访问绑定后的模型，传入系统指令和当前消息，把完整 `AIMessage` 与递增后的 `llm_calls` 一起写回。State 只有 `MessagesState.messages` 和 `llm_calls`、`final_answer`、`stop_reason` 三个新增字段。显式传入历史消息可以继续对话，新一轮仍从零开始计数。 [LangGraph Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)

`ToolNode` 执行 `get_weather`，并以 `tool_call_id` 把 `ToolMessage` 与模型的请求配对。北京、上海返回模拟天气。未收录城市触发 `ValueError`，`ToolNode(handle_tool_errors=ValueError)` 将它转换为 `status="error"` 的消息。下一次 `LLM` 调用会看到这条消息。 [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)

| Finish 路径 | 输出 |
| --- | --- |
| 最后的 `AIMessage` 没有工具请求 | `stop_reason="answered"`。保留模型文本，并列出本轮工具的成功结果与失败信息。回答本身可以说明查询失败。 |
| 第三次模型调用仍提出工具请求 | `stop_reason="llm_limit"`。逐条按原 ID 追加 `status="error"` 的 `ToolMessage`，内容为「因预算限制未执行」。最终答案列出已成功、已失败和未完成的请求。 |

`llm_calls` 是模型调用次数。默认 `recursion_limit=8` 则约束 LangGraph 的 super-step 数，作为拓扑异常时的兜底。测试还验证第三次模型调用直接给出回答时可以正常完成。 [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit)

## 本次实测

| 检查 | 结果 |
| --- | --- |
| M4-4 离线测试 | 7 passed，覆盖工具反馈、调用上限、ID 配对、State 隔离和递归保护。 [pytest 输出](evidence/pytest.txt) |
| Demo | 实际输出正常完成、工具执行失败和预算终止三种场景。 [Demo 输出](evidence/demo.txt) |
| 依赖 | `pip check` 通过，43 个已安装版本与锁文件一致。 [依赖检查](evidence/pip-check.txt) |
| 回归 | M4-2 工具相关 7 passed，M4-3 10 passed、1 skipped（默认关闭真实联调）。 [M4-2](evidence/pytest-regression-m4-2-tools.txt) · [M4-3](evidence/pytest-regression-m4-3.txt) |
