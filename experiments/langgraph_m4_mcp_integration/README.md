# Stage 4 / M4-6 · LangGraph MCP Integration

`START → LLM → ToolNode → LLM`，当 LLM 不再请求工具时走 `LLM → END`。`MCPAdapter.list_tools()` 在图外发现并适配工具，同一组 `BaseTool` 传给 `model.bind_tools()` 和 `ToolNode`。图在 Adapter 的连接上下文内用 `ainvoke()` 运行。`MessagesState` 按消息合并更新，`recursion_limit=8` 限制循环的 super-step 数。 [MCPAdapter 参考](https://reference.langchain.com/python/langchain/mcp/adapter/MCPAdapter) · [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit)

## 运行

从仓库根目录使用 PowerShell。已有 Web Page Fetch 独立环境时，复用它的解释器；首次使用按 Server README 建立该环境。 [Server README](../../mcp_servers/web_page_fetch/README.md#安装与运行)

```powershell
Set-Location experiments/langgraph_m4_mcp_integration
py -3.12 -m venv .venv
$m46Python = '.\.venv\Scripts\python.exe'
& $m46Python -m pip install -r requirements.lock.txt
& $m46Python -B -X utf8 -u demo.py
& $m46Python -B -X utf8 -m pytest -v
```

`stdio_adapter.py` 用 Server 自己的 Python 启动子进程，默认入口为 `-m web_page_fetch`。Demo 显式选择仓库已有的 `tests/support/fixture_server.py`。Fixture 使用预设网页数据和生产工具工厂，经真实 stdio 执行 `tools/list` 和 `tools/call`。Windows 测试需要能够建立 asyncio 内部的本地回环，本次在支持回环的 Windows 进程中运行。MCP 消息走 stdio 管道，客户端测试拦截外部连接和域名解析。

## 结果与错误边界

| 场景 | 实测行为 |
| --- | --- |
| 生产入口的工具发现 | 协商协议 `2025-11-25`，发现 `fetch_web_page`，描述与 Server 一致，Schema 要求 `url: string`。 |
| Fixture 成功 | 结构化的五个字段进入 `ToolMessage.artifact["structured_content"]`。模型可见的正文在 `ToolMessage.content`，Fake 模型用它生成最终回答。 |
| Fixture 工具失败 | Server 返回 `isError=true`，Beta Adapter 保留原调用 ID 并生成 `status="error"` 的消息。Fake 模型收到 `NETWORK_ERROR` 后明确报告抓取失败。 |
| 调用中断连 | 测试子进程在工具执行中关闭 stdio。`MCPError`（`CONNECTION_CLOSED`）传给 `graph.ainvoke()` 的调用方，模型未收到工具结果。 |

`ToolNode` 设置 `handle_tool_errors=False`，工具错误结果由已验证版本的 Adapter 自己转换。Server 的 stderr 继续使用原日志白名单。失败用例验证了固定错误代码，以及原始异常 canary 未进入工具结果或 stderr。 [LangChain 1.5.2 工具转换源码](https://github.com/langchain-ai/langchain/blob/langchain%3D%3D1.5.2/libs/langchain_v1/langchain/mcp/tools.py)

## 依赖与验证

| 位置 | 实测版本 |
| --- | --- |
| M4-6 Client | Python 3.12.9、LangChain 1.5.2、LangChain Core 1.8.1、LangGraph 1.2.22、FastMCP 4.2.5、官方 MCP SDK 2.3.0、mcp-types 2.3.0 |
| Web Page Fetch Server | Python 3.12.9、官方 MCP SDK 1.26.0 |

两个环境在官方 SDK 主版本不同的情况下成功协商并完成 stdio 往返。实际导入 `langchain.mcp.MCPAdapter` 产生预期的 `LangChainBetaWarning`，因此锁定了本次验证过的版本。联调范围为生产入口发现和离线 Fixture 工具执行。 [实际 API 签名](evidence/client-api.txt) · [版本与 pip check](evidence/pip-check.txt)

- M4-6 离线集成 **4 passed，1 warning**。 [pytest 输出](evidence/pytest.txt)
- Server 原有 stdio 回归 **2 passed**，覆盖安全拒绝、脱敏日志和正常退出。 [Server 回归输出](evidence/pytest-server-stdio.txt)
- Demo 展示工具发现、结构化成功回包、工具错误和最终回答。 [Demo 输出](evidence/demo.txt)
