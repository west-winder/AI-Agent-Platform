# Stage 4 / M4-7 · LangGraph Memory / Context Integration

`START → LoadMemory → Answer → ExtractMemory → SaveMemory → END`

实验沿用 Stage 0 中读取、提取与写入分工，使用独立的内存服务和两个注入的模型。调用方用 `await graph.ainvoke({"user_id": 1, "query": "…"})` 启动流程。每次调用各自初始化 State，跨调用可见的记忆由共享的 `FakeMemoryService` 保存。 [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

## 运行

在仓库根目录打开 PowerShell。

```powershell
Set-Location experiments/langgraph_m4_memory_integration
py -3.12 -m venv .venv
$m47Python = '.\.venv\Scripts\python.exe'
& $m47Python -m pip install -r requirements.lock.txt
& $m47Python -B -X utf8 demo.py
& $m47Python -B -X utf8 -m pytest -v
& $m47Python -m pip check
```

实测为 Windows、Python 3.12.9、LangGraph 1.2.22、LangChain Core 1.8.1。测试禁用 LangSmith 上报，阻断外部连接和域名解析，保留 Windows asyncio 所需的数字回环地址。

## 节点与契约

| 节点 | 输入与输出 |
| --- | --- |
| `LoadMemory` | 按 `user_id` 取该用户数据，再按 Query 的词项筛选，返回 `memory_context`。找不到匹配时返回空字符串。 |
| `Answer` | 将 `query` 和 `memory_context` 放入 JSON 的独立字段，交给回答模型，返回纯文本 `answer`。 |
| `ExtractMemory` | 只把当前 `query` 和提取指令交给提取模型。`with_structured_output(MemoryDecision)` 将模拟模型的工具调用解析为严格的 Pydantic 对象，字段为 `should_save: bool` 和 `content: str \| None`。 |
| `SaveMemory` | `should_save=False` 时返回 `memory_saved=False`。保存决策要求非空内容，且去除首尾空白后须出现在当前用户原话中。等待服务写入成功后才返回 `True`，校验或写入异常传给调用方。 |

保留原话片段是实验采用的保守规则，便于验证内容出处。长期价值、引用归属等语义判断交由提取模型的 Prompt 约束。Fake 模型使用可控回复，Fake 服务的相关性规则只是简单词项交集，存储生命周期与服务实例一致。 [LangChain Structured Output](https://docs.langchain.com/oss/python/langchain/models#structured-output)

## 实测

- **9 passed**。覆盖上下文注入、结构化决策、长期偏好写入、闲聊跳过、用户与 State 隔离、写入异常，以及空白、缺失和非原文候选的拒绝。 [pytest 输出](evidence/pytest.txt)
- Demo 通过同一张图运行三次。用户 1 保存一条偏好，用户 2 只读取自己的记忆，用户 1 再次调用时读到刚保存的偏好。 [Demo 输出](evidence/demo.txt)
- `pip check` 通过，43 个已安装版本与锁文件一致。 [依赖检查](evidence/pip-check.txt)
