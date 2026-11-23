# Stage 4 / M4-5 · LangGraph RAG Integration

本实验用 `START → Retrieve → LLM → END` 验证 Stage 2 的公开检索契约如何进入 LangGraph。Demo 注入 Fake Retriever 和 `FakeListChatModel`，直接使用仓库里的 `Chunk` 与 `RagRetrievedChunk` 创建测试数据。 [LangChain Retrieval](https://docs.langchain.com/oss/python/deepagents/retrieval#2-step-rag)

## 运行

在仓库根目录打开 PowerShell。

```powershell
$m45Dir = 'experiments/langgraph_m4_rag_integration'
py -3.12 -m venv "$m45Dir/.venv"
$m45Python = "$m45Dir/.venv/Scripts/python.exe"
& $m45Python -m pip install -r "$m45Dir/requirements.lock.txt"
& $m45Python -X utf8 -m experiments.langgraph_m4_rag_integration.demo
& $m45Python -X utf8 -m pytest -v $m45Dir
& $m45Python -m pip check
```

实测环境为 Windows、Python 3.12.9、LangGraph 1.2.22、LangChain Core 1.8.1。测试禁止网络连接和域名查询，关闭 LangSmith 上报。

## State 与节点职责

`build_rag_graph(retriever, chunk_store, model, top_k=5)` 将三个依赖交给节点闭包。图只依赖 Retriever 的公开 `retrieve()` 方法和 Stage 2 的数据契约。调用方使用 `graph.invoke({"query": ...})`，每次调用各自初始化普通 `TypedDict` State。

- `Retrieve` 把 `query`、原 `chunk_store` 和 `top_k` 传给 Retriever。轻量 Wrapper 将每个 `RagRetrievedChunk` 映射为 `RetrievedDoc`，复制 `chunk_id`、`document_id`、`content`、`source`，将 `retrieval_rank` 映射为 `rank`，保留原列表顺序。
- `LLM` 将 `query` 和 `retrieved_docs` 编成带独立字段的 JSON，作为 HumanMessage 传给模型。SystemMessage 要求知识库证据只能来自检索片段正文，证据不足时明确说明。节点只回写 `answer`，LangGraph 保留其余字段。 [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api#default-reducer)

Stage 2 的 `ContextAssembler` 同样按检索顺序格式化片段。本实验直接格式化图内的 `RetrievedDoc`，使 State 中的排名、来源和正文都能在模型输入里核对。离线测试核对 Prompt 契约，真实模型的回答是否有充分证据仍需答案级验证。

| 场景 | 图的行为 |
| --- | --- |
| A 正常检索 | 保留检索结果，调用模型一次，返回纯文本答案。 |
| B 空检索 | 保留 `retrieved_docs=[]`，LLM 节点直接返回「知识库未检索到与问题相关的内容。」，模型调用次数为零。 |
| C 检索失败 | Retriever 的异常向调用方传播，模型调用次数为零。 |

## 实测结果

- **4 passed**。覆盖固定拓扑、参数与字段映射、顺序、模型输入、三种场景和 State 隔离。 [pytest 输出](evidence/pytest.txt)
- Demo 输出了备份保留期限、空检索说明和原样传播的检索异常。 [Demo 输出](evidence/demo.txt)
- `pip check` 通过，43 个已安装版本与锁文件一致。 [依赖检查](evidence/pip-check.txt)
