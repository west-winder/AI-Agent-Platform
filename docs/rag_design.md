# RAG 设计文档

本文记录 Stage 2 Unified RAG（检索增强生成）的当前实现。核心能力与 Phase 7 Finalization 已完成，Stage 2 正式冻结。范围是后端文档摄入、检索、上下文组装与生成；ChatService、Agent Loop 和前端接线尚未完成。

## 1. Architecture Overview（总体架构）

### Ingestion Pipeline（文档摄入）

```text
Document → Parsing → Chunking → Embedding → PostgreSQL / pgvector Persistence
```

实际入口为 `FileIngestionService.ingest_file(file_name, file_bytes, chunk_size)`：校验文件输入、选择 Loader（加载器）、分配文件名和 `document_id`，再解析为 `Document` 元数据与原始文本。支持 TXT、Markdown 和 PDF；Markdown 保留原文标记，PDF 使用 PyMuPDF 提取文本，未实现 OCR（图像文字识别）。

`RAGIngestionService.ingest(document, raw_text, chunk_size)` 复用 `chunk_document` 按固定字符长度切块，通过 `build_chunk_vectors` 批量生成语料向量，校验向量与 chunk 的一致性、模型身份和维度。全部数据准备并校验通过后，在同一事务中写入 Document、Chunk 和 Chunk Embedding；成功提交，持久化异常回滚并抛出。向量由 pgvector 保存，以 `chunk_id` 与 `embedding_model` 为联合主键。

空白文本切块后返回 `status="empty"`、`chunk_count=0`，不写数据库；解析失败不等同于空文档。PDF 无可提取文本、加密或疑似存在未提取视觉内容时有显式错误。

### Retrieval & Generation Pipeline（检索与生成）

```text
Query → Query Embedding → Dense + BM25 → RRF → Reranker
      → RetrievedCandidate → Hydrator → RagRetrievedChunk
      → ContextAssembler → Generation → RAGAnswer
```

`RAGRetriever` 是唯一检索流程编排器。当前先执行查询向量化，再调用 Dense，随后调用 BM25；两路是顺序调用，不是并行执行。Dense 检索器在构造时绑定向量语料或模型身份；BM25 的文本从调用方传入的 `chunk_store` 派生。

两路分别获取最多 `top_k` 个候选，RRF（Reciprocal Rank Fusion，倒数排名融合）保留融合后的完整候选池。`RerankDocumentBuilder` 从 `chunk_store` 取正文，`RerankerAdapter` 调用注入的重排能力；重排后才统一截取 `top_k`，转换为 `RetrievedCandidate` 并交给 Hydrator（数据补全组件）。因此公开方法 `RAGRetriever.retrieve(query, chunk_store, top_k=5)` 当前返回的是 `list[RagRetrievedChunk]`，不是候选 ID 列表。

检索之后，由调用方将结果交给 `ContextAssembler.assemble(chunks)`，再调用异步函数 `generate_rag_answer(query, assembled_context)`。上下文组装与生成不属于 `RAGRetriever` 的职责。

## 2. Component Responsibilities & Contracts（组件职责与契约）

契约定义见 [contracts.py](../backend/rag/contracts.py)，检索组件位于 [backend/rag/retrieval/](../backend/rag/retrieval/)。表中省略实例方法的 `self`。

| 组件 / 接口 | 输入 | 输出 | 职责及关键边界 |
| --- | --- | --- | --- |
| `DenseRetriever.search(query_vector, top_k=5)` | `np.ndarray` 查询向量、条数 | `list[DenseRetrieverResult]` | Protocol（协议接口）；实现负责相似度、排名与存储访问，不做查询向量化或补全。实现为 `ExactDenseSearch`、`PostgreSQLVectorSearch`。 |
| `BM25Retriever.search(query, chunk_texts, top_k=5)` | 查询、`dict[str, str]` 文本映射 | `list[BM25RetrieverResult]` | 独立 BM25 Okapi 打分与排序，只保留分数大于 0 的命中；默认 `RegexTokenizer`，可注入 `JiebaTokenizer`；不访问数据库。 |
| `RRFFusion.fuse(dense_results, bm25_results, top_k=None)` | 两路排名结果 | `list[FusedRetrieverResult]` | 校验输入排名并按 RRF 融合；保留各路分数与排名供阶段内使用，不直接相加原始分数；不取正文。 |
| `RerankDocumentBuilder.build(fused_results, chunk_store)` | 融合结果、`dict[str, Chunk]` | `list[RerankDocument]` | 按 ID 获取重排所需文本并保留输入顺序；不自行访问存储、不排序、不执行最终补全。 |
| `RerankerAdapter.rerank(query, documents)` | 查询、`list[RerankDocument]` | `list[RerankedResult]` | 调用注入的 `rerank(query, texts)`，校验返回数量、索引与分数，按分数降序稳定排序；不加载模型、不访问文本存储。 |
| `RAGRetriever.retrieve(query, chunk_store, top_k=5)` | 查询、`dict[str, Chunk]`、条数 | `list[RagRetrievedChunk]` | 唯一检索流程编排器：依次调用各阶段、适配候选并裁剪最终条数；不实现算法、SQL、上下文组装或生成。 |
| `RAGHydrator.hydrate(candidates)` | `list[RetrievedCandidate]` | `list[RagRetrievedChunk]` | 经注入的 `ChunkRepository`、`DocumentRepository` 的 `get_by_ids` 批量补全正文与来源，并检查数据缺失；保持输入顺序，不排序、不算分。 |
| `ContextAssembler.assemble(chunks)` | `list[RagRetrievedChunk]` | `AssembledContext` | 保持检索顺序，将独立 chunk 块拼成文档上下文，同时保留实际使用的完整 chunks；不做检索或回答判断。 |
| Generation：`generate_rag_answer(query, assembled_context)` | 查询、`AssembledContext` | 异步返回 `RAGAnswer` | 经 `call_llm_structured` 获取结构化输出；校验引用 ID，再解析真实来源；不负责检索。 |

### 数据边界

| 契约 | 当前字段 | 语义 |
| --- | --- | --- |
| `DenseRetrieverResult` | `chunk_id`、`dense_score`、`dense_rank` | Dense 阶段结果，排名从 1 开始。 |
| `BM25RetrieverResult` | `chunk_id`、`bm25_score`、`bm25_rank` | BM25 阶段结果，排名从 1 开始。 |
| `FusedRetrieverResult` | `chunk_id`、`dense_score`、`dense_rank`、`bm25_score`、`bm25_rank`、`rrf_score`、`rrf_rank` | 未被某一路命中的对应字段为 `None`；融合排名从 1 开始。 |
| `RerankDocument` | `chunk_id`、`content` | 重排所需的最小文本输入。 |
| `RerankedResult` | `chunk_id`、`previous_rank`、`rerank_score`、`rerank_rank` | `previous_rank` 为重排输入位置加 1；`rerank_rank` 为重排后的排名。 |
| `RetrievedCandidate` | **只有 `chunk_id`、`rank`** | 排名阶段到补全阶段的边界；不携带阶段分数。 |
| `RagRetrievedChunk` | **`chunk_id`、`document_id`、`content`、`source`、`retrieval_rank`** | 完整下游检索结果，不暴露 score；`retrieval_rank` 原样取候选的 `rank`。 |
| `AssembledContext` | `context_text`、`used_chunks` | 提供给 LLM 的文本及实际使用的 chunks。 |
| `RAGGenerationOutput` | `answer`、`supporting_chunk_ids` | LLM 结构化输出；引用 ID 仍须后端校验。 |
| `RAGAnswer` | `answer`、`sources` | 后端最终回答与解析后的来源。 |

Hydrator 以 Repository 返回的数据补全内容，`source` 取 Document 的 `file_name`。Repository 返回顺序不决定检索顺序，Hydrator 按候选输入顺序生成结果。

Citation（引用）的可信来源由后端确定：Generation 只允许 `supporting_chunk_ids` 引用 `used_chunks` 中存在的 ID，再从对应 `RagRetrievedChunk.source` 解析来源，按首次引用顺序去重。LLM 自行生成的文件名或来源文字不能被视为真实引用来源。

## 3. Design Decisions（关键设计决策）

- **Dense + BM25**：Dense 覆盖语义相似表达，BM25 补充精确词语、技术名词和 ID 等词项匹配；二者各自输出排名，便于独立验证。
- **使用 RRF**：向量相似度与 BM25 原始分数的量纲和分布不同，直接相加缺乏统一语义。RRF 使用排名融合，不依赖原始分数归一化；当前排名常数为 60。
- **Retriever 只做编排**：通过依赖注入调用各组件，使内存与 PostgreSQL Dense 实现共享同一条流程，算法、存储与下游生成的边界保持清晰。
- **分离候选与完整 chunk**：`RetrievedCandidate` 仅表达 ID 与最终排名，让 Hydrator 不必理解 Dense、BM25、RRF 或 Reranker 的分数；`RagRetrievedChunk` 提供上下文与引用需要的正文和来源。
- **当前采用 Exact Search（精确检索）**：内存实现直接计算余弦相似度，PostgreSQL 实现按 pgvector 余弦距离排序并限制条数。现阶段优先保持可核查的排名基线，尚未引入 ANN（Approximate Nearest Neighbor，近似最近邻）索引与相关调参、召回率权衡。
- **暂不直接接入 ChatService**：当前先完成独立 RAG 链路及其契约验证；未来计划在 Stage 3 通过 Tool Contract（工具契约）接入 Agent Runtime（智能体运行时）。这仍是计划，不是现有 Chat 或 Agent Tool 能力。

## 4. Failure Policy（失败策略）

**Empty Retrieval（空检索）是合法状态，Failure（执行失败）不等于 Empty。** 当前流程不捕获组件执行异常并替换为空列表。

| 情况 | 当前行为 |
| --- | --- |
| 两路正常返回空结果 | Fusion、重排、补全均可返回空列表；Assembler 产生 `context_text=""`、`used_chunks=[]`。 |
| 单路正常返回空列表，另一路有结果 | RRF 可继续融合另一条路的结果；这是正常空结果处理。 |
| Embedding、Dense、BM25 或重排执行抛出异常 | 默认向上传播，终止本次调用；没有单路异常自动降级。Dense 抛出异常时，后续 BM25 尚未执行。 |
| PostgreSQL 中存在 chunk，但所选模型没有可用向量 | 抛出 `EmbeddingCompatibilityError`，不能作为空知识库处理。 |
| Fusion 输入排名错误 | 非正整数、不连续的 `1..N`、重复排名或同一路重复 chunk ID 等显式报错，不静默重编号。 |
| 重排文本缺失 | Builder 在 `chunk_store` 中找不到候选时抛出 `KeyError`。 |
| Reranker 返回契约不合法 | 结果数必须等于输入文档数；索引必须是非布尔整数、唯一且在 `[0, N)` 内，分数必须是非布尔有限数值，否则抛出 `ValueError`。 |
| Hydration 数据缺失 | 缺 Chunk 或 Document 时抛出 `KeyError`，不丢弃候选继续回答。 |
| LLM 调用失败或结构化输出不合法 | 异常向上传播，不伪装成正常回答。 |
| LLM 返回上下文中不存在的引用 ID | Generation 抛出 `ValueError`，不接受虚构来源。 |

空上下文仍可进入生成。当前系统提示要求模型说明知识库未提供相关信息，允许补充自身知识但必须区分证据来源；没有 chunk 支持时应返回空引用列表。这是提示约束，不是已实现的语义正确性校验。后端对返回的引用 ID 执行确定性校验。

未来可以显式定义 Degraded Retrieval（降级检索），但当前未实现；单路正常为空不能被描述为捕获异常后的自动降级。

## 5. Evaluation & Testing（评估与测试）

### 已实现的确定性验证

现有测试使用 fake 组件或内存语料验证行为，覆盖如下边界；这些测试不等同于真实模型回答质量评估。

| 验证项 | 现有测试证据 |
| --- | --- |
| Retrieval Pipeline E2E（检索链路端到端） | [test_hybrid_pipeline_e2e.py](../tests/rag/test_hybrid_pipeline_e2e.py)、[test_rag_e2e.py](../tests/rag/test_rag_e2e.py)：混合候选流转、重排后裁剪、空语料及上下文与生成衔接。 |
| Ranking / Contract Validation（排序与契约校验） | [test_fusion.py](../tests/rag/test_fusion.py)、[test_reranker_adapter.py](../tests/rag/test_reranker_adapter.py)：排名连续性、融合依据、重排数量、索引与分数校验。 |
| Hydration Consistency（补全一致性） | [test_rag_hydrator.py](../tests/rag/test_rag_hydrator.py)：保留候选顺序与排名、正文来源补全、缺失数据报错。 |
| Citation Referential Integrity（引用参照完整性） | [test_rag_generation.py](../tests/rag/test_rag_generation.py)：合法 ID 解析来源、空引用、非法 ID 拒绝。 |
| Error Propagation（异常传播） | [test_rag_retriever.py](../tests/rag/test_rag_retriever.py)、[test_rag_generation.py](../tests/rag/test_rag_generation.py)：Embedding、数据一致性与 LLM 失败传播。 |

PostgreSQL / pgvector 持久化测试及真实 Qwen 冒烟测试另有 opt-in（显式启用）条件，不应把确定性测试覆盖当作所有真实依赖均已验证。[Phase 5 历史验收报告](testing/phase5_final_acceptance_report.md) 保存了当时的检查范围、测试结果与证据缺口；它是历史证据，不代表本次文档状态同步重新执行了全部验收。当前 Phase 7 Finalization 已完成，Stage 2 正式冻结。

### 已学习但当前未完整保留的评估能力

Hit@K（前 K 命中率）、Recall@K（前 K 召回率）、MRR（平均倒数排名）是已学习的基础检索指标，但当前仓库没有独立的 Retrieval Evaluation（检索评估）实现，不能声称存在完整可执行评估模块。

### 尚未实现

- Faithfulness（回答忠实度）自动化评估。
- Answer Correctness（回答正确性）自动化评估。
- 完整 Evaluation Dataset（评估数据集）。

**Citation ID 合法只证明引用指向提供过的 chunk，不证明被引用内容真正支持回答。** 当前引用参照完整性校验不能替代忠实度与正确性评估。

## 6. Known Limitations（当前限制）

- RAG 尚未接入 ChatService、Agent Loop 或前端，没有文件上传前端入口。
- 当前没有 Conversation 与 Document 的作用域管理。
- BM25 依赖调用方提供 `chunk_store`，没有独立持久化 Sparse Index（稀疏索引）；重排文本也来自该映射，调用方需要提供与 Dense 候选及 Repository 数据一致的语料。
- 尚无 Query Rewrite（查询改写）、ANN 或 Multi-hop Retrieval（多跳检索）。
- 不具备 Hybrid 单路异常自动降级能力。
- 没有完整自动化 Faithfulness Evaluation，也没有完整回答正确性评估与评估数据集。

Stage 3 计划通过 Tool Contract 将 RAG 接入 Agent Runtime；该接线与工具能力尚未完成。
