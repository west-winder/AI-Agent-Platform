# Phase 5 Final Acceptance Report

Date: 2026-10-12
Scope: AI Agent Platform — Stage 2 Unified RAG / Retrieval Layer
Baseline: `fbf63e2 feat(rag): complete hybrid retrieval pipeline (BM25 + RRF + Reranker)`
Review type: Read-only acceptance review（本轮未修改任何代码）

---

## 1. Scope

本报告检查 Phase 5 Retrieval Layer 是否满足预期设计，并给出是否可结束 Phase 5、
进入 Stage 3 的结论。

- 检查对象：`backend/rag/retrieval/**`、`backend/rag/context_assembler.py`、
  `backend/rag/generation.py`
- 检查方式：静态依赖审计（AST）、运行时依赖审计（独立子进程 import + `sys.modules`）、
  全仓引用 grep、测试执行
- 本轮**不新增功能、不做架构重构、不做代码清理**；发现的非阻塞问题**只记录**

---

## 2. Acceptance Checklist

### 2.1 Retrieval Orchestrator

**结论：PASS**

- **唯一 Orchestrator**：`backend/rag/retrieval/retriever.py::RAGRetriever`；
  `retrieve()` 实现在全仓只有 1 处。
- **旧 PostgreSQL-specific Pipeline 已消失**：`backend/rag/retrieval/postgres_retriever.py`
  已删除；`PostgreSQLRAGRetriever` 在生产代码与测试中命中数 = **0**。
- **只负责流程编排**：

  ```
  RAGRetriever.__init__(
      self, embedder, dense_retriever, bm25_retriever, fusion,
      rerank_document_builder, reranker, hydrator,
  )
  RAGRetriever.retrieve(self, query, chunk_store, top_k=5)
  ```

  私有 stage 方法全部为纯编排、无算法实现：
  `_embed_query`、`_dense_retrieve_candidates`、`_retrieve_sparse_candidates`、
  `_fuse`、`_rerank`、`_hydrate`

- **不直接依赖 PostgreSQL / Exact Search / storage**（AST 提取的 import）：

  | import | 性质 |
  |---|---|
  | `numpy` | 类型注解 |
  | `backend.rag.contracts` | 契约 |
  | `backend.rag.retrieval.bm25_retriever::BM25Retriever` | 组件注入 |
  | `backend.rag.retrieval.dense_retriever::DenseRetriever` | **Protocol（抽象）** |
  | `backend.rag.retrieval.fusion::RRFFusion` | 组件注入 |
  | `backend.rag.retrieval.hydrator::RAGHydrator` | 组件注入 |
  | `backend.rag.retrieval.rerank_document_builder::RerankDocumentBuilder` | 组件注入 |
  | `backend.rag.retrieval.reranker_adapter::RerankerAdapter` | 组件注入 |

  无 `sqlalchemy` / `SessionLocal` / ORM，无 `postgres_vector_search`，无 `exact_search`。

- **运行时验证**（独立进程 import `backend.rag.retrieval.retriever`）：

  ```
  被拉起：backend, backend.rag, backend.rag.contracts, backend.rag.retrieval,
          …bm25_retriever, …dense_retriever, …fusion, …hydrator,
          …rerank_document_builder, …reranker_adapter, …retriever,
          …tokenizer, backend.retrieval, backend.retrieval.rrf
  storage 泄漏：无
  ```

### 2.2 Dense Retrieval

**结论：PASS**

Contract：`backend/rag/retrieval/dense_retriever.py`

```python
class DenseRetriever(Protocol):
    def search(self, query_vector: np.ndarray, top_k: int = 5)
        -> list[DenseRetrieverResult]: ...
```

| 实现 | 文件 | 构造 | `search` 签名 | 默认值 |
|---|---|---|---|---|
| InMemory Dense Search | `exact_search.py` | `ExactDenseSearch(chunk_vectors)` | `(self, query_vector, top_k)` | `top_k=5` |
| PostgreSQL Dense Search | `postgres_vector_search.py` | `PostgreSQLVectorSearch(embedding_model, session_factory=SessionLocal)` | `(self, query_vector, top_k)` | `top_k=5` |

- 两者 `search` 参数列表**逐字一致**，均满足同一 Contract。
- 语料（`chunk_vectors`）与 Embedding Model 身份（`embedding_model`）在**构造期**绑定，
  `search()` 只接收 Query Vector。

职责边界：

| 实现 | 直接 import | 是否 hydration / context assembly / generation |
|---|---|---|
| `ExactDenseSearch` | `numpy`, `backend.rag.contracts` | 否 |
| `PostgreSQLVectorSearch` | `math`, `numpy`, `sqlalchemy.select`, `SessionLocal`, 3 个 ORM, `backend.rag.contracts` | 否 |

`PostgreSQLVectorSearch` 的 storage import 全部服务于 vector search 本身
（pgvector distance 表达式 + `embedding_model` 过滤），未越界。

### 2.3 Sparse Retrieval

**结论：PASS**

- **独立存在**：`backend/rag/retrieval/bm25_retriever.py::BM25Retriever`，
  直接 import 仅 `math` + `contracts` + `tokenizer`，
  不依赖 Dense / Fusion / Reranker / Hydrator。
- **返回 `BM25RetrieverResult`**：`contracts.py:71`，字段
  `chunk_id / bm25_score / bm25_rank`（1-based）；不含 content，不含向量信息。
- **参与 Hybrid Pipeline**：`RAGRetriever._retrieve_sparse_candidates()` 由 `chunk_store`
  派生 `chunk_texts` 后调用，产物与 Dense 结果一起进入 `_fuse()`。
- 语义约定：`score <= 0`（与 Query 无任何词重合）的 Chunk 不进入候选池，避免污染 Fusion。

### 2.4 Fusion

**结论：PASS**

- **独立组件**：`backend/rag/retrieval/fusion.py::RRFFusion`，公开方法仅 `fuse()`。
- **不依赖具体 Retrieval backend**：该模块的 backend 依赖只有
  - `backend.rag.contracts`（契约）
  - `backend.retrieval.rrf`（通用 RRF 算法；实测该文件**不引用** `backend.rag`）

  不 import 任何 retriever 实现，输入只是两条 rank 序列。
- **保留 Dense/Sparse ranking contract**：输出 `FusedRetrieverResult` 完整保留
  `dense_score` / `dense_rank` / `bm25_score` / `bm25_rank`（单边缺失为 `None`），
  以及 `rrf_score` / `rrf_rank`。
- **上游 ranking 破损即失败**（不静默重编号）：`_ordered_by_rank()` 校验 chunk_id 非空、
  组内不重复、rank 为正整数且必须连续 `1..N`。
- `RRF_RANK_CONSTANT = 60` 定义在 Fusion 内部，不进入 Contract。

### 2.5 Reranker

**结论：PASS**

```
FusedRetrieverResult[]
    ↓  rerank_document_builder.py::RerankDocumentBuilder.build(fused_results, chunk_store)
RerankDocument[]
    ↓  reranker_adapter.py::RerankerAdapter.rerank(query, documents)
RerankedResult[]
```

- **经 Adapter 接入**：`RerankerAdapter` 通过构造注入底层通用排序能力
  （`rerank(query, texts) -> list[RerankResult]`），自身不加载模型。
- **不直接访问 storage**：
  - `reranker_adapter.py` 直接 import = `math` / `numbers` / `backend.rag.contracts`
  - `rerank_document_builder.py` 直接 import = `backend.rag.contracts`
  - 文本来源 `chunk_store` 由调用方（`RAGRetriever._rerank()`）传入，Builder 不自行取数。
- Adapter 校验底层输出契约：index 必须 int、唯一、在 `[0, N)`；score 必须数值且 finite；
  随后按 score 降序稳定排序，翻译回 `RerankedResult`（含 `previous_rank`，
  刻意不叫 `fusion_rank`——不假设上游一定是 Fusion）。

### 2.6 Hydration Boundary

**结论：PASS**

- `backend/rag/retrieval/hydrator.py::RAGHydrator` 是全仓唯一 Hydrator
  （生产代码中 `def hydrate` 只有 1 处；`PostgreSQLHydrator` 命中数 = **0**）。
- 转换方向：`RetrievedCandidate` → `RagRetrievedChunk`；不重排、不算分、保持输入顺序，
  `retrieval_rank` 原样取 `candidate.rank`。
- **不知道 Dense / BM25 / RRF / Reranker**：直接 import 仅 `typing.TYPE_CHECKING` 与
  `backend.rag.contracts`；`ChunkRepository` / `DocumentRepository` 的 import 位于
  `if TYPE_CHECKING:` 块内。
- **运行时验证**（独立进程 import `backend.rag.retrieval.hydrator`）：被拉起模块为
  `backend` / `backend.rag` / `backend.rag.contracts` / `backend.rag.retrieval(.hydrator)`
  → **sqlalchemy 泄漏 = 无**。
- 一致性检查属 Hydrator 职责：
  - 缺 chunk → `KeyError("Hydration failed: RetrievedCandidate references missing chunks: [...]")`
  - 缺 document → `KeyError("Hydration failed: chunks reference missing documents: [...]")`

### 2.7 Repository Boundary

**结论：PASS**

`backend/rag/retrieval/repositories.py`：

| 类 | 公开方法 | 职责 |
|---|---|---|
| `ChunkRepository` | `get_by_ids(chunk_ids)` | 按 chunk_id 批量查询 |
| `DocumentRepository` | `get_by_ids(document_ids)` | 按 document_id 批量查询 |

- 只有 persistence access，**无任何 retrieval pipeline logic**
  （无 ranking / fusion / hydration / 一致性检查——一致性检查在 Hydrator）。
- 生产文件中**不含** InMemory 实现；测试 fake repository 位于
  `tests/rag/inmemory_repositories.py`（测试目录），符合「不要把测试实现混入
  production repository 文件」的要求。

### 2.8 测试覆盖（分类）

**结论：PASS**（原始执行结果见 §6）

- **[KEEP] deterministic**（不联网 / 不连库，默认进入回归）：
  除下表 [MANUAL KEEP] 与被 opt-in gate 的文件外的全部用例。本阶段相关：
  `test_dense_retriever_injection.py`(3)、`test_exact_dense_search.py`(5)、
  `test_rag_hydrator.py`(4)、`test_rag_retriever.py`(4)、
  `test_hybrid_pipeline_e2e.py`(3)、`test_rag_integration.py`(3)、`test_rag_e2e.py`(3)、
  `test_fusion.py`(16)、`test_bm25_retriever.py`(12)、`test_reranker_adapter.py`(15)、
  `test_rerank_document_builder.py`(5)、`test_context_assembler.py`(5)、
  `test_tokenizer.py`(11)。`test_postgres_dense_retriever.py` 17 个用例中 11 个为
  确定性（构造期 / 参数校验）。
- **[MANUAL KEEP]**：`test_real_qwen_persistent_smoke.py`（真实 Qwen + 真实 PG）；
  `test_file_ingestion_postgres.py` 的 Phase 3.1 / 3.2 / 3.3 段。
- **[TEMP]**：`tests/` 全仓扫描 `[TEMP]` 标记 → **0 命中**（无遗留临时脚本）。

---

## 3. Architecture Conformance Summary

| Phase 5 目标架构段 | 当前实现 | 状态 |
|---|---|---|
| Query | — | — |
| Embedding | `RAGRetriever._embed_query()` | OK |
| RAGRetriever（唯一 Orchestrator） | `retriever.py::RAGRetriever` | OK |
| Dense Retrieval | `DenseRetriever` Contract + `ExactDenseSearch` / `PostgreSQLVectorSearch` | OK |
| BM25 Sparse Retrieval | `bm25_retriever.py::BM25Retriever` | OK |
| Fusion (RRF) | `fusion.py::RRFFusion` + `backend.retrieval.rrf` | OK |
| Reranker | `RerankDocumentBuilder` + `RerankerAdapter` | OK |
| RetrievedCandidate | `contracts.py::RetrievedCandidate` | OK |
| Hydrator | `hydrator.py::RAGHydrator` | OK |
| RagRetrievedChunk | `contracts.py::RagRetrievedChunk` | OK |
| ContextAssembler | `backend/rag/context_assembler.py::ContextAssembler` | OK |
| Generation | `backend/rag/generation.py::generate_rag_answer` | OK |

禁止存在的组件：`PostgreSQLRAGRetriever` = **0 命中**、`PostgreSQLHydrator` = **0 命中**
（生产代码与测试均无）。

Phase 5 检索层最终类清单（`backend/rag/retrieval/`）：

```
RAGRetriever          retriever.py
DenseRetriever        dense_retriever.py          (Protocol)
ExactDenseSearch      exact_search.py
PostgreSQLVectorSearch postgres_vector_search.py
BM25Retriever         bm25_retriever.py
RRFFusion             fusion.py
RerankDocumentBuilder rerank_document_builder.py
RerankerAdapter       reranker_adapter.py
RAGHydrator           hydrator.py
ChunkRepository       repositories.py
DocumentRepository    repositories.py
```

---

## 4. Blocking Issues

**None.**

判定依据（逐条核对，均未命中阻塞条件）：

| 阻塞条件 | 核对结果 |
|---|---|
| Contract 错误 | 无。Dense/BM25/Fusion/Rerank/Hydration 各段契约字段完整，两个 Dense 实现的 `search` 签名逐字一致；`FusedRetrieverResult` 完整保留 dense/bm25 rank 与 score。 |
| Pipeline 无法运行 | 无。完整 Hybrid 链路可运行（`test_hybrid_pipeline_e2e.py`、`test_rag_integration.py`、`test_rag_e2e.py` 均确定性通过）。 |
| 测试失败 | 无。`tests/rag` 的 **failed = 0**（详见 §6）。 |
| 架构边界严重破坏 | 无。RAGRetriever 不依赖具体 Dense 实现与 storage；Hydrator 运行时零 storage 依赖；Repository 无 pipeline logic；PostgreSQL 不再拥有 Pipeline。 |

补充说明（不构成阻塞）：

- 全仓 `pytest -q` 存在 4 个 error，位于 `tests/memory/test_memory_reader_v2_e2e.py`
  （`fixture 'reader' not found`）。该文件本轮未被改动，属既有 [MANUAL KEEP] 状态，
  与 RAG 检索层无关，也不影响 Stage 3（Tool Calling + Agent Loop）的开展。
- §5 的 O6（两个 opt-in 持久化测试未实测）是**证据缺口**而非失败：它们无法在当前
  环境（无真实 PostgreSQL + pgvector / 无真实 Qwen）下执行，不能据此判定实现有缺陷。

---

## 5. Non-blocking Issues

以下问题**只记录，本轮未修改**。

| ID | 类别 | 问题 | 影响 | 建议处理时机 |
|---|---|---|---|---|
| O1 | storage abstraction refinement | `RAGRetriever.retrieve()` 仍要求调用方传入 `chunk_store`。持久化形态下必须先物化全量语料（`test_rag_retriever_postgres.py` 用 `ChunkRepository.get_by_ids()` 手工装载）。目标架构未定义 Sparse 语料来源，且**不存在持久化 BM25 索引**。 | Phase 5 的 Hybrid 在 PG 形态下无法脱离调用方提供的语料。属设计边界，非缺陷。 | Stage 3 之后；若要支持大规模持久化 Hybrid 检索，需要单独的 Sparse Index 设计专题。 |
| O2 | API ergonomics | 全仓 grep 显示 `RAGRetriever` / `ContextAssembler` / `RAGHydrator` / `generate_rag_answer` 目前只被 tests 引用，**无 API / service 接线**。 | 阶段符合预期（检索层刚收敛完）。 | Stage 3 若把 RAG 作为 Agent 的一个工具，需要补一层 service / API 接线。 |
| O3 | 命名优化 | 类名冲突：`backend/retrieval/bm25.py::BM25Retriever`（`bm25s` + jieba，Memory 层使用）与 `backend/rag/retrieval/bm25_retriever.py::BM25Retriever`（纯 Python BM25 Okapi，RAG 层使用）同名、不同实现。 | 既有状态，非本阶段引入。跨子系统不影响正确性，但检索时可读性差。 | 低优先级；建议某一阶段统一命名（如 `LibraryBM25Retriever` / `RagBM25Retriever`）。 |
| O4 | API ergonomics | `DenseRetriever` Protocol 未加 `@runtime_checkable`，无法 `isinstance` 校验，契约仅由静态层保障。 | 无功能影响。 | 若后续需要运行期契约校验再补。 |
| O5 | 命名优化 | ~~两个迁移后测试的 docstring 仍引用旧文件名：`test_postgres_dense_retriever.py:17`、`test_rag_retriever_postgres.py:12`（「原 test_postgres_retriever.py …迁移到本文件」的来源注记）。~~ **已修复**（2026-10-12）：两处来源注记已删除，`postgres_retriever` / `postgres_hydrator` 命名在生产代码、测试、文档中命中数均为 0。 | 已消除。 | 已关闭 |
| O6 | 测试证据缺口 | 两个 opt-in 持久化测试（`test_rag_retriever_postgres.py`、`test_real_qwen_persistent_smoke.py`）未实测：需真实 PostgreSQL + pgvector，后者还需真实 Qwen 模型。改接 `RAGRetriever` 后链路增加 BM25 + Fusion + Rerank 三段，原有 dense-only 排序断言需人工复核。 | 唯一无法给出 PASS 证据的区域。 | 建议在真实 PG 环境安排一次 manual run（不阻塞 Stage 3 开工）。 |

---

## 6. Test Result

```
$ python -m pytest -q tests/rag
.................................................................sssssss [ 33%]
sssss............................................ssssss................. [ 67%]
......ssssssssss.......sssssss.......................................    [100%]
178 passed, 35 skipped in 15.07s
```

| 指标 | 数值 |
|---|---|
| **passed** | **178** |
| **skipped** | **35** |
| **failed** | **0** |
| error | 0 |
| 收集总数 | 213 |
| 进程退出码 | 0 |

### 跳过原因分布（35）

| 文件 | 跳过数 | 原因 |
|---|---|---|
| `test_file_ingestion_postgres.py` | 12 | 需 `opt-in` 环境变量 |
| `test_postgres_dense_retriever.py` | 6 | 需 `opt-in` 环境变量 |
| `test_rag_schema.py` | 5 | 需 `opt-in` 环境变量 |
| `test_rag_ingestion.py` | 5 | 需 `opt-in` 环境变量 |
| `test_rag_hydrator_postgres.py` | 5 | 需 `opt-in` 环境变量 |
| `test_rag_retriever_postgres.py` | 1 | 需 `opt-in` 环境变量 |
| `test_real_qwen_persistent_smoke.py` | 1 | **[MANUAL KEEP]** 真实 Qwen + 真实 PG |

### 按文件用例数（收集口径）

| 文件 | 用例 |
|---|---|
| test_file_ingestion.py | 31 |
| test_postgres_dense_retriever.py | 17 |
| test_fusion.py | 16 |
| test_reranker_adapter.py | 15 |
| test_file_ingestion_postgres.py | 12 |
| test_bm25_retriever.py | 12 |
| test_tokenizer.py | 11 |
| test_markdown_loader.py | 11 |
| test_pdf_loader.py | 9 |
| test_txt_loader.py | 8 |
| test_rerank_document_builder.py | 5 |
| test_rag_schema.py | 5 |
| test_rag_ingestion.py | 5 |
| test_rag_hydrator_postgres.py | 5 |
| test_rag_generation.py | 5 |
| test_ingestion_validator.py | 5 |
| test_exact_dense_search.py | 5 |
| test_context_assembler.py | 5 |
| test_chunker.py | 5 |
| test_rag_retriever.py | 4 |
| test_rag_hydrator.py | 4 |
| test_chunk_embedding.py | 4 |
| test_rag_integration.py | 3 |
| test_rag_e2e.py | 3 |
| test_hybrid_pipeline_e2e.py | 3 |
| test_dense_retriever_injection.py | 3 |
| test_real_qwen_persistent_smoke.py | 1 |
| test_rag_retriever_postgres.py | 1 |

### 全仓参考（非本阶段门槛）

```
$ python -m pytest -q
369 passed, 35 skipped, 4 errors
```

4 个 error 全部位于 `tests/memory/test_memory_reader_v2_e2e.py`
（`fixture 'reader' not found`），该文件本轮未改动，属既有 [MANUAL KEEP] 状态。

---

## 7. Recommendation

**可以结束 Phase 5。**

依据：

1. §2 的 8 项验收检查全部 PASS；§3 的目标架构对照全部 OK。
2. §4 **Blocking Issues = None**：无 Contract 错误、Pipeline 可运行、
   `tests/rag` failed = 0、架构边界未被破坏。
3. 被禁止的 `PostgreSQLRAGRetriever` / `PostgreSQLHydrator` 在两个命名空间中命中数均为 0，
   检索层已收敛到单一 RAG Pipeline。

**建议进入：Stage 3 Tool Calling + Agent Loop。**

进入 Stage 3 时建议一并携带的已知项（均不阻塞开工）：

- **O6**（证据缺口）：在真实 PostgreSQL + pgvector 环境对
  `test_rag_retriever_postgres.py` 做一次 manual run，复核改接 Hybrid 后的排序断言。
- **O2**（接线）：Stage 3 若把 RAG 检索作为 Agent 可调用工具，需要补
  `RAGRetriever → service / API` 的接线层。
- **O1**（设计边界）：持久化 Hybrid 检索的 Sparse 语料来源尚未定义，
  留给后续 Sparse Index 专题。

清理项（随时可做，非必须）：O3（BM25 同名类消歧）。

O5 已在本次收尾中修复，不再列为清理项。

---

## 8. 审查约束与后续动作

审查约束（本报告主体）：

- 只读审查，未修改任何代码。
- §5 记录的非阻塞问题按约束**只记录不修**。

审查后收尾（2026-10-12，不改变上述任何结论）：

- **O5 已修复**：删除 `test_postgres_dense_retriever.py` /
  `test_rag_retriever_postgres.py` 中的迁移来源 docstring 注记。
  修复后 `pytest -q tests/rag` 仍为 178 passed / 35 skipped。
- Phase 5 变更已以 `feat(rag): complete unified retrieval pipeline` 提交。
