# Memory Read V2 Validation

Date: 2026-09-08
Status: Integration validation completed

## 1. Scope

This document records validation for the Memory Read V2 pipeline:

- Repository
- Dense Retrieval
- BM25 Retrieval
- RRF Fusion
- Hybrid Candidate Mapping
- Cross-Encoder Reranker
- Top-K Selection
- LLM Judge
- MemoryInjector
- ChatService Integration
- Graceful Degradation
- Boundary / Failure Contracts

The purpose is not to preserve full console logs, but to preserve:
- what was tested
- why it was tested
- key evidence
- final result
- important bugs / engineering conclusions

---

## 2. Baseline

Before Swagger integration tests:

- user_id = 1
- Memory Count = 8
- Max Memory ID = 8

Key Memories:

- ID 1: 用户最近开始学习LangGraph
- ID 2: 用户计划深入学习LangGraph的Agent工作流
- ID 3: 用户是测绘工程专业硕士研究生
- ID 4: 用户长期学习Python后端开发
- ID 5: 用户正在使用Python进行后端开发学习
- ID 6: 用户不打算学习Python后端开发了
- ID 7: 用户准备转向Java后端开发
- ID 8: 用户正在比较Python后端和Java后端，以便做出技术方向选择

After Case A / B, GET /memories confirmed:

- Memory Count remained 8
- Max Memory ID remained 8
- No Memory content changed
- No updated_at changed

Result:

PASS

Conclusion:

The two Swagger `/chat` tests did not create or modify long-term Memory through the downstream Memory Write path.

---

## 3. Automated / Regression Tests

### [KEEP] effective_top_n Regression

Purpose:

Ensure `top_n` means “retrieve up to N”, rather than “the corpus must contain N items”.

Historical Bug:

Real E2E previously failed because:

`top_n = 20`

but corpus size was only 8, causing bm25s to raise:

`ValueError: k of 20 is larger than the number of available scores`

Current behavior:

`effective_top_n = min(top_n, len(memories))`

Controlled boundary case:

- corpus size = 3
- requested top_n = 20
- requested top_k = 5
- effective_top_n = 3

Expected:

- Dense receives top_n = 3
- BM25 receives top_n = 3
- RRF receives top_n = 3
- Judge receives at most 3 items

Result:

PASS

---

### [KEEP] Index / Component Contract Tests

Covered cases include:

- Dense original index mapping
- Duplicate RRF index
- RRF index out of range
- Reranker result count mismatch
- Reranker index coverage
- Judge result count mismatch
- Judge index coverage
- `top_k > top_n`

Result:

PASS

---

### [KEEP] ChatService Memory Success Contract

Purpose:

Verify ChatService calls MemoryReader with:

- correct user_id
- correct query
- top_n = 20
- top_k = 5

And injects returned `memory_context` into the system message.

Result:

PASS

---

### [KEEP] ChatService Memory Failure Degradation

Purpose:

Verify Memory Read failure does not make the entire Chat request fail.

Controlled failure:

`MemoryReader.read()` raises:

`ValueError("LLM Judge 返回了空内容")`

Expected:

- exception does not escape `chat()`
- memory_context remains empty
- original agent system prompt remains
- Chat LLM still executes
- ChatResponse still returns

Result:

PASS

Engineering conclusion:

Memory Read is currently an enhancement layer. If it fails, ChatService falls back to normal chat.

---

## 4. Manual Integration Tests

These are `[MANUAL KEEP]` tests because they use real infrastructure and models:

- FastAPI
- SQLite
- real Memory ORM
- Qwen3-Embedding-0.6B
- jieba / bm25s
- RRF
- bge-reranker-v2-m3
- DeepSeek Judge
- MemoryInjector
- ChatService

They are valuable, but slower and partially non-deterministic because the external LLM Judge may occasionally return empty content.

---

### [MANUAL KEEP] Case A - Relevant Memory / Positive Path

Query:

`我最近在学习哪个 Agent 工作流框架？`

Expected:

- relevant LangGraph memories are retrieved
- relevant memories rank near the top
- Judge selects relevant memories
- memory_context is non-empty
- `/chat` returns 200
- final answer uses the LangGraph memory

Actual:

Repository:
- 8 memories loaded

Boundary:
- requested_top_n = 20
- corpus_size = 8
- effective_top_n = 8

Dense:
- Memory ID 2 ranked 1
- Memory ID 1 ranked 2

BM25:
- Memory ID 2 ranked 1
- Memory ID 1 ranked 2
- zero-score items were preserved in raw trace but excluded from BM25 contribution to RRF

RRF:
- Memory ID 2 ranked 1
- Memory ID 1 ranked 2

Hybrid Candidate Mapping:
- original `memories[]` index correctly mapped to `hybrid_candidates[]`

Reranker:
- Memory ID 2 ranked 1
- Memory ID 1 ranked 2
- local reranker index correctly mapped back to the hybrid candidate

Judge:
- Memory ID 2 -> True
- Memory ID 1 -> True
- unrelated Top-K memories -> False

Injector:
- selected_count = 2
- memory_context non-empty

Chat:
- HTTP 200
- final response correctly identified LangGraph

Result:

PASS

---

### [MANUAL KEEP] Case B - No Relevant Memory

Query:

`根据你记住的内容，我最喜欢吃什么？`

Expected:

- Retriever may still return candidates
- irrelevant candidates must not be injected
- Judge should reject unrelated memories
- memory_context should be empty
- `/chat` should still return 200

Actual:

Repository:
- 8 memories loaded

Boundary:
- requested_top_n = 20
- effective_top_n = 8

Dense:
- returned 8 candidates
- no strong relevant candidate existed

BM25:
- 2 candidates received positive lexical scores
- remaining candidates had score = 0
- score = 0 items did not contribute to RRF

RRF:
- still returned candidates because Dense ranking existed

Reranker:
- all scores were very low and close together
- no obviously strong relevant item appeared

Judge:
- Top-5 all False

Selected:
- selected_count = 0

Injector:
- memory_context empty

Chat:
- HTTP 200

Important note:

The final Chat response still mentioned LangGraph, but this request used the same conversation that had just discussed LangGraph.

The MemoryReader trace proved:

- selected_count = 0
- memory_context = empty

Therefore the LangGraph mention came from conversation history rather than long-term Memory injection.

Result:

PASS

Engineering conclusion:

Candidate generation can produce false positives. Final Memory use is controlled by the Judge, not by retrieval alone.

---

### [MANUAL KEEP] Case C - Semantic Positive

Query:

`我最近在研究什么 Agent 编排相关的技术？`

Purpose:

Verify the system can retrieve LangGraph even when the query does not explicitly contain the word `LangGraph`.

This primarily validates semantic retrieval behavior in the real `/chat` pipeline.

Actual:

Repository:
- 8 memories loaded

Boundary:
- requested_top_n = 20
- corpus_size = 8
- effective_top_n = 8

Dense:
- Memory ID 2 ranked 1
  - similarity = 0.612967
  - content: 用户计划深入学习LangGraph的Agent工作流
- Memory ID 1 ranked 2
  - similarity = 0.514872
  - content: 用户最近开始学习LangGraph

BM25:
- Memory ID 2 ranked 1
  - score = 1.367153
- Memory ID 1 ranked 2
  - score = 0.866712
- Memory ID 3 and ID 8 also received positive lexical scores
- zero-score items were excluded from BM25 ranking contribution to RRF

RRF:
- Memory ID 2 ranked 1
- Memory ID 1 ranked 2

Hybrid Candidate Mapping:
- index mapping remained correct

Reranker:
- Memory ID 1 ranked 1
- Memory ID 2 ranked 2

Judge:
- Memory ID 1 -> True
- Memory ID 2 -> True
- unrelated Java / Python direction memories -> False

Selected:
- selected_count = 2

Injector:
- memory_context non-empty
- contained both LangGraph memories

Chat:
- HTTP 200

Result:

PASS

Engineering conclusion:

The real pipeline successfully recovered LangGraph from a semantically related query that did not explicitly name LangGraph.

Note:

BM25 also had lexical signal in this case, so this is not a “Dense-only” experiment. It validates the real hybrid pipeline rather than isolating Dense Retrieval.

---

### [MANUAL KEEP] Case D - No Memory User

Setup:

A new conversation was created for:

- user_id = 2

Query:

`我最近在学习什么？`

Purpose:

Verify that a user with no stored Memory follows the valid empty-corpus path rather than being treated as a failure.

Actual:

Input Validation:
- PASS

Repository:
- returned 0 memories

MemoryReader:
- entered the `NO MEMORY` early-return branch

Runtime trace:

`Repository returned 0 memories`

Then:

`Memory Read returns empty result.`

The following components did not run:

- Dense Retrieval
- BM25
- RRF
- Reranker
- Judge
- Injector

Chat:
- HTTP 200

Final answer:
- correctly stated that there was currently no learning-context/history available to answer directly

Result:

PASS

Engineering conclusion:

“No Memory” is a valid business state, not an exception.

The early-return branch correctly prevents unnecessary retrieval / reranking / LLM Judge work.

---

## 5. Manual Real Retrieval Smoke

### [MANUAL KEEP] Identifier Retrieval

Query:

`22ecfec 是什么提交？`

Target Memory:

`Memory Read V1 对应 Git commit 为 22ecfec`

Observed:

- Dense Rank 2
- BM25 Rank 1
- RRF Rank 1
- Reranker Rank 1

Result:

PASS

Engineering conclusion:

Hybrid Retrieval improves recall for exact identifiers such as Git commit hashes, where lexical retrieval is especially useful.

---

## 6. LLM Judge Non-Determinism

Observed real behavior:

DeepSeek Judge may occasionally return empty content:

`ValueError: LLM Judge 返回了空内容`

The same E2E case may:

- pass on one run
- fail at the Judge call on another run

Conclusion:

Real LLM E2E is not a deterministic regression test.

Do not interpret a single empty Judge response as a Dense / BM25 / RRF failure.

Current handling:

ChatService catches Memory Read failure and falls back to normal chat.

Status:

Known Tech Debt

---

## 7. Known Tech Debt

- DeepSeek Judge occasionally returns empty content
- ChatService currently catches broad `Exception`
- ChatService currently uses `print` rather than formal logging
- temporary MemoryReader debug prints should be removed after V2 validation
- DenseMemoryRetriever is still Memory-specific and may later be generalized
- requirements.txt and the real virtual environment have dependency drift
- repeated test bootstrap through `sys.path` should later be replaced by a cleaner pytest/package setup

These issues should not block Memory Read V2 completion.

---

## 8. Swagger Integration Summary

| Case | Purpose | Result |
|---|---|---|
| A | Relevant Memory / positive injection | PASS |
| B | No relevant Memory / reject false positives | PASS |
| C | Semantic positive retrieval | PASS |
| D | No Memory user / early return | PASS |

Additional checks:

- `top_n > corpus` real HTTP path: PASS
- BM25 zero-score handling: PASS
- index-space mapping: PASS
- Judge selection / rejection: PASS
- memory_context generation: PASS
- ChatService HTTP success path: PASS
- no Memory Write side effects after A / B: PASS

---

## 9. Final Cleanup Checklist

- [ ] Remove `[TEMP]` Runtime Trace debug prints
- [ ] Search old V1 symbols:
  - `MemoryRetriever`
  - `RetrievedMemory`
  - `retrieval_score`
  - `_retriever`
- [ ] Check old tests for legacy fields
- [ ] Run all `[KEEP]` regression tests
- [ ] Run one final `[MANUAL KEEP]` real E2E
- [ ] Organize `test/memory/`
- [ ] Review `git diff`
- [ ] Confirm changed business files
- [ ] Commit Memory Read V2

After the V2 commit:

Memory Lifecycle Maintenance
→ PostgreSQL + pgvector
→ RAG
