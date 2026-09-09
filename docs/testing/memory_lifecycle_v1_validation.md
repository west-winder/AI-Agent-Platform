# Memory Lifecycle V1 Validation

Date: 2026-09-09
Status: Final validation completed (Code Freeze)

## 1. Scope

本文档记录 Memory Lifecycle V1 的最终验证，覆盖：

- Lifecycle V1 Scope（只做 current / historical 两种状态）
- current / historical Contract
- Transaction Contract
- Judge Coverage Contract
- [KEEP] Tests
- [MANUAL KEEP] Tests
- Controlled Integration Cases（A / B / C / D）
- Swagger Cases（1 / 2 / 3）
- Memory Read status filtering
- 发现的真实 Bug（无）
- 已知 Tech Debt
- 不在 V1 Scope 的内容

验证数据库是 F:\AI_Agent_Platform\test.db 的副本，位于 Worktree 根目录，
所有 mutation 只发生在副本上。F 盘数据库未被修改。

---

## 2. Pre-check

Worktree 副本 `test.db`：

- memories 列：`id, user_id, content, memory_type, created_at, updated_at, memory_status, historical_at`
- 原有 8 条 Memory：`memory_status = current`（8/8），`historical_at = NULL`（8/8）

结果：PASS

---

## 3. [KEEP] Tests

```
tests/memory/test_memory_lifecycle_v1.py              24/24 PASS
tests/memory/test_chat_service_memory_regression.py    2/2 PASS
```

结果：PASS

---

## 4. [MANUAL KEEP] Tests

```
tests/memory/test_memory_reader_v2_e2e.py           4/4 PASS（Case 1-4）
tests/memory/test_memory_reader_v2_real_retrieval.py PASS（Target Rank 1）
```

结论：

新增 memory_status 过滤没有破坏 Dense / BM25 / RRF / Reranker / Judge / Injector。

---

## 5. Controlled Integration Cases

驱动真实 Pipeline（真实 Extractor + Qwen Embedding + DeepSeek Judge），
每个 Case 使用独立 user_id（9901..9904），不碰 F 盘 user_id=1 数据。

### Case A — Conflict Transition（PASS）

- BEFORE：id=9 current「用户正在学习 Python 后端开发」
- 输入：「我已经决定停止学习 Python 后端开发了，以后不学了」
- AFTER：
  - id=9 current → **historical**（historical_at SET）
  - id=10 新插入 **current**「用户决定停止学习Python后端开发，以后不再学习。」
- 旧 Memory 未被物理删除。

### Case B — Related Coexistence（PASS，需说明）

- 初始输入「我最近在学 Spring Boot 框架」→ saved=0。
  - 原因：LLM Validator 判定「用户正在学习Spring Boot」为**临时性状态**，valid=False，
    在 Validation 阶段被丢弃，未走到 Relationship Judge。
  - 这是真实 LLM 非确定性行为，不是代码 bug。
- 改用长期语义输入「我一直很喜欢用 Java 后端生态做开发」：
  - AFTER：id=11 current「用户正在学习 Java 后端开发」保持 current，
    id=15 新插入 current「用户一直很喜欢用 Java 后端生态做开发」
  - 两者共存，符合 related 语义。
- related → 共存的确定性逻辑由 [KEEP] 单测覆盖。

### Case C — Exact Duplicate（PASS）

- BEFORE：id=12 current「用户正在学习 LangGraph」
- 输入：「用户正在学习 LangGraph」（精确重复）
- AFTER：saved=0，无重复行，id=12 仍 current。
- 确认 Exact Dedup 在 Relationship Judge 之前短路。

### Case D — Historical Does Not Block New Current（PASS）

- BEFORE：id=13 **historical**「用户正在学习 Python 后端」
- 输入：「我重新开始学习 Python 后端了」
- AFTER：
  - id=13 仍 historical
  - id=14 新插入 current「用户正在重新学习Python后端开发」
- 证明 Exact Dedup / Related Retrieval / Relationship Judge 只考虑 current corpus。

---

## 6. Memory Read Status Filtering（PASS）

在 user_id=9904（id=13 historical + id=14 current）上执行 `get_memories_for_read`：

- 只返回 id=14（current）。
- id=13（historical）被排除，不进入 Dense/BM25/RRF corpus。

---

## 7. Swagger Real Integration

从 Worktree 根目录启动 uvicorn（cwd 指向 Worktree，`sqlite:///./test.db` 命中副本）。

### Case 1 — Normal Chat（PASS）

- `POST /chat`（conversation_id=2）→ 200，answer 正常，message_id 正常。
- Memory 无变化（普通闲聊未提取 memory）。

### Case 2 — Conflict Lifecycle（PASS）

- 输入：「我以后不学习 LangGraph 了，打算放弃」
- `GET /memories` 显示：
  - id=1「用户最近开始学习LangGraph」→ **historical**
  - id=2「用户计划深入学习LangGraph的Agent工作流」→ **historical**
  - id=16「用户决定以后不再学习LangGraph，打算放弃」→ **current**（新插入）
- Response 正确暴露 `memory_status` 与 `historical_at`。

### Case 3 — No Relevant / No Conflict（PASS）

- 输入：「我今天中午吃了碗牛肉面」→ 200。
- 其余 current Memory 未被错误 historical。

---

## 8. Transaction Evidence

真实成功路径确认：一次 Candidate Write 后，old historical + new current 同时出现
（Case A / Swagger Case 2 均验证）。

Transaction Failure Path 已由 [KEEP] SQLite 单测覆盖（rollback / commit failure），
本轮不做人为破坏性测试。

---

## 9. 发现的真实 Bug

无。

验证过程中遇到的两个"现象"均非代码 bug：

1. conversation_id=1 已被软删除（deleted_at 非 NULL），`get_conversation` 返回 None → 404，
   属 F 盘既有数据状态，非 Lifecycle 引入。
2. Case B 初始输入被 LLM Validator 判为临时状态而 valid=False，属 LLM 非确定性。

---

## 10. 已知 Tech Debt

- `backend/services/llm_service.py` 在 import 时即构造 OpenAI client，缺 key 会抛 OpenAIError，
  使任何 import 链触及它的单测都必须先注入环境变量。
- DeepSeek 偶发空 response 属外部 LLM 非确定性，Read 路径已做降级处理。

---

## 11. 不在 V1 Scope 的内容

- Historical Retrieval / Query Classifier
- 第三种 memory_status
- Versioning / superseded_by / parent_id
- Merge / Physical Delete
- Alembic
- PostgreSQL / pgvector / Redis / Kafka
