# AI Agent Platform

基于 **FastAPI + PostgreSQL + DeepSeek + Gradio** 构建的 AI Agent 应用平台。

当前版本：

**v0.3 — PostgreSQL Runtime Baseline + Backend / LLM I/O Foundation**

---

## Project Purpose

AI Agent Platform 不是简单的 Chatbot Demo。

项目目标是通过一个持续演进的真实工程，学习和展示完整的 AI Agent 工程能力：

```text
Backend Engineering
LLM Application Engineering
Memory
RAG
Tool Calling
Agent Loop
Workflow
Observability / Evaluation
Frontend
Deployment
```

项目重点不是训练大语言模型，而是围绕现有 LLM、Embedding、Reranker 等模型，构建完整的 AI 应用工程系统。

项目开发过程中重点关注：

- Component Boundary
- Separation of Concerns
- Dependency Injection
- Runtime Trace
- Failure Contract
- Graceful Degradation
- Deterministic Regression Test
- Incremental Architecture Evolution

---

## Current Stage

```text
v0.3

Stage 0 — Memory + PostgreSQL
COMPLETE

Stage 1 — Backend + LLM I/O Foundation
COMPLETE

Stage 2 — Unified RAG
NEXT
```

### v0.1 — Basic Agent Chat Platform

- Agent
- Conversation
- Message
- Agent Snapshot
- DeepSeek
- Gradio

**Completed**

### v0.2 — Long-term Memory System

- Memory Extraction
- Memory Validation
- Exact Dedup
- Related Memory Retrieval
- Relationship Judge
- Memory Lifecycle
- Hybrid Memory Read
- Dense Retrieval
- BM25
- RRF
- Reranker
- Relevance Judge
- Memory Injection
- Historical Retrieval
- current / historical / both Query Scope

**Completed**

### v0.3 — PostgreSQL Runtime Baseline + Backend / LLM I/O Foundation

- PostgreSQL 18 + psycopg3 runtime baseline
- Async LLM I/O
- SSE Streaming
- Structured Output
- Timeout
- Retry
- Error Handling V1
- Basic Logging V1

**Completed**

---

## 当前架构

```text
Gradio Frontend
        |
        v
FastAPI Router
        |
        v
Service Layer
        |
        +----------------------+
        |                      |
        v                      v
Conversation / Message      Memory System
        |                      |
        v                      v
SQLAlchemy ORM           Memory Read / Write
        |                      |
        +----------+-----------+
                   |
                   v
              LLM Service
                   |
                   v
        DeepSeek OpenAI-compatible API
```

LLM Service 是所有外部 LLM 调用的唯一出口，统一提供：

```text
Async
SSE Streaming
Structured Output
Timeout
Retry
Error Handling
```

Memory System 内部进一步拆分为：

```text
                    User Message / Query
                           |
             +-------------+-------------+
             |                           |
             v                           v
        Memory Write                  Memory Read
             |                           |
             v                           v
         Extractor                Query Scope Judge
             |                           |
         Validator          current / historical / both
             |                           |
        Exact Dedup                    Repository
             |                           |
     Related Retrieval          +------+------+
             |                  |             |
      Relationship Judge      Dense          BM25
             |                  |             |
        Lifecycle                +------+------+
             |                           |
             v                           v
        PostgreSQL                    RRF
                                         |
                                     Reranker
                                         |
                                      Top-K
                                         |
                               Relevance Judge
                                         |
                                      Injector
                                         |
                                         v
                                  Memory Context
```

---

## 技术栈

### Backend

- Python 3.12
- FastAPI
- SQLAlchemy
- Pydantic
- Uvicorn

### Database

- PostgreSQL 18
- psycopg3（SQLAlchemy driver：`postgresql+psycopg`）

### LLM

- DeepSeek API
- OpenAI-compatible SDK（AsyncOpenAI）

### Embedding

- Qwen3-Embedding-0.6B

主要用于：

- Memory Dense Retrieval
- Memory Similarity Search

### Reranker

- BAAI bge-reranker-v2-m3

主要用于：

- Hybrid Retrieval Candidate Reranking

### Retrieval

- Dense Retrieval
- BM25
- jieba
- bm25s
- Reciprocal Rank Fusion（RRF）

### Frontend

- Gradio

当前已适配 Gradio 6.x messages format。

---

## 项目结构

```text
AI_Agent_Platform
│
├── backend
│   │
│   ├── config
│   │   └── logging_config.py
│   │
│   ├── database
│   │   ├── database.py
│   │   └── migrations
│   │
│   ├── embedding
│   │   └── embedder.py
│   │
│   ├── exceptions
│   │   └── external_exceptions.py
│   │
│   ├── memory
│   │   │
│   │   ├── memory_read
│   │   │   ├── memory_reader.py
│   │   │   ├── memory_repository.py
│   │   │   ├── memory_query_scope.py
│   │   │   ├── memory_query_scope_judge.py
│   │   │   ├── dense_retriever.py
│   │   │   ├── memory_relevance_judge.py
│   │   │   └── memory_injector.py
│   │   │
│   │   └── memory_write
│   │       ├── memory_pipeline.py
│   │       ├── memory_extractor.py
│   │       ├── memory_validator.py
│   │       ├── memory_deduplicator.py
│   │       ├── memory_similarity.py
│   │       ├── memory_relationship_judge.py
│   │       └── memory_lifecycle.py
│   │
│   ├── models
│   ├── reranking
│   │   └── reranker.py
│   │
│   ├── retrieval
│   │   ├── bm25.py
│   │   └── rrf.py
│   │
│   ├── routers
│   ├── schemas
│   ├── services
│   └── main.py
│
├── frontend
│   └── app.py
│
├── tests
│   ├── async
│   ├── frontend
│   ├── logging
│   └── memory
│
├── docs
│   └── testing
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 已实现功能

### 1. Agent System

支持创建和管理不同类型的 AI Agent。

Agent 包含：

- Name
- System Prompt
- Created At
- Updated At

例如：

```text
科研助手

你是一名科研领域专家，
负责帮助用户进行论文分析。
```

---

### 2. Conversation System

实现独立 Conversation 管理。

每个 Conversation 可以关联：

- User
- Agent
- Message

支持：

- 创建新聊天
- 删除聊天
- 查看聊天列表
- 加载历史 Conversation
- Conversation Message 持久化

---

### 3. Agent Snapshot

创建 Conversation 时保存 Agent 快照。

例如：

```json
{
  "name": "科研助手",
  "system_prompt": "你是一名科研助手"
}
```

即使之后修改 Agent：

历史 Conversation 仍然能够保持创建时的 Agent 配置。

这样可以避免历史会话随着 Agent 配置变化而产生行为漂移。

---

### 4. Message System

Conversation 中的消息会持久化保存。

核心结构包括：

```text
Message

id
conversation_id
role
content
created_at
```

当前主要支持：

```text
user
assistant
```

两种消息角色。

---

### 5. DeepSeek LLM Integration

LLM 调用通过 Service Layer 统一封装。

API Key 使用环境变量管理。

密钥不会直接写入代码仓库。

---

## LLM I/O Foundation（Stage 1）

v0.3 完成 Backend + LLM I/O Foundation。

所有能力集中在 `llm_service.py` 单一出口：

```text
call_llm            → 一次性 Chat Completion
stream_llm          → SSE Streaming Generator
call_llm_structured → Structured Output（Pydantic Model）
```

### Async

全部 LLM 调用基于 AsyncOpenAI，async / await 贯穿 Service 与 Router。

---

### SSE Streaming

Chat 支持 Streaming 模式：

```text
POST /chat          → 一次性返回
POST /chat/stream   → SSE 逐块返回
```

Streaming 过程中已生成的内容不会因为后续错误而丢失。

---

### Structured Output

需要结构化输出的场景（Memory Extractor / Validator / Judge 等）
统一使用 Structured Output：

```text
LLM Response
    |
    v
Pydantic Model 解析
    |
    v
类型安全的结构化结果
```

不再依赖手工 JSON 字符串解析。

---

### Timeout

超时策略分层配置：

```text
Connect Timeout
Read Timeout
Write Timeout
Pool Timeout
Chat Overall Timeout
Structured Overall Timeout
```

均为环境变量可配置。

---

### Retry

LLM Client 显式配置最大重试次数。

重试策略在 Service 层统一声明，不散落在各调用点。

---

### Error Handling V1

统一异常层级：

```text
Exception
    |
    v
ExternalServiceError
    ├── ExternalTimeoutError
    └── ExternalRequestError
```

OpenAI SDK 异常在 `llm_service` 边界统一翻译，
不会泄漏到 Router / Frontend 层。

Router 层固定映射：

```text
ExternalTimeoutError    → 504
ExternalRequestError    → 500
ExternalServiceError    → 503
```

---

### Basic Logging V1

```text
业务模块 logging.getLogger(__name__)
        |
        v
propagation → root
        |
        v
统一格式输出
```

Memory 读写边界（Memory Boundary）统一记录 WARNING 级别日志，
包含：

- operation
- user_id / conversation_id
- fallback 行为
- 原始异常信息

日志不改变任何 Failure Contract，只增加可观测性。

---

## Long-term Memory System

v0.2 的核心能力是用户长期 Memory 系统。

Memory 并不是简单地把聊天记录直接写入数据库，而是被拆分为：

```text
Memory Write
Memory Read
Memory Lifecycle
Historical Retrieval
```

四个主要部分。

---

## Memory Write

Memory Write 负责从用户消息中提取真正值得长期保存的信息。

当前流程：

```text
User Message
    |
    v
Memory Extractor
    |
    v
Memory Validator
    |
    v
Exact Dedup
    |
    v
Related Memory Retrieval
    |
    v
Relationship Judge
    |
    v
Memory Lifecycle
    |
    v
Persistence
```

---

### Memory Extractor

Memory Extractor 使用 LLM 从用户消息中提取：

```text
MemoryCandidate
```

例如用户输入：

```text
我最近开始学习 LangGraph，
准备以后深入研究 Agent Workflow。
```

可能提取：

```text
用户最近开始学习 LangGraph

用户计划深入学习 LangGraph 的 Agent Workflow
```

Extractor 基于 Structured Output 获取类型安全的结果。

---

### Memory Validator

Validator 对 Candidate 进行第二层准入判断。

包含两个阶段。

#### Deterministic Rule Validation

检查：

- `content` 非空
- 长度合法
- `memory_type` 合法

#### LLM Semantic Validation

判断 Candidate 是否真正具有长期 Memory 价值。

当前支持识别：

- Current State
- State Change
- State Termination
- State Transition
- State Restart

例如：

```text
我最近没学 C++ 了
```

可以被识别为新的状态变化，而不是因为存在否定表达就直接丢弃。

---

### Exact Dedup

在写入前首先进行 Exact Dedup。

如果 Candidate 与已有 current Memory 完全重复：

```text
Candidate
    |
    v
Duplicate
    |
    v
Skip Persistence
```

避免相同 Memory 被反复写入数据库。

---

### Related Existing Memory Retrieval

如果不存在 Exact Duplicate：

系统继续寻找可能与 Candidate 相关的 current Memory。

这些 Memory 会交给后续 Relationship Judge。

---

### Relationship Judge

LLM Relationship Judge 判断新 Candidate 与已有 Memory 的关系。

当前支持：

```text
duplicate
conflict
related
new
```

Relationship Judge 只负责：

```text
判断关系
```

它不直接修改数据库。

---

## Memory Lifecycle

Memory Lifecycle V1 当前使用两个状态：

```text
current
historical
```

### current

表示：

该 Memory 仍然代表用户当前状态。

### historical

表示：

该 Memory 曾经成立，但现在已经不再代表用户当前状态。

例如：

```text
旧 Memory：

用户最近在学习 C++

        ↓

用户：

我已经不学习 C++ 了

        ↓

旧 Memory：
historical

新 Memory：

用户已停止学习 C++
current
```

Lifecycle V1 明确禁止：

```text
historical → current
```

如果用户未来重新开始学习 C++：

系统会创建新的 current Memory，

而不是重新恢复旧 historical Memory。

---

### Lifecycle Transaction Boundary

对于一个 Memory Candidate：

```text
Mark Conflicting Memories Historical
            +
Save New Candidate
            |
            v
          Commit
```

属于同一个数据库事务。

如果中间任何一步失败：

```text
Rollback
```

避免出现：

```text
旧 Memory 已经变成 historical
但新的 current Memory 没有保存成功
```

这种不一致状态。

---

## Memory Read

Memory Read V2 当前采用：

**Hybrid Retrieval Pipeline**

```text
Query
   |
   +-------------------+
   |                   |
   v                   v
Dense Retrieval      BM25
   |                   |
   +---------+---------+
             |
             v
            RRF
             |
             v
       Candidate Pool
             |
             v
         Reranker
             |
             v
           Top-K
             |
             v
     Relevance Judge
             |
             v
          Injector
             |
             v
       Memory Context
```

---

### Dense Retrieval

使用：

```text
Qwen3-Embedding-0.6B
```

将：

```text
Query
Memory
```

转换为向量。

然后通过 Cosine Similarity 计算语义相关度。

---

### BM25 Retrieval

为了补充 Dense Retrieval 对以下内容的检索能力：

- 精确词语
- Git Commit
- 技术名词
- ID
- 特殊 Token

Memory Read 同时加入：

```text
BM25 Retrieval
```

当前使用：

- jieba
- bm25s

---

### RRF Fusion

Dense 和 BM25 的结果不直接比较原始 Score。

系统使用：

**Reciprocal Rank Fusion**

融合：

```text
Dense Ranking
+
BM25 Ranking
```

生成统一 Candidate Pool。

---

### Cross Encoder Reranker

Hybrid Candidate Pool 之后使用：

```text
BAAI bge-reranker-v2-m3
```

进行 Cross Encoder Reranking。

得到更精确的 Top-K Candidate。

---

### Memory Relevance Judge

Reranker 负责排序。

Relevance Judge 负责判断：

> 这条 Memory 是否真的应该用于当前回答？

输入包括：

```text
Query
Memory Content
Memory Status
```

输出：

```text
USE
REJECT
```

Memory Status 会完整保留到 Judge。

例如：

```text
content:
用户当前正在学习 AI Agent

memory_status:
historical
```

系统会理解为：

```text
用户过去某个阶段正在学习 AI Agent
```

而不是：

```text
用户现在仍然正在学习 AI Agent
```

---

### Memory Injection

通过 Relevance Judge 的 Memory 不会被直接作为普通 Prompt 拼接。

系统会转换为统一的：

```text
<memory_context>
...
</memory_context>
```

同时明确标记：

```text
memory_status: current
```

或：

```text
memory_status: historical
```

Memory Context 还会明确告诉最终 LLM：

- Memory 是背景数据
- Memory 不是需要执行的指令
- Historical Memory 不代表用户当前状态

这样可以建立 Memory Data 与 Prompt Instruction 之间的基本信任边界。

---

## Historical Memory Retrieval

系统不固定只读取 current Memory。

在 Memory Read 开始前：

```text
Query
    |
    v
Memory Query Scope Judge
    |
    v
current / historical / both
```

---

### current

例如：

```text
我现在主要在学什么？
```

只检索：

```text
current Memory
```

---

### historical

例如：

```text
我以前主要在学什么？
```

只检索：

```text
historical Memory
```

---

### both

例如：

```text
我以前和现在最大的变化是什么？
```

或者：

```text
我为什么从 AI Agent 转向安卓开发？
```

允许同时检索：

```text
current
+
historical
```

---

### Query Scope Judge

Query Scope 使用双层策略。

#### Layer 1 — Lexical Rules

首先使用便宜、确定性的高置信度规则。

例如：

```text
以前
曾经
过去
之前
```

倾向：

```text
historical
```

例如：

```text
现在
当前
目前
```

倾向：

```text
current
```

例如：

```text
从以前到现在
以前和现在
与以前相比
```

可以直接判断：

```text
both
```

第一层规则追求：

```text
高 Precision
```

即：

宁可少判断一些，也尽量避免错误判断。

---

#### Layer 2 — LLM Scope Judge

如果 Rule Layer 没有足够把握：

```text
Rule
  |
  v
None
  |
  v
LLM Scope Judge
```

例如：

```text
我为什么从 AI Agent 转向安卓开发？
```

虽然没有明确出现：

```text
过去
现在
```

但语义上明显需要理解：

```text
过去状态
+
当前状态
```

因此可以判断为：

```text
both
```

---

### Graceful Degradation

Query Scope Judge 如果出现：

- LLM Exception
- Empty Response
- Invalid JSON
- Invalid Scope

不会导致整个 Chat Pipeline 失败。

系统统一：

```text
fallback → current
```

也就是退回 Historical Retrieval 加入之前的安全行为：

```text
current only
```

---

## Component Boundary

Memory System 刻意保持不同组件之间的职责分离。

```text
Query Scope Judge
→ 判断 Query Scope

Repository
→ 根据 Scope 获取 Memory Corpus

Dense / BM25
→ Candidate Retrieval

RRF
→ Ranking Fusion

Reranker
→ Candidate Reranking

Relevance Judge
→ USE / REJECT

Injector
→ 构造 Memory Context

MemoryReader
→ Orchestration + Data Adaptation

ChatService
→ 只调用 MemoryReader

llm_service
→ 所有 LLM 调用的唯一出口
```

因此：

```text
ChatService
```

不知道：

```text
current
historical
both
```

Dense / BM25 / RRF / Reranker 也完全不知道 Memory Lifecycle。

这样可以减少组件之间的耦合。

未来修改某一个组件时，可以尽量避免影响整个 Memory Pipeline。

---

## Testing

tests/ 按领域组织：

```text
tests/
│
├── async      → LLM I/O 契约（Timeout / Retry / Error）
├── frontend   → 前端错误语义契约
├── logging    → 日志契约
└── memory     → Memory Read / Write / Lifecycle
```

测试分为三类：

```text
[KEEP]
[MANUAL KEEP]
[TEMP]
```

---

### [KEEP]

Deterministic Regression Test。

特点：

- 不依赖真实 LLM
- 不加载昂贵真实模型
- 使用 Fake Component
- 可长期运行作为回归测试

例如：

```text
test_llm_error_contract.py

test_llm_retry_contract.py

test_llm_timeout_contract.py

test_frontend_chat_error_contract.py

test_chat_service_logging_contract.py

test_memory_read_historical_scope.py

test_memory_relevance_judge_status_contract.py

test_memory_lifecycle_v1.py

test_memory_write_state_change_regression.py
```

---

### [MANUAL KEEP]

真实集成测试。

可能使用：

- PostgreSQL
- Qwen3 Embedding
- BM25
- bge-reranker
- DeepSeek

例如：

```text
test_memory_reader_v2_real_retrieval.py

test_memory_reader_v2_e2e.py

manual_historical_retrieval_v1.py
```

---

### [TEMP]

阶段性诊断或实验脚本。

主要用于：

- Debug
- Runtime Trace
- 问题定位
- 临时验证

验证结束后可以删除。

---

## Validation Documentation

重要验证过程记录在：

```text
docs/testing/
```

其中包括：

```text
memory_read_v2_validation.md

memory_lifecycle_v1_validation.md

memory_read_historical_retrieval_v1_validation.md

structured_output_query_scope_v1.md
```

用于记录：

- Design Contract
- Test Scope
- Regression Evidence
- Real Bugs
- Engineering Conclusions
- Known Technical Debt

---

## Gradio Frontend

项目提供基础 Gradio Chat UI。

支持：

- Conversation 列表
- 新建 Conversation
- 删除 Conversation
- 加载历史消息
- 发送消息
- AI Response 展示
- Streaming / Non-Streaming 双模式

当前已适配：

```text
Gradio 6.x
```

Chatbot 使用统一 messages format：

```json
[
  {
    "role": "user",
    "content": "你好"
  },
  {
    "role": "assistant",
    "content": "你好，有什么可以帮你？"
  }
]
```

Backend Message 与 Gradio Chat Message 会在 Frontend Adapter 层完成格式转换。

因此：

```text
Backend API
```

不会直接依赖 Gradio 的数据格式。

---

## 运行方式

### 1. 克隆项目

```bash
git clone <your_repository_url>

cd AI-Agent-Platform
```

---

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
```

本项目使用 Python 3.12。

---

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

### 4. 准备 PostgreSQL

准备一个可访问的 PostgreSQL 数据库（当前 baseline：PostgreSQL 18）。

---

### 5. 配置环境变量

复制：

```text
.env.example
```

为：

```text
.env
```

按需配置：

```text
DEEPSEEK_API_KEY       DeepSeek API Key
DEEPSEEK_BASE_URL      DeepSeek OpenAI-compatible Base URL
DEFAULT_MODEL          默认对话模型

EMBEDDING_MODEL_PATH   本地 Embedding 模型路径（Qwen3-Embedding-0.6B）
RERANKER_MODEL_PATH    本地 Reranker 模型路径（BAAI/bge-reranker-v2-m3）

DB_HOST                PostgreSQL 主机
DB_PORT                PostgreSQL 端口
DB_NAME                数据库名
DB_USER                数据库用户
DB_PASSWORD            数据库密码
```

密钥与本地模型路径只存在于 `.env`，

`.env` 已被 `.gitignore` 排除，不会进入仓库。

---

### 6. 启动 FastAPI

```powershell
python -m uvicorn backend.main:app --reload
```

Swagger UI：

```text
http://127.0.0.1:8000/docs
```

---

### 7. 启动 Gradio

```powershell
python frontend/app.py
```

默认地址：

```text
http://127.0.0.1:7860
```

---

## 当前限制

当前仍然是学习和工程实践版本。

暂未实现：

- 用户登录认证
- JWT / RBAC
- pgvector（当前 Dense Retrieval 为进程内暴力检索）
- Unified RAG Knowledge Base
- 文件上传与文档解析
- Tool Calling
- Agent Loop
- LangGraph Workflow
- Multi-Agent Collaboration
- Docker Deployment
- 完整 Production Observability
- 完整 Temporal Memory Timeline
- Memory Version Graph
- Goal Supersession
- Historical Data Backfill

---

## Known Technical Debt

### Legacy Memory Lifecycle Data

早期 Memory 在 Lifecycle Migration 时统一初始化为：

```text
current
```

因此部分旧 Memory 可能存在语义冲突。

当前暂不自动进行 Historical Data Backfill。

---

### Goal Supersession

当前 Lifecycle 尚未定义：

```text
Primary Goal
Exclusive Goal
Goal Supersession
```

因此两个职业目标是否互斥，目前仍依赖普通 Relationship 判断。

---

### Question Presupposition

问句中包含的隐含前提，未来需要进一步研究是否应该进入 Memory Write。

例如：

```text
我为什么从 AI Agent 转向安卓开发？
```

其中：

```text
已经发生了从 AI Agent 转向安卓开发
```

可能只是本次提问的前提，

并不一定代表用户正在主动声明一个新的长期状态。

因此未来需要进一步区分：

```text
Explicit User Statement
```

与：

```text
Question Presupposition
```

---

## Roadmap

### v0.4 — Stage 2: Unified RAG（NEXT）

将 Memory Retrieval 中验证过的 Hybrid 架构推广为统一 RAG 能力：

- Document Upload
- Parsing
- Chunking
- Embedding
- Vector Retrieval
- Hybrid Retrieval（Dense + BM25 + RRF）
- Reranking
- Knowledge Context Injection

---

### Tool Calling

让 Agent 能够调用真实外部工具。

---

### Agent Loop / Workflow

进一步学习和实现：

- Agent Loop
- LangGraph
- Workflow State
- Multi-step Agent
- Multi-Agent Collaboration

---

### Observability / Evaluation

在 Basic Logging V1 之上继续演进：

- Structured Logging
- Tracing
- Metrics
- Evaluation

---

### Frontend / Full-stack

继续增强：

- Chat UI
- Memory Management UI
- RAG Knowledge Base UI
- Agent Configuration UI

---

### Deployment

- Docker
- Production Deployment

---

## Version

Current:

```text
v0.3
```

Core Milestone:

```text
PostgreSQL Runtime Baseline
+
Stage 1 — Backend + LLM I/O Foundation
```

---

## License

This project is for learning and engineering practice.
