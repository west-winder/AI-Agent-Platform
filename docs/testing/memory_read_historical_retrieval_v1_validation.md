# Memory Read Historical Retrieval V1 Validation

## 1. 文档目的

本文记录 AI Agent Platform 中：

Memory Read Historical Retrieval V1

的设计目标、组件边界、测试范围、真实运行结果、
修复过程中发现的问题，以及当前仍然保留的技术债。

本功能是在现有 Memory Read V2 基础上的增量演进。

原 Memory Read 行为：

Query
→ Repository
→ current Memory only
→ Dense
→ BM25
→ RRF
→ Reranker
→ Relevance Judge
→ Injector

Historical Retrieval V1 的目标是：

让 Memory Read 能够根据当前 Query，
选择不同时间范围的 Memory：

- current
- historical
- both

同时保持现有 Retrieval Pipeline 的组件边界，
避免让 Dense / BM25 / RRF / Reranker 等底层检索组件
感知 Memory Lifecycle。


---

# 2. 背景问题

Memory Lifecycle V1 引入：

- current
- historical

两种 Memory 状态。

其中：

current：
仍然代表用户当前状态。

historical：
曾经成立，
但现在已经不再代表用户当前状态。

原 Memory Read Repository 默认只读取：

memory_status == current

该行为能够避免 historical Memory 污染普通回答，
但导致系统无法回答：

- 我以前主要在学什么？
- 我曾经做过什么方向？
- 我的学习方向发生过什么变化？
- 我为什么从一个方向转向另一个方向？

因此需要增加：

Historical Retrieval。


---

# 3. 设计目标

Historical Retrieval V1 只解决：

Query
→ 判断 Retrieval Scope
→ 根据 Scope 构造 Memory Corpus
→ 使用现有 Retrieval Pipeline

支持三种 Scope：

## current

只允许 current Memory
进入 Retrieval Corpus。

例如：

我现在主要在学什么？


## historical

只允许 historical Memory
进入 Retrieval Corpus。

例如：

我以前主要在学什么？


## both

允许：

current
+
historical

同时进入 Retrieval Corpus。

例如：

我从以前到现在最大的变化是什么？

或者：

我为什么从 AI Agent 转向安卓开发？


---

# 4. 明确不做的内容

Historical Retrieval V1 不实现：

- Timeline
- valid_from / valid_to
- Memory Version Graph
- Temporal Reranking
- 时间权重
- latest always wins
- 历史 Memory 恢复为 current
- 历史 Memory 参与正常 Memory Write
- 自动修复旧数据库中的 Lifecycle 状态
- 完整 Goal Supersession 语义

Historical Retrieval V1 只解决：

“哪些生命周期状态的 Memory
应该进入本次 Retrieval Corpus”。


---

# 5. 最终 Pipeline

最终 Memory Read 数据流：

Query
    ↓
MemoryQueryScopeJudge
    ↓
MemoryQueryScopeDecision
    ↓
Repository
    ↓
Memory[]
    ↓
Dense Retrieval
    +
BM25 Retrieval
    ↓
RRF
    ↓
Hybrid Candidate Pool
    ↓
Reranker
    ↓
Top-K
    ↓
MemoryRelevanceCandidate[]
    ↓
MemoryRelevanceJudge
    ↓
JudgeDecision[]
    ↓
MemoryInjectionItem[]
    ↓
MemoryInjector
    ↓
MemoryReadResult


其中：

MemoryReader 只负责：

- orchestration
- data adaptation
- component wiring

MemoryReader 不负责解释：

- current
- historical
- both
- memory_status 的业务含义


---

# 6. Query Scope Contract

新增：

backend/memory/memory_read/memory_query_scope.py

定义：

- MEMORY_QUERY_SCOPE_CURRENT
- MEMORY_QUERY_SCOPE_HISTORICAL
- MEMORY_QUERY_SCOPE_BOTH
- ALLOWED_MEMORY_QUERY_SCOPES
- MemoryQueryScopeDecision


MemoryQueryScopeDecision：

- scope
- reason
- source


source 支持：

rule：
由第一层词法规则判断。

llm：
第一层无法可靠判断，
由 LLM 判断。

fallback：
LLM 调用或返回失败，
降级为 current。


---

# 7. Query Scope Judge 双层设计

新增：

backend/memory/memory_read/memory_query_scope_judge.py


采用两层判断。


## Layer 1：Cheap Lexical Rules

第一层使用高置信度词法规则。

目标不是覆盖所有 Query，
而是：

- 快
- 便宜
- deterministic
- 高 Precision

典型 historical marker：

- 以前
- 曾经
- 过去
- 之前
- 当时
- 最开始
- 起初
- 当初

典型 current marker：

- 现在
- 当前
- 目前
- 现阶段
- 如今
- 眼下

明确 both pattern：

- 从以前到现在
- 从过去到现在
- 以前和现在
- 过去和现在
- 和以前相比
- 与过去相比

等。


第一层设计原则：

宁可不判断，
也不要低置信度误判。


例如：

我现在想知道我以前主要学什么？

同时包含：

现在
以前

但不能直接判断 both。

第一层返回：

None

交给 LLM。


---

## Layer 2：LLM Scope Judge

当第一层无法可靠判断时：

Query
→ LLM
→ current / historical / both


例如：

我为什么从 AI Agent 转向安卓开发？

没有显式：

以前
过去
现在
当前

但语义上需要：

过去 AI Agent 状态
+
当前安卓状态

因此 LLM 应判断：

both。


---

# 8. Scope Failure Contract

Scope Judge 属于 Memory Read 的增强能力。

Historical Retrieval 引入之前，
系统原始行为就是：

current only。

因此所有 Scope LLM Failure：

- 调用异常
- 空响应
- 非字符串
- 非法 JSON
- JSON 非 object
- scope 缺失
- 非法 scope

统一：

fallback → current


这是当前 Historical Retrieval V1
的 Graceful Degradation 策略。


目标：

即使新的 Scope 判断能力失败，
普通 Memory Read 仍然保持旧版本的安全行为。


---

# 9. Repository Contract

Memory Repository 从：

get_memories_for_read(
    db,
    user_id
)

演进为：

get_memories_for_read(
    db,
    user_id,
    scope=current
)


保留默认参数：

scope=current

因此旧调用仍然保持：

current only

保证 Backward Compatibility。


Repository 对 Scope 的解释：

current：
memory_status == current

historical：
memory_status == historical

both：
memory_status IN (
    current,
    historical
)


both 明确列出两种允许状态，
而不是简单取消 status filter。

原因：

如果以后新增：

- archived
- deleted
- invalid

等状态，

不应该自动进入 both Corpus。


---

# 10. 关注点分离

Historical Retrieval V1 明确保持以下边界。


## ChatService

不知道：

- Query Scope
- current
- historical
- both
- MemoryQueryScopeJudge

ChatService 仍然只调用：

MemoryReader


## MemoryReader

知道：

存在 Scope Decision。

负责：

Scope Judge
→ decision.scope
→ Repository

但不解释：

current
historical
both

并且不包含：

if scope == ...
if memory_status == ...


## Repository

负责：

scope
→ Database Retrieval Corpus


Repository 不理解自然语言 Query。


## Dense / BM25 / RRF / Reranker

完全不知道：

- Memory Lifecycle
- current
- historical
- both
- memory_status


## MemoryRelevanceJudge

知道：

Memory 自身的：

memory_status

但不知道：

Query Scope。


## MemoryInjector

知道：

Memory 自身的：

memory_status

但不知道：

Query Scope。


---

# 11. Injector Status Preservation

Historical Retrieval 初版接通后发现：

Memory ORM 中虽然一直保存：

content
+
memory_status

但 MemoryReader 在进入 Injector 前曾转换为：

list[str]

导致：

memory_status 丢失。


这会造成：

both 场景中最终 LLM 只能看到：

用户正在学习 AI Agent
用户正在学习安卓开发

但不知道：

哪条是 current
哪条是 historical。


因此新增：

MemoryInjectionItem

包含：

- content
- memory_status


MemoryReader 负责：

Memory ORM
→ MemoryInjectionItem


Injector 最终输出：

memory_status: current

或：

memory_status: historical


并在 Context Header 中明确：

historical Memory：

曾经成立，
但现在已经不再代表用户当前状态。

historical 只能作为：

- 过去状态
- 历史背景
- 状态变化过程

使用。


---

# 12. Relevance Judge Status Preservation

真实 Historical Retrieval 测试中发现了第二个重要问题。


数据库存在：

content:
用户当前正在学习AI Agent

memory_status:
historical


旧 MemoryRelevanceJudge 输入只有：

content


因此 Judge 实际看到：

用户当前正在学习AI Agent

却不知道：

memory_status=historical


在 Query：

我以前主要学什么？

中，

Judge 曾因为 content 中出现：

“当前”
“正在”

而错误地认为：

该 Memory 属于当前状态，
不应该用于历史问题。


这不是 Repository 问题。

Repository 已正确返回 historical Memory。

问题发生在：

Memory ORM
→ list[str]
→ Relevance Judge

的数据边界。


---

# 13. Relevance Judge Contract 修复

新增：

MemoryRelevanceCandidate

包含：

- content
- memory_status


Relevance Judge 输入从：

texts: list[str]

升级为：

candidates: list[MemoryRelevanceCandidate]


实际发送给 LLM：

{
    "index": 0,
    "content": "用户当前正在学习AI Agent",
    "memory_status": "historical"
}


Prompt 明确：

memory_status 是系统当前维护的
Lifecycle 状态。

它比 content 中遗留的：

- 当前
- 正在
- 目前
- 现在

等时间措辞更权威。


因此：

content:
用户当前正在学习AI Agent

memory_status:
historical

应该理解为：

用户过去某个阶段正在学习 AI Agent，

而不是：

用户现在仍然正在学习 AI Agent。


同时：

historical
不等于：

selected=true。


Relevance Judge 仍然需要结合：

Query
+
content
+
memory_status

判断该 Memory
是否真的对当前回答有帮助。


---

# 14. Deterministic Tests

## [KEEP]

新增：

tests/memory/test_memory_read_historical_scope.py

共：

49 tests

结果：

49 / 49 PASS


覆盖：

- Scope Rule current
- Scope Rule historical
- Scope Rule both
- ambiguous Query → LLM
- implicit transition → LLM
- Scope LLM 合法结果
- Scope LLM Failure fallback
- Scope Input Contract
- Repository current
- Repository historical
- Repository both
- user_id isolation
- Repository backward compatibility
- MemoryReader Scope orchestration
- Early Return scope preservation
- Injector status preservation
- Architecture boundary


---

## [KEEP]

新增：

tests/memory/test_memory_relevance_judge_status_contract.py

共：

43 tests

结果：

43 / 43 PASS


覆盖：

- historical status preservation
- current status preservation
- mixed candidate status
- Prompt lifecycle semantics
- Candidate input validation
- Judge output validation
- index contract
- coverage contract
- Reader → Judge data adaptation
- Architecture boundary


---

# 15. Existing Regression

以下 deterministic regression
在 Historical Retrieval 修改后仍然通过：

tests/memory/test_memory_lifecycle_v1.py
→ PASS

tests/memory/test_chat_service_memory_regression.py
→ PASS

tests/memory/test_memory_write_state_change_regression.py
→ PASS


说明 Historical Retrieval
没有破坏现有：

- Memory Lifecycle
- ChatService Memory integration
- Memory Write state-change behavior


---

# 16. Manual Tests

## [MANUAL KEEP]

tests/memory/test_memory_reader_v2_real_retrieval.py

结果：

PASS


使用：

- real Qwen3 Embedding
- real BM25
- real bge-reranker-v2-m3

使用 Fake：

- Repository
- Relevance Judge
- Injector

主要验证：

真实 Retrieval / Reranker
仍能适配新的 Memory Contract。


---

## [MANUAL KEEP]

tests/memory/test_memory_reader_v2_e2e.py

结果：

PASS

5 / 5 cases。


包含真实：

- SQLite Read
- Embedding
- BM25
- RRF
- Reranker
- DeepSeek Relevance Judge
- MemoryInjector


运行前后数据库未变化。


---

## [MANUAL KEEP]

tests/memory/manual_historical_retrieval_v1.py

用于真实 Historical Retrieval 主链路验证。


真实测试验证：

Case 1：

我现在主要在学什么？

→ current
→ source=rule


Case 2：

我以前主要在学什么？

→ historical
→ source=rule


Case 3：

我从以前到现在最大的变化是什么？

→ both
→ source=rule


Case 4：

我为什么从 AI Agent 转向安卓开发？

→ both
→ source=llm


其中 Case 4 不包含显式：

以前
过去
现在
当前
目前

证明：

第二层 LLM 能处理隐式时间变化语义。


---

# 17. Swagger E2E Validation

最终通过 Swagger UI
使用独立 Conversation
验证真实：

POST /chat


测试时数据库基线：

user_id=1

总 Memory：

15


historical：

ID 9
用户最近在学C++

ID 11
用户当前正在学习AI Agent

ID 12
用户希望未来成为AI Agent工程师


其余：

current


---

## Case 1

Query：

我以前主要在学什么？


结果：

回答正确使用 historical Memory：

- AI Agent
- C++

并明确说明：

这些属于 historical，
不能说明用户现在仍然在学习这些内容。


该 Case 验证：

Relevance Judge
能够正确理解：

content:
用户当前正在学习AI Agent

memory_status:
historical


修复前出现的：

因为 content 含“当前”
而误判为当前状态

的问题已消失。


---

## Case 2

Query：

我现在想知道我以前主要在学什么？


结果：

最终回答仍然围绕历史状态：

- C++
- AI Agent

没有把 Query
错误理解为简单的：

当前 vs 过去

比较。


说明：

同时出现 historical/current marker
但没有明确 both pattern 时，

规则层不强行判断，

交由 LLM 判断的策略有效。


---

## Case 3

Query：

我为什么从 AI Agent 转向安卓开发？


结果：

回答能够同时区分：

historical：

- 曾学习 AI Agent
- 曾希望成为 AI Agent 工程师

current：

- 已经不再学习 AI Agent
- 转向安卓应用开发
- 希望成为安卓软件开发工程师


对于：

为什么转向

系统明确指出：

现有 Memory 只能确认
发生了方向变化，

并没有足够 Memory
说明具体原因。


没有根据缺失信息自行编造原因。


该 Case 验证：

implicit both Query
+
historical/current status
+
final answer

完整主链路正常。


---

# 18. Swagger 数据库副作用验证

最终三次 Swagger E2E 测试：

每个 Case
均使用独立 Conversation。


测试前：

Memory count = 15

historical IDs：

9
11
12


测试后：

Memory count = 15

historical IDs：

9
11
12


没有：

- 新 Memory
- current → historical
- historical → current


因此本轮最终 Swagger
没有造成 Memory Write 副作用。


---

# 19. 已发现但暂不处理的技术债

## 19.1 Question Presupposition

早期 Swagger 测试中：

Query：

我为什么从 AI Agent 转向安卓开发？


测试后曾观察到：

ID 12

用户希望未来成为AI Agent工程师

从：

current

变为：

historical。


该变化发生在 `/chat`
同时运行 Memory Write 的过程中。


可能原因：

问句中的隐含前提：

“已经从 AI Agent 转向安卓”

被 Memory Write
当作新的状态信息处理。


当前仅记录为：

Question Presupposition /
Interrogative Memory Write

技术债。


本阶段不修复。


不能简单采用：

“问句不写 Memory”

因为：

我最近开始学 Rust 了，
你觉得我应该怎么学？

虽然是问句，

但其中明确包含
值得写入 Memory 的新状态。


后续需要区分：

明确陈述的新信息

与：

用于提问的隐含前提。


---

## 19.2 Legacy Lifecycle Data

数据库中仍存在
部分旧 Memory 状态语义不一致。


例如：

用户长期学习Python后端开发

用户正在使用Python进行后端开发学习

用户不打算学习Python后端开发了

用户准备转向Java后端开发

目前都可能同时保持 current。


原因：

Lifecycle Migration
对历史旧数据采用：

默认 current

而没有进行语义回填。


该问题属于：

Legacy Data Quality /
Lifecycle Backfill

不属于 Historical Retrieval V1。


---

## 19.3 Goal Supersession

Memory Lifecycle 当前仍不具备：

Primary Goal
Exclusive Goal
Goal Supersession

等明确语义。


例如两个职业目标是否互斥：

AI Agent 工程师

Android 软件开发工程师

当前不能仅依赖普通 conflict 判断
稳定处理。


该问题属于：

Memory Lifecycle 后续能力，

不在 Historical Retrieval V1 范围。


---

# 20. Engineering Conclusions

Historical Retrieval V1
最终形成了以下边界：


Query Scope Judge

负责：

Query
→ Retrieval Scope


Repository

负责：

Scope
→ Database Corpus


MemoryReader

负责：

Component Orchestration
+
Data Adaptation


Dense / BM25 / RRF / Reranker

负责：

Retrieval / Ranking

完全不知道 Lifecycle。


MemoryRelevanceJudge

负责：

Query
+
Memory Content
+
Memory Status
→ USE / REJECT


MemoryInjector

负责：

Selected Memory
+
Memory Status
→ Final Memory Context


ChatService

继续只依赖：

MemoryReader

不知道：

Historical Retrieval
内部实现细节。


---

# 21. Final Validation Result

Historical Retrieval V1：

Implementation
→ PASS

Scope Rule
→ PASS

Scope LLM
→ PASS

Scope Failure Fallback
→ PASS

Repository Scope Filtering
→ PASS

User Isolation
→ PASS

MemoryReader Orchestration
→ PASS

Relevance Status Preservation
→ PASS

Injector Status Preservation
→ PASS

Architecture Boundary
→ PASS

Deterministic Tests
→ PASS

Manual Real Retrieval
→ PASS

Manual E2E
→ PASS

Swagger HTTP E2E
→ PASS

Final Swagger DB Side Effect Check
→ PASS


结论：

Memory Read Historical Retrieval V1
验证完成。

当前可以进入：

文档收尾
→ Git Review
→ Commit

阶段。