# Structured Output Query Scope V1

## 1. Scope

本记录描述 AI Agent Platform 中：

```text
MemoryQueryScopeJudge
```

从旧的：

```text
Prompt JSON
→ LLM 返回 str
→ json.loads()
→ dict
→ 手工字段校验
→ MemoryQueryScopeDecision
```

迁移到：

```text
Pydantic Schema
→ Provider Structured Output
→ SDK Parse
→ MemoryQueryScopeLLMOutput
→ Business Adaptation
→ MemoryQueryScopeDecision
```

的过程、设计边界、测试证据与工程结论。

本次迁移只针对：

```text
Memory Read
→ Query Scope Judge
→ LLM Layer
```

没有修改：

- Lexical Rule
- MemoryReader scope orchestration
- Repository
- Dense Retrieval
- BM25
- RRF
- Reranker
- Relevance Judge
- Injector
- Memory Lifecycle
- Database
- Historical Retrieval Business Contract

本次属于：

```text
Implementation Migration
```

目标是：

```text
改变 LLM 输出实现方式
```

但尽量保持：

```text
Business Behavior
```

不变。

---

# 2. Why

## 2.1 原实现的问题

原 `MemoryQueryScopeJudge` 中，Prompt 会要求 LLM：

```text
严格返回 JSON
```

例如：

```json
{
  "scope": "historical",
  "reason": "用户询问过去状态"
}
```

但实际 LLM Contract 仍然是：

```text
call_llm()
→ Awaitable[str]
```

因此 Judge 必须自己承担：

```text
str validation
↓
empty check
↓
json.loads()
↓
dict validation
↓
scope extraction
↓
scope normalization
↓
scope allowed-value validation
↓
reason extraction
↓
reason validation
```

即：

```text
Prompt 要求 JSON
≠
真正的 Structured Output
```

Prompt 只是自然语言层面的要求。

它不能天然保证：

```text
JSON 语法合法
字段存在
字段类型正确
scope 只属于允许值
```

---

# 3. Structured Output Core Idea

Structured Output：

```text
结构化输出
```

本次迁移中的核心思想是：

```text
Prompt
→ 负责语义判断规则

Schema
→ 负责输出结构约束

Provider
→ 提供 Structured Output 能力

Pydantic
→ 定义并验证 Python Application Model

Business Component
→ 只依赖已经验证过的内部对象
```

因此需要区分：

```text
Semantic Correctness
语义是否判断正确

和

Structural Validity
输出结构是否合法
```

例如：

```json
{
  "scope": "current",
  "reason": "..."
}
```

即使结构完全合法，

如果用户询问的是：

```text
“我以前学过什么？”
```

那么语义仍然可能判断错误。

Structured Output 解决的主要是：

```text
输出结构可靠性
```

而不是：

```text
LLM 语义永远正确
```

---

# 4. Provider / SDK / Application Boundary

当前项目使用：

```text
DeepSeek
→ Provider

OpenAI Python SDK
→ Client SDK

backend/services/llm_service.py
→ Provider / SDK Adapter

MemoryQueryScopeJudge
→ Business Component
```

Structured Output 的 Provider-specific 细节包括：

```text
responses.parse()
text_format
output_parsed
JSON Schema
SDK Response Object
```

这些不应该泄漏到：

```text
MemoryQueryScopeJudge
MemoryReader
```

因此继续遵守已有架构边界：

```text
Business Component
↓
llm_service
↓
SDK
↓
Provider
```

---

# 5. Old Contract

迁移前：

```text
MemoryQueryScopeJudge
↓
call_llm()
↓
str
```

然后 Judge 内部执行：

```text
response type check
↓
strip()
↓
empty check
↓
json.loads()
↓
dict check
↓
data.get("scope")
↓
strip().lower()
↓
allowed scope check
↓
data.get("reason")
↓
reason validation
↓
MemoryQueryScopeDecision
```

典型逻辑：

```python
response = await call_llm(messages)

data = json.loads(response)

scope = data.get("scope")

if isinstance(scope, str):
    scope = scope.strip().lower()

if scope not in ALLOWED_MEMORY_QUERY_SCOPES:
    return fallback_current(...)
```

旧实现中：

```text
MemoryQueryScopeJudge
```

既负责：

```text
Business Decision
```

也负责：

```text
LLM Raw Output Parsing
```

职责较多。

---

# 6. New Contract

迁移后新增：

```text
call_llm_structured()
```

其 Contract：

```text
Input:
messages
output_model

Output:
Awaitable[T]
```

其中：

```text
T
```

必须是：

```text
Pydantic BaseModel subclass
```

例如：

```text
MemoryQueryScopeLLMOutput
```

因此：

```text
output_model = MemoryQueryScopeLLMOutput

↓

StructuredOutputT
=
MemoryQueryScopeLLMOutput

↓

return type
=
MemoryQueryScopeLLMOutput instance
```

---

# 7. Generic Structured Output Capability

`llm_service.py` 新增：

```python
StructuredOutputT = TypeVar(
    "StructuredOutputT",
    bound=BaseModel,
)
```

其作用不是执行 Runtime Parsing。

它主要用于：

```text
Static Type Relationship
```

表达：

```text
调用方传入什么 Pydantic Model Class
↓
函数返回对应 Model Instance
```

例如：

```python
result = await call_llm_structured(
    messages=messages,
    output_model=MemoryQueryScopeLLMOutput,
)
```

类型系统可以推导：

```text
result:
MemoryQueryScopeLLMOutput
```

因此业务代码可以直接使用：

```python
result.scope
result.reason
```

---

# 8. llm_service Architecture

新增两个函数：

```text
structured_completion()
call_llm_structured()
```

职责分别为：

## structured_completion()

Provider / SDK-facing function。

知道：

```text
AsyncOpenAI
responses.parse()
text_format
output_parsed
```

核心链路：

```text
Pydantic Model
↓
SDK 生成 Structured Output Schema
↓
DeepSeek
↓
Structured Response
↓
SDK Parse
↓
Pydantic Model Instance
```

## call_llm_structured()

Application-facing wrapper。

给：

```text
MemoryQueryScopeJudge
MemoryRelevanceJudge
MemoryRelationshipJudge
未来其他 LLM Component
```

提供统一 Structured LLM Contract。

业务组件不需要知道：

```text
responses.parse()
output_parsed
text_format
```

---

# 9. Structured Output Model

新增：

```python
class MemoryQueryScopeLLMOutput(BaseModel):
    scope: Literal[
        "current",
        "historical",
        "both",
    ]

    reason: str | None = None
```

它只描述：

```text
LLM 应该返回什么
```

而不是整个 Query Scope Judge 对外的最终业务对象。

---

# 10. Why Not Reuse MemoryQueryScopeDecision

已有：

```python
@dataclass
class MemoryQueryScopeDecision:
    scope: str
    reason: str
    source: str
```

其中：

```text
source
```

可能是：

```text
rule
llm
fallback
```

`source` 描述：

```text
本次 Decision 是由哪条 Application Execution Path 产生
```

它属于：

```text
MemoryQueryScopeJudge
```

的责任。

LLM 不知道：

```text
Lexical Rule
Fallback
MemoryReader
Application Control Flow
```

因此不应该要求 LLM 输出：

```json
{
  "scope": "historical",
  "reason": "...",
  "source": "llm"
}
```

正确边界：

```text
DeepSeek
↓
MemoryQueryScopeLLMOutput
{
    scope,
    reason
}

↓

MemoryQueryScopeJudge
根据执行路径补：

source="llm"

↓

MemoryQueryScopeDecision
{
    scope,
    reason,
    source
}
```

因此：

```text
MemoryQueryScopeLLMOutput
```

属于：

```text
LLM Boundary Model
```

而：

```text
MemoryQueryScopeDecision
```

属于：

```text
Business Decision Model
```

---

# 11. Data Flow

迁移后的完整 Data Flow：

```text
User Query
↓
MemoryQueryScopeJudge.judge()
↓
Lexical Rule
├── 能判断
│   ↓
│   MemoryQueryScopeDecision(
│       source="rule"
│   )
│
└── 无法判断
    ↓
    _judge_by_llm()
    ↓
    call_llm_structured()
    ↓
    llm_service
    ↓
    structured_completion()
    ↓
    AsyncOpenAI.responses.parse()
    ↓
    DeepSeek
    ↓
    Structured Output
    ↓
    SDK Parsing
    ↓
    MemoryQueryScopeLLMOutput
    ↓
    Judge Data Adaptation
    ↓
    MemoryQueryScopeDecision(
        source="llm"
    )
```

如果 Structured LLM 调用失败：

```text
Exception
↓
MemoryQueryScopeJudge
↓
_fallback_current()
↓
MemoryQueryScopeDecision(
    scope="current",
    source="fallback"
)
```

---

# 12. Prompt Responsibility

迁移后 Prompt 仍然保留：

```text
current
historical
both
```

的语义判断规则。

例如：

```text
当前 / 现在
→ current

过去 / 曾经
→ historical

变化 / 对比 / 转变过程
→ both
```

Prompt 仍然负责：

```text
How to Decide
```

但是原来的：

```text
严格返回 JSON
不要输出 JSON 以外内容
格式如下：
...
```

不再是主要结构约束来源。

因为：

```text
Output Shape
```

已经由：

```text
MemoryQueryScopeLLMOutput
```

表达。

因此形成：

```text
Prompt
→ Semantic Contract

Pydantic Schema
→ Structural Contract
```

---

# 13. Scope Normalization Compatibility

旧实现中：

```python
scope = scope.strip().lower()
```

因此：

```text
"HISTORICAL"
" historical "
```

都会变成：

```text
historical
```

如果直接改成：

```python
Literal[
    "current",
    "historical",
    "both",
]
```

则默认会要求精确值。

这会悄悄改变旧 Business Behavior。

为了保持迁移兼容性，增加：

```python
@field_validator(
    "scope",
    mode="before",
)
```

执行：

```text
str
↓
strip()
↓
lower()
↓
Literal Validation
```

因此：

```text
" HISTORICAL "
↓
historical
↓
PASS
```

这属于：

```text
Backward Compatibility
```

而不是新增业务规则。

---

# 14. Reason Compatibility

旧 Business Contract：

```text
scope
→ 真正控制 Retrieval

reason
→ Runtime Trace
```

因此：

```text
scope 合法
+
reason 缺失
```

不应该导致：

```text
整个 Decision fallback current
```

新 Model 使用：

```python
reason: str | None = None
```

并通过 validator 将：

```text
None
""
"   "
non-str
```

归一为：

```text
None
```

Judge 再补：

```text
"LLM 返回了合法 scope，但没有提供有效 reason"
```

因此：

```text
scope validity
```

与：

```text
reason availability
```

继续保持解耦。

---

# 15. Error Boundary

`llm_service` 不负责：

```text
fallback current
```

因为：

```text
current
historical
both
```

属于 Memory Query Scope 的业务知识。

因此：

```text
llm_service

成功
→ return StructuredOutputT

失败
→ raise
```

而：

```text
MemoryQueryScopeJudge

try
→ call_llm_structured()

except
→ _fallback_current()
```

这样保持：

```text
Provider Error Handling
```

和：

```text
Business Fallback Policy
```

之间的边界。

---

# 16. TEMP Experiments

Structured Output 正式迁移前，建立过：

```text
[TEMP]
tests/async/test_structured_output_temp.py
```

主要实验：

## A. JSON Mode

```text
Prompt 强制：
scope = "past"

response_format:
json_object
```

观察到：

```text
JSON Syntax
→ PASS

Pydantic Validation
→ FAIL
```

说明：

```text
合法 JSON
≠
合法业务 Schema
```

---

## B. JSON Schema Structured Output

实验了：

```text
JSON Schema
enum:
current
historical
both
```

同时 Prompt 故意要求：

```text
past
```

真实 DeepSeek 行为并非每次都稳定表现为：

```text
非法 enum 一定无法生成
```

因此形成工程结论：

```text
Provider Structured Output
不能替代 Application Validation
```

---

## C. responses.parse()

使用：

```text
Pydantic Model
↓
SDK Schema Generation
↓
Provider
↓
SDK Parse
↓
Pydantic Object
```

成功得到：

```text
MemoryQueryScopeDecision-like Pydantic Instance
```

验证了：

```text
Single Source of Truth
```

方向可行。

即：

```text
Pydantic Model
```

既可以作为：

```text
Application Model
```

也可以成为：

```text
Structured Output Schema Source
```

---

# 17. Important Provider Observation

实验中发现：

```text
DeepSeek Responses API
+
json_schema
```

的实际行为，与理想的：

```text
所有 enum 都被绝对 constrained decoding
```

并不总是完全一致。

因此当前工程设计不依赖：

```text
Provider 一定永远输出合法结构
```

而采用：

```text
Provider Structured Output
+
SDK Parse
+
Pydantic Validation
+
Application Fallback
```

多层保证。

这体现：

```text
Trust Boundary
信任边界
```

原则：

```text
外部 Provider 的结果
不能直接成为内部业务事实
```

必须先经过：

```text
Application Boundary Validation
```

---

# 18. [KEEP] New Structured Output Regression

新增：

```text
tests/memory/test_memory_query_scope_structured_output.py
```

类型：

```text
[KEEP]
```

特点：

```text
不调用真实 DeepSeek
不依赖 PostgreSQL
不加载 Embedding
不加载 Reranker
使用 Fake Structured LLM
Deterministic
```

初始结果：

```text
6 passed
```

覆盖：

```text
1. scope normalization

2. invalid scope rejection

3. Structured LLM Output
   → Business Decision

4. reason=None
   不丢失合法 scope

5. Structured LLM failure
   → fallback current

6. Lexical Rule
   → bypass LLM
```

---

# 19. Old Regression Migration

原：

```text
tests/memory/test_memory_read_historical_scope.py
```

仍使用旧 Fake Contract：

```text
call_llm()
→ JSON str
```

Structured Output 迁移后：

```text
call_llm
```

已不再存在于：

```text
memory_query_scope_judge module
```

首次运行结果：

```text
28 passed
21 failed
```

21 个失败全部发生在：

```text
patch("call_llm")
```

目标解析阶段。

Production：

```text
judge()
```

一次都没有执行。

因此分类为：

```text
B. Test Contract Stale
```

而不是：

```text
Production Regression
```

---

# 20. Test Responsibility Migration

旧 Judge 负责：

```text
raw str
empty response
blank response
JSON syntax
dict shape
scope extraction
scope validation
```

新 Judge 不再负责：

```text
raw JSON parsing
JSON object validation
string response validation
```

因此以下旧测试：

```text
empty_response
blank_response
none_response
non_string_response
invalid_json
non_object_json
missing_scope
illegal_scope
```

不能机械地继续留在：

```text
MemoryQueryScopeJudge
```

测试层。

原则：

```text
Component Responsibility
↓
Test Responsibility
```

组件不再承担的责任，

测试也不应该继续假装它承担。

---

# 21. Fake Contract Migration

旧：

```python
FakeLLM.__call__(self, messages)
```

返回：

```text
JSON str
```

新：

```python
FakeScopeStructuredLLM.__call__(
    self,
    *,
    messages,
    output_model,
    model=None,
)
```

返回：

```text
MemoryQueryScopeLLMOutput
```

同时 Fake 验证：

```text
output_model
is
MemoryQueryScopeLLMOutput
```

从而确保 Judge 调用了正确的 Structured Output Model。

---

# 22. Test Consolidation

迁移过程中没有为了保持测试数量而机械复制旧 Case。

例如原来的：

```text
empty response
blank response
None response
non-string response
invalid JSON
non-object JSON
missing scope
illegal scope
exception
```

很多在新架构里都会表现成：

```text
Structured LLM Call Failure
↓
Exception
↓
Judge fallback current
```

如果机械保留 9 个测试，

表面上：

```text
测试数量很多
```

但实际上：

```text
都走同一个 Production Branch
```

属于：

```text
False Coverage Inflation
虚假覆盖膨胀
```

因此进行了：

```text
Coverage Consolidation
覆盖合并
```

---

# 23. Migrated Regression Result

迁移后：

```text
tests/memory/test_memory_read_historical_scope.py

49 tests
↓
39 tests
```

测试数量下降，

但：

```text
Business Contract Coverage
```

没有下降。

主回归：

```text
test_memory_read_historical_scope.py
+
test_memory_query_scope_structured_output.py
```

结果：

```text
45 passed
0 failed
0 skipped
13.13s
```

---

# 24. Supplemental Regression

补充运行：

```text
test_memory_relevance_judge_status_contract.py

test_chat_service_memory_regression.py

test_memory_write_state_change_regression.py
```

结果：

```text
59 passed
0 failed
0 skipped
12.73s
```

因此本轮 deterministic regression：

```text
45
+
59
=
104 passed
0 failed
```

---

# 25. Anti-False-Pass Verification

为了确认绿色不是 Fake 或 patch 失效造成的假 PASS，

进行了反向验证。

## Fake 永远抛异常

结果：

```text
LLM 路径测试
→ FAIL

Lexical Rule 测试
→ PASS
```

证明：

```text
Rule Layer
```

确实没有错误调用 LLM。

---

## Fake 永远返回 current

结果：

```text
expected historical
→ FAIL

expected both
→ FAIL

expected current
→ PASS
```

证明：

```text
LLM scope assertion
```

确实在验证 Production Behavior，

而不是测试失效。

---

# 26. [MANUAL KEEP] Real DeepSeek Verification

新增：

```text
tests/memory/manual_memory_query_scope_structured_output.py
```

类型：

```text
[MANUAL KEEP]
```

依赖：

```text
真实 DeepSeek
真实 Network
真实 Responses API
真实 Pydantic Parse
```

运行方式：

```powershell
python -m tests.memory.manual_memory_query_scope_structured_output
```

避免直接：

```powershell
python tests/memory/xxx.py
```

造成：

```text
ModuleNotFoundError: backend
```

---

# 27. Real DeepSeek Cases

## Case 1

Query：

```text
我现在想知道我以前主要学什么？
```

Layer 1：

```text
current marker
+
historical marker
```

因此：

```text
defer to LLM
```

真实结果：

```text
scope:
historical

source:
llm
```

结果：

```text
PASS
```

---

## Case 2

Query：

```text
我为什么从 Python 转向 Java？
```

没有显式：

```text
以前
现在
```

但语义涉及：

```text
Transition
过去与现在关系
```

真实结果：

```text
scope:
both

source:
llm
```

结果：

```text
PASS
```

---

## Case 3

Query：

```text
我的主要技术方向是什么？
```

时间范围不明确。

Prompt 规则：

```text
优先 current
```

真实结果：

```text
scope:
current

source:
llm
```

结果：

```text
PASS
```

---

# 28. Manual Result

真实 DeepSeek：

```text
historical
→ PASS

both
→ PASS

current
→ PASS
```

并且三条：

```text
source == "llm"
```

证明没有：

```text
Lexical Rule Shortcut
```

也没有：

```text
fallback
```

真实生产链：

```text
MemoryQueryScopeJudge
↓
call_llm_structured
↓
structured_completion
↓
responses.parse
↓
DeepSeek
↓
Pydantic
↓
MemoryQueryScopeLLMOutput
↓
MemoryQueryScopeDecision
```

完整通过。

---

# 29. manual_historical_retrieval_v1 Migration

旧：

```text
scope_judge_module.call_llm
```

已迁移为：

```text
scope_judge_module.call_llm_structured
```

旧：

```text
RecordingScopeLLM(messages)
```

迁移为：

```text
RecordingScopeLLM(
    *,
    messages,
    output_model,
    model=None,
)
```

并透传：

```text
output_model
```

因此：

```text
[MANUAL KEEP]
Historical Retrieval
```

不再依赖已经消失的旧 Fake Contract。

---

# 30. Frozen Business Contract Verification

本次迁移后确认以下 Contract 均未改变：

## Lexical Rule

```text
current
historical
both
```

规则层行为不变。

---

## Mixed Markers

例如：

```text
我现在想知道我以前学什么？
```

仍然：

```text
Rule Layer
→ defer LLM
```

---

## LLM Output

合法 Structured Output：

```text
source="llm"
```

---

## Failure

Structured LLM Failure：

```text
scope=current
source=fallback
```

---

## Reason

```text
reason 缺失
```

不丢弃合法：

```text
scope
```

---

## MemoryReader

```text
current
historical
both
```

scope 行为不变。

---

## Historical Retrieval

历史记忆检索行为保持冻结 Contract。

---

# 31. Known Tech Debt / Limitation

## 31.1 Provider Structured Output Reliability

当前 DeepSeek Structured Output 的实际行为：

```text
不应被视为完全替代 Application Validation
```

因此继续保留：

```text
Pydantic Validation
+
Exception Handling
+
Fallback
```

---

## 31.2 Semantic Errors Still Possible

Schema 可以保证：

```text
scope ∈ {
    current,
    historical,
    both
}
```

但不能保证：

```text
模型一定选择语义正确的那个 scope
```

因此仍然需要：

```text
Prompt
Evaluation
Regression
Manual Test
```

---

## 31.3 Sync Blocking Work

本次迁移没有处理：

```text
Sync SQLAlchemy
Embedding
Reranker
```

这些属于已有 Stage 1 Tech Debt。

不在 Structured Output Migration 范围。

---

## 31.4 Other LLM Components Still Use Old JSON Contract

当前：

```text
MemoryQueryScopeJudge
```

已经迁移。

其他组件例如：

```text
MemoryRelevanceJudge
MemoryRelationshipJudge
MemoryExtractor
MemoryValidator
```

仍可能使用：

```text
Prompt JSON
→ str
→ json.loads()
```

后续逐个迁移。

不一次性批量修改。

---

# 32. TEMP Cleanup

以下文件：

```text
tests/async/test_structured_output_temp.py
```

已经完成学习与实验使命。

分类：

```text
[TEMP]
```

可以：

```text
删除
或
不提交
```

它验证过：

```text
JSON Mode
JSON Schema
responses.parse
Prompt vs Schema
Provider vs Pydantic
```

生产链已经完成后，

不再需要长期保留。

---

# 33. Tests to Keep

## [KEEP]

```text
tests/memory/test_memory_query_scope_structured_output.py
```

负责：

```text
Structured Output Model Contract
Judge Structured Path
Fallback
Rule Bypass
```

---

## [KEEP]

```text
tests/memory/test_memory_read_historical_scope.py
```

负责：

```text
Query Scope
Historical Retrieval
MemoryReader
Business Contract
```

---

## [MANUAL KEEP]

```text
tests/memory/manual_memory_query_scope_structured_output.py
```

负责：

```text
真实 DeepSeek Structured Output Integration
```

---

## [MANUAL KEEP]

```text
tests/memory/manual_historical_retrieval_v1.py
```

负责：

```text
真实 Historical Retrieval E2E
```

---

# 34. Architecture Conclusion

本次迁移前：

```text
MemoryQueryScopeJudge
```

既负责：

```text
Semantic Decision
```

又负责：

```text
Raw LLM Parsing
JSON Parsing
Structural Validation
```

迁移后：

```text
llm_service
→ Provider / SDK Boundary

MemoryQueryScopeLLMOutput
→ LLM Structured Contract

MemoryQueryScopeJudge
→ Business Adaptation

MemoryQueryScopeDecision
→ Application Business Contract
```

职责更加清晰。

---

# 35. Final Data Boundary

最终边界：

```text
Provider-specific:

responses.parse
text_format
output_parsed
SDK Response Object

全部限制在：

llm_service
```

业务层只看到：

```text
MemoryQueryScopeLLMOutput
```

最终上层：

```text
MemoryReader
```

只依赖：

```text
MemoryQueryScopeDecision
```

因此 Provider / SDK 不会泄漏到：

```text
MemoryReader
```

---

# 36. Engineering Conclusion

本次迁移证明：

Structured Output 不只是：

```text
“让 LLM 更容易返回 JSON”
```

而是一次：

```text
LLM Boundary Contract
```

的升级。

核心变化：

```text
Old:

Prompt
↓
LLM
↓
str
↓
Business Component Parsing
↓
Business Decision
```

变成：

```text
New:

Prompt
→ Semantic Rules

Pydantic Model
→ Structural Rules

Provider / SDK
→ Structured Output

LLM Boundary Model
→ Validated Application Data

Business Component
→ Business Adaptation

Business Decision
```

---

# 37. Test Engineering Conclusion

本次还验证了一个重要测试原则：

```text
Component Responsibility
发生变化

↓

Test Responsibility
也必须变化
```

旧测试不能因为历史上存在：

```text
invalid JSON
blank JSON
non-object JSON
```

就永远保留在：

```text
MemoryQueryScopeJudge
```

层。

当 Parsing Responsibility 被移到：

```text
SDK / Pydantic Boundary
```

以后，

Judge Test 应该只守：

```text
Judge 真正拥有的 Contract
```

而不是：

```text
已经消失的 Implementation Detail
```

---

# 38. Final Status

```text
MemoryQueryScopeJudge Structured Output V1

Status:
COMPLETED
```

Evidence：

```text
py_compile
PASS

New Structured Output [KEEP]
6 passed

Migrated Main Regression
45 passed
0 failed

Supplemental Regression
59 passed
0 failed

Deterministic Total
104 passed
0 failed

Real DeepSeek [MANUAL KEEP]
3 / 3 PASS

Frozen Memory Business Contract Regression
NO REGRESSION FOUND
```

因此：

```text
Query Scope Structured Output
Production Vertical Slice
正式完成
```

---

# 39. Next Step

下一步不继续扩展 Query Scope。

优先选择：

```text
MemoryRelevanceJudge
```

作为第二个 Structured Output Migration Component。

原因：

```text
输出结构简单
selected: bool
reason: str
```

适合验证：

```text
call_llm_structured()
```

是否真正具备：

```text
Reusable Generic Capability
```

目标：

```text
第一次迁移
→ 建立 Structured Output Capability

第二次迁移
→ 验证 Capability 可以被其他 Component 复用
```

原则：

```text
不修改 llm_service 通用 Contract
除非发现真实能力缺口

不批量迁移所有 LLM Component

继续采用：

Small Vertical Slice
↓
Contract Review
↓
Production Migration
↓
[KEEP]
↓
[MANUAL KEEP]
↓
Regression
↓
Next Component
```