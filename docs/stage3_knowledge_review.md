# Stage 3 核心专业知识复习手册

适用项目：AI Agent Platform · Tool Calling + Minimal Agent Loop（Module 1～8）。核对日期：2026-11-11。

## 1. Stage 3 一页知识总览

```text
用户请求 → AgentRuntime → ContextBuilder → LLMPort / DeepSeekAdapter
                ↑                            ↓ 决策
                └── Tool Message ← Dispatcher ← Tool Call
                                      ├─ 本地 Handler
                                      └─ MCP Tool Adapter → ClientSession → Server
循环边界：上下文预算、最大迭代、超时、有限重试、异常与取消
观测旁路：ObservationEvent → Observer → 结构化日志 + 进程内 Metrics
离线评估：Dataset → EvaluationRunner → ExecutionEvidence → EvaluationEngine
```

| 层次 | 要回答的问题 | 主要职责 |
| --- | --- | --- |
| 决策与执行 | 下一步做什么，何时结束？ | Runtime 编排模型、工具和终止条件 |
| 工具、上下文与 MCP | 能调用什么，携带哪些信息？ | 契约校验、上下文构造、远程工具接入 |
| 工程可靠性 | 失败后怎么办，资源如何收口？ | 有限恢复、故障隔离、取消与清理 |
| 质量与可观测性 | 做了什么，是否做对？ | 事件解释过程，指标描述运行，评估判断目标 |

**阅读标记**：【基】基础概念；【实】当前已实现；【知】已讨论但相应能力未实现或未验证。以下速查表解释基础定义，第三节标明项目落点。

现有记录中 Module 1～8 已完成各自约定范围，Stage 3 总状态仍写为待最终确认。本手册不修改冻结或进度记录；Runtime 已组装，旧 `/chat`、`/chat/stream` 尚未完整迁移到它。

## 2. 专业术语速查表

### 2.1 决策与执行

| 英文术语 | 中文解释 | 一句话含义 |
| --- | --- | --- |
| LLM | 大语言模型 | 根据消息生成回答或工具调用请求。 |
| Agent | 智能体 | 结合模型决策、工具执行和反馈来完成任务的系统。 |
| Agent Runtime | 智能体运行时 | 管理一次任务的实际执行过程。 |
| Agent Loop | 智能体执行循环 | 反复执行“决策—工具—反馈”，直至终止。 |
| Function Calling | 函数调用机制 | 模型按给定定义输出函数名和参数，由应用执行。 |
| Tool Calling | 工具调用 | 模型请求使用工具的机制，本项目以 function 形式承载。 |
| Tool Call / Tool Result | 工具请求／结果 | 前者表达要执行什么，后者反馈实际成功或失败。 |
| `tool_call_id` | 工具调用关联标识 | 将一个结果对应到发起它的请求。 |
| LLMPort / Adapter | 模型接口／适配器 | 前者定义统一调用契约，后者转换具体供应商协议。 |
| User Message | 用户消息 | 表达本轮任务需求。 |
| Assistant Message | 助手消息 | 承载模型文本及工具调用请求。 |
| Tool Message | 工具消息 | 承载工具结果并关联调用 ID。 |
| Iteration | 执行迭代 | Runtime 的一次逻辑模型调用轮次。 |
| Final Answer | 最终回答 | 模型不再请求工具时交给用户的文本。 |
| `COMPLETED` | 正常完成终态 | 模型返回无工具请求的响应，循环正常结束。 |
| `MAX_ITERATIONS_REACHED` | 达到迭代上限 | 末轮仍请求工具，循环按限制退出。 |

### 2.2 工具与上下文

| 英文术语 | 中文解释 | 一句话含义 |
| --- | --- | --- |
| Tool Contract | 工具契约 | 约定注册信息、参数、Handler、结果和错误边界。 |
| Tool Schema | 工具模式定义 | 以机器可读结构描述工具及其参数约束。 |
| Input Schema / Output Schema | 输入／输出结构 | 分别规定参数与成功结果应满足的结构。 |
| Schema Validation | 结构校验 | 检查类型、必填项等声明约束。 |
| Business Validation | 业务校验 | 检查结构合法的数据在业务上能否成立。 |
| ToolRegistry | 工具注册表 | 保存并查询可用工具的定义。 |
| ToolDispatcher | 工具分发器 | 统一查找、校验参数、调用 Handler 并封装结果。 |
| Tool Handler | 工具业务处理器 | 执行天气查询等具体工作。 |
| ToolSuccess / ToolFailure | 工具成功／失败 | 用统一结构表达一次调用的业务结果。 |
| Structured Output | 结构化输出 | 按约定字段表达结果，便于程序解析和校验。 |
| State | 状态 | 执行过程中需要保存的数据及当前进展。 |
| History | 历史记录 | Runtime 保留的完整输入与执行消息。 |
| Context | 模型上下文 | 本次真正发送给模型的信息。 |
| ContextBuilder | 上下文构建器 | 从 History 构造满足本地预算的消息快照。 |
| Context Window / Token Budget | 上下文窗口／预算 | 前者是模型容量，后者是应用分配的输入输出额度。 |
| Context Trimming | 上下文裁剪 | 删除可舍弃的旧消息组以满足预算。 |
| RunContext | 单次运行上下文 | 保存一次 Run 的标识和计时起点。 |
| `run_id` | 运行标识 | 关联同一次 Run 的执行事件。 |
| `llm_call_id` | 逻辑模型调用标识 | 关联同一轮模型请求及其重试。 |
| `parent_llm_call_id` | 父模型调用标识 | 指明工具请求由哪次模型决策产生。 |
| Shared State / Run-local State | 共享／运行局部状态 | 前者跨 Run 复用，后者只属于本次执行。 |

### 2.3 MCP

| 英文术语 | 中文解释 | 一句话含义 |
| --- | --- | --- |
| MCP — Model Context Protocol | 模型上下文协议 | 规范宿主与外部工具等能力的交互。 |
| MCP Host | MCP 宿主 | 使用模型并集成 MCP 能力的应用，本项目即平台。 |
| MCP Client | MCP 客户端 | Host 内与某个 Server 通信的一端。 |
| MCP Server | MCP 服务端 | 对外提供工具等协议能力的进程或服务。 |
| MCP Tool | MCP 工具 | Server 通过协议公开的可调用能力。 |
| MCPClientManager | 客户端管理器 | 管理连接、Session 健康状态、租约与关闭。 |
| ClientSession | 客户端会话 | 承载初始化及后续协议请求的 SDK 对象。 |
| stdio Transport | 标准输入输出传输 | 通过子进程 stdin/stdout 交换协议消息。 |
| Tool Discovery | 工具发现 | 使用 `tools/list` 获取 Server 的工具定义。 |
| Tool Registration | 工具注册 | 将适配后的定义加入本地 Registry。 |
| Session Lifecycle | 会话生命周期 | 从建立、初始化、使用到关闭的全过程。 |
| Session Lease | 会话租约 | 在受控作用域内借用有效 Session。 |
| Server-level Atomic Registration | Server 级原子注册 | 一个 Server 的候选工具整组接纳或整组丢弃。 |
| Partial Availability | 部分可用性 | 部分 Server 失败时仍保留其他可用能力。 |
| MCP Tool Adapter | MCP 工具适配器 | 将远程调用转换为统一本地 Handler 契约。 |

### 2.4 工程可靠性

| 英文术语 | 中文解释 | 一句话含义 |
| --- | --- | --- |
| Error Handling | 错误处理 | 按错误所属边界选择反馈、恢复或终止。 |
| Exception | 异常 | 通过控制流报告执行无法正常继续。 |
| Error Propagation | 错误传播 | 将当前层无法处理的错误交给调用方。 |
| Failure Isolation / Fault Isolation | 故障隔离 | 限制局部故障对其他工作和主流程的影响。 |
| Retry | 重试 | 在同一操作失败后重新发起尝试。 |
| Retry Policy | 重试策略 | 决定哪些错误可重试、次数、等待与预算。 |
| Retryable / Non-retryable Error | 可／不可重试错误 | 区分值得有限恢复的故障和应直接反馈的故障。 |
| Timeout | 超时 | 限制一次等待或操作的持续时间。 |
| Deadline / Time Budget | 截止时间／时间预算 | 前者规定结束时刻，后者规定可耗费时长。 |
| Max Iterations | 最大迭代次数 | 限制模型决策轮数，防止循环无限执行。 |
| Cancellation | 取消 | 调用方要求中断工作，应保持取消信号。 |
| Cleanup | 资源清理 | 在成功、失败或取消后释放所持资源。 |
| Resource Lifecycle | 资源生命周期 | 明确资源由谁创建、持有和关闭。 |
| Idempotency | 幂等性 | 同一操作重复执行仍保持相同业务效果。 |
| Side Effect | 副作用 | 对订单、文件或外部系统产生实际改变。 |
| Per-Tool-Call Failure Isolation | 单次工具调用故障隔离 | 一个可反馈失败不抹掉其他调用的成功结果。 |
| Graceful Degradation | 优雅降级 | 局部能力失效后仍提供边界明确的剩余服务。 |

### 2.5 评估、观测与指标

| 英文术语 | 中文解释 | 一句话含义 |
| --- | --- | --- |
| Agent Evaluation | 智能体评估 | 根据标准和证据判断任务完成质量。 |
| EvaluationRunner | 评测执行器 | 组装并运行用例，记录执行结果和证据。 |
| EvaluationEngine | 评测引擎 | 校验执行批次并组织规则与模型评判。 |
| Evaluation Case / Dataset | 评测用例／数据集 | 前者是一项任务及判据，后者是一组用例。 |
| Deterministic Evaluation | 确定性评估 | 使用固定脚本与 Fixture 复现工程行为。 |
| RuleEvaluator | 规则评估器 | 用明确规则检查状态、调用和参数等。 |
| JudgeEvaluator / JudgePort | 模型评判器／接口 | 前者组织语义评分，后者隔离具体评判模型。 |
| FakeLLM | 模拟模型 | 按固定脚本返回模型响应，隔离真实模型的不确定性。 |
| Fixture | 固定测试数据／替代结果 | 用已知输入输出代替不可控外部执行。 |
| ExecutionEvidence | 执行证据 | 当前 Run 可确认的回答、状态、调用和结果。 |
| PASS / FAIL / UNDETERMINED | 通过／失败／无法判定 | 分别表示标准满足、明确违反、依据不足。 |
| COMPLETE / PARTIAL / UNAVAILABLE | 完整／部分／不可用 | 描述工具调用证据的可确认程度。 |
| Observability | 可观测性 | 通过事件、日志和指标解释系统内部行为。 |
| Observer | 观测组件 | 接收执行事件的轻量同步扩展点。 |
| ObservationEvent | 观测事件 | 描述某个运行阶段发生了什么。 |
| Structured Logging | 结构化日志 | 将事件按固定字段输出为 JSON 等结构。 |
| Logs / Traces | 日志／调用链追踪 | 前者记录事件，后者关联事件以还原调用关系。 |
| Correlation ID | 关联标识 | 将同一 Run、模型调用或工具请求串起来。 |
| Immutable Event | 不可变事件 | 创建后不能修改的事件快照，避免接收者污染记录。 |
| Metrics | 运行指标 | 对事件进行计数或聚合，描述系统运行状况。 |
| Counter | 计数器 | 记录累计发生次数。 |
| Gauge | 仪表值 | 记录可增可减的当前值。 |
| Histogram | 直方图 | 按固定桶聚合耗时等数值分布。 |
| MetricsState / Snapshot | 指标状态／快照 | 前者累计更新，后者提供独立只读视图。 |
| LLM Calls / Attempts / Retries | 逻辑调用／请求尝试／重试 | 分别计一轮决策、实际开始请求、额外请求。 |
| Tool Call Count / Duration | 工具调用次数／耗时 | 统计工具处理终态及完整处理时间。 |
| Active Runs | 活跃运行数量 | 已开始且尚未收到有效终态的 Run 数。 |
| Run Duration | 运行耗时 | 从 Run 开始到终态的单调时钟时间差。 |

## 3. 四层知识详解

### 第一层：Agent 如何决策和结束

【基】单次 LLM 调用只产生一次响应；Agent 还需要应用真正执行工具，把结果送回模型，再决定下一步。Function Calling 不表示模型自己运行 Python，Tool Call 只是请求。

【实】`AgentRuntime` 维护完整 History、构造每轮 Context、调用 `LLMPort.complete()`、校验响应、顺序分发工具并判断终止。`DeepSeekAdapter` 转换供应商消息和参数；Runtime 不依赖其具体响应格式。User 表达需求，Assistant 发起调用，Tool 返回结果，三者不能互换角色。

【实】无 Tool Call 时返回 `COMPLETED`；有调用则先追加 Assistant，再执行各工具并追加对应 Tool Message，供下一轮使用。已知 ToolFailure 可让模型修正参数、选择其他工具或解释限制，不必立即终止 Run。

【实】Iteration 统计逻辑 LLM 调用，底层重试不多占一轮。默认上限为 5；末轮若仍请求工具，返回 `MAX_ITERATIONS_REACHED`、`final_answer=None`，**不执行也不追加该批调用**，因为已无下一轮消费结果。异常和取消向上传播，不额外包装成 `FAILED` 状态。

### 第二层：工具边界、上下文与 MCP

【实】Registry 保存 `RegisteredTool`；Dispatcher 负责查找、参数校验、Handler 执行和结果封装；Handler 处理具体业务。比如 city 是合法字符串属于 Schema 问题，查询后找不到地点属于业务问题。工具定义是调用说明，不能代替执行时校验。

【实】输入使用 Pydantic Schema；成功结果统一为 `ToolSuccess(data=...)`，失败为 `ToolFailure(error=...)`。Dispatcher 要求 Handler 返回字典且可序列化为标准 JSON，**没有统一的每工具业务 Output Schema 字段**。MCP 若声明 outputSchema，由 SDK 检查 structuredContent，Adapter 再校验结果；这与通用 Dispatcher 的检查不是一回事。Structured Output 强调可解析结构，不保证事实正确。

【实】History 保存完整执行消息；Context 是每轮从中选择的独立深拷贝。ContextBuilder 保留所有 System 和当前 User 之后的消息，从最早的已完成旧 User Turn 开始整组裁剪，不修改 History。工具请求和结果必须完整关联，不能留下孤立 Tool Message；同批 ID 唯一，不同批可复用。

【实】预算按 JSON UTF-8 字节及固定开销近似估算，计入工具定义；可用输入＝配置总额度－输出预留－安全余量。不可删除部分仍超额就抛 `ContextBudgetExceededError`，不发送该轮请求。【知】真实 Context Window 与精确 Token Budget 是相关基础概念；当前没有精确 tokenizer 计数，输出预留也不是实际 `max_tokens` 限制。

【实】History、轮次和 RunContext 都是 Run-local State；Runtime 不保存“当前 Run”。Registry、连接和默认 Observer 可共享，但共享不等于混用用户历史。RunContext 保存运行 ID 与时间；另一个真实接口 `ToolContext` 携带可信的 user_id、conversation_id、db，两者职责不同，后者不发给 MCP Server。

【实】平台作为 Host，内部 Client 通过 stdio 与独立 Server 通信。启动时由 Bootstrap 组织完整分页发现、Schema 适配及注册；MCPClientManager 管连接和健康，不承担工具注册。MCP Tool Adapter 借 Session Lease 发起远程调用，把结果或已识别错误转换为统一工具契约。

【实】Server 级原子注册通过临时 Registry 组装实现，不是 Registry 自带事务：一组工具全部合格才接纳，任何一项失败就丢弃整组。失败 Server 不妨碍本地 `get_weather` 和其他健康 Server，构成 Partial Availability。运行期 Server 不可用时，Registry 保留定义，Manager 拒绝新租约；Adapter 安全反馈不可用错误。

【实】V1 使用静态配置、启动时发现，输入 Schema 仅支持已审核子集。【知】没有自动重连、动态工具刷新或运行期注销。Lease 是受控借用 Session，不表示重连权限；关闭时由资源持有任务完成 Session 和传输的逆序退出。

### 第三层：失败、恢复与执行边界

【实】工具不存在、参数错误及 Handler 明确抛出的 `ToolExecutionError` 转为 ToolFailure；未知程序异常、非法 Handler 返回等继续传播。单次调用隔离仅针对可反馈失败：同批其他工具仍可继续，已成功结果保留。未知异常会中断后续调用，不能宣称所有错误都被隔离。

【基】LLM 看到失败后重新决定调用工具，是新的决策轮次；Retry 是底层对同一操作再次尝试。`retryable=True` 只是错误元数据，不自动授予重试权限。不同错误需要不同策略：瞬时网络故障可能恢复，参数或协议错误重复执行通常无益。

【实】Runtime 对已分类的暂时性 LLM 故障最多额外尝试一次；429 还要求有效等待提示和足够剩余预算。首次、等待及第二次尝试共享 **180 秒逻辑调用预算**；Adapter 默认单次超时为 120 秒，SDK 内部重试另有既有配置。天气 Handler 仅在两个明确授权的只读 GET 边界有限重试，Dispatcher 不通用重试所有工具，MCP 也不自动重试。

【基】Timeout 限制等待，Deadline 防止多次尝试各自重置总额度；Max Iterations 只限制决策轮数。【知】当前没有覆盖全部模型和工具执行的统一整体 Run Time Budget。天气 HTTP 分阶段 10 秒超时，也不是整个工具严格 10 秒的承诺。

【基】创建订单有 Side Effect，超时不代表远端没创建成功；未确认 Idempotency 就重试可能重复下单。【知】这是已讨论的工程原则，当前没有订单工具或通用幂等机制。

【实】Cancellation 优先传播，避免继续请求和消耗资源；必要 Cleanup 完成后仍保留取消语义。MCP 连接由明确 owner 持有，关闭可重复调用；某个 Server 清理失败要显式记录并继续清理其他连接。诊断日志和事件构造、发送、指标更新的普通异常被局部隔离；取消等 BaseException 不被吞掉。隔离观测故障不能顺便掩盖业务异常。

### 第四层：解释过程与判断质量

【实】Runner 按 Case 组装 Runtime，采集执行结果和当前 Run 的 ExecutionEvidence；Engine 校验 Dataset 与执行批次对应关系，再交给 RuleEvaluator 和 JudgeEvaluator。Case 给任务与判据，Dataset 组织用例；FakeLLM 固定模型响应，Fixture 替代外部工具结果，二者分别控制决策输入与执行反馈。

【实】Deterministic Evaluation 验证工程行为可复现，不证明真实模型质量。规则可检查调用、参数和结果，但观察到请求不等于 Handler 真正运行，例如参数可能已被拒绝。JudgePort 隔离评分模型，JudgeEvaluator 根据参考事实和 rubric 组织语义评判。【知】Judge 接口与适配器已有，真实 Judge API 校准尚未完成；离线模式的 Judge 项为不适用，不能充当语义通过。

【实】必需判据明确违反则 FAIL；全部必需判据通过且没有阻断错误才 PASS；证据、上下文或判定不足则 UNDETERMINED。预期异常用例可以 PASS，因此通过代表满足该 Case 的标准，不天然代表用户任务成功。

【实】COMPLETE 表示工具证据完整，空集合可证明没有调用；PARTIAL 表示仅能确认部分过程，UNAVAILABLE 表示调用情况未知。当前证据主要来自返回 History 的新增部分，不从观测日志拼造；达到迭代限制时为 PARTIAL，末轮跳过的调用未写入 History，不能据此断言其细节。证据不足时，“没找到调用”不能轻易等同于“没有调用”。

【实】Observer 接收不可变事件，LoggingObserver 分别安全写 JSON 日志和更新 MetricsState。`run_id → llm_call_id → tool_call_id` 串起调用，工具事件以 `parent_llm_call_id` 指向模型轮次。同批之外 ID 可复用，精确定位还应结合轮次或父调用 ID。

【实】Counter 记录累计量，Gauge 记录活跃 Run，Histogram 保存 count、sum 和累计桶。Calls 按逻辑调用开始计数；Attempts 按真正进入 LLMPort 计数；Retries 只在额外尝试开始时增加，等待期间取消不虚增，SDK 内部 HTTP 尝试不可见。工具次数按 success/failure/exception/cancelled 终态分类，耗时覆盖完整处理；Run Duration 使用单调时钟。

【实】默认指标在进程内跨 Runtime 共享，Snapshot 复制冻结，重启清零；日志失败与指标失败相互隔离。观测允许丢事件，因此指标可能不完整。【知】当前没有 Token 消耗统计、完整分布式 Trace、跨进程汇总或线上监控平台。

## 4. 易混淆概念对照

| 对照 | 复习时抓住的区别 |
| --- | --- |
| LLM / Agent / Runtime | 模型决策；系统完成任务；运行时负责执行编排。 |
| Function Calling / 真正执行 | 输出调用意图，不等于已经执行函数。 |
| Registry / Dispatcher / Handler | 存定义；管统一入口；做具体业务。 |
| Schema / Business Validation | 数据符合结构，不代表业务成立。 |
| History / Context | 完整记录与当轮输入不同，裁剪不删历史。 |
| ToolFailure / Runtime Exception | 可反馈业务失败与必须向上传播的异常不同。 |
| Iteration / Attempt / Retry | 一轮决策可含多次尝试，额外尝试才是重试。 |
| RunContext / ToolContext | 运行标识和计时，与可信业务依赖不同。 |
| Discovery / Registration | 远程发现定义，与本地纳入调用目录不同。 |
| Runtime `COMPLETED` / Evaluation PASS | 正常结束与满足评测标准不同，回答仍可能错误。 |
| COMPLETE / COMPLETED | 前者描述证据，后者描述 Runtime 终态。 |
| Observability / Metrics / Evaluation | 解释发生什么；聚合运行数字；判断是否达标。Metrics 属于 Observability。 |

## 5. Agent 完整执行链路

以“查询武汉当前天气并回答”为例，本地工具路径如下，MCP 分支可在相同 Dispatcher 后替换 Handler：

```text
启动：注册 get_weather；连接 MCP Server → 发现 → 适配 → 原子注册
运行：User → 局部 History / RunContext → run.started
第1轮：ContextBuilder → llm_call_id → LLMPort → get_weather(city=武汉)
工具：Assistant Call → Dispatcher 校验 → Weather Handler
      → ToolSuccess / ToolFailure → 携原 tool_call_id 的 Tool Message
第2轮：重建 Context → LLM 读取结果 → 无 Tool Call → Final Answer
终止：COMPLETED → run.finished；Observer 同步输出日志并聚合指标
离线复现：Runner 用脚本和 Fixture 执行 → Engine 按规则／语义判据评估
```

工具成功却回答错温度，Runtime 可为 COMPLETED，指标也正常，任务质量仍可能 FAIL。工具失败后如实说明限制可以正常终止，但是否达标取决于 Case 的业务要求；不能把“没崩溃”当成“完成用户目标”。

## 6. 最后一页速记

- **模型提议，Runtime 编排，Dispatcher 分发，Handler 执行。**
- 请求与结果靠 ID 配对；裁剪按完整旧 Turn，当前请求受到保护。
- 工具已知失败进反馈，未知异常向上传播，取消信号优先。
- 一轮模型决策不等于一次网络请求；重试需要分类、授权和预算。
- 180 秒是逻辑 LLM 调用预算；轮次限制不等于整体 Run 截止时间。
- MCP Manager 管连接，Bootstrap 管发现注册，Adapter 管契约转换。
- 局部业务状态按 Run 隔离；共享指标只能说明进程运行情况。
- **终止状态、证据完整性、质量判定是三个独立维度。**
- 判断“没有发生”需要完整证据；无法确认时保留 UNDETERMINED。
- 日志、指标、评估各司其职；COMPLETED 不保证任务成功。

核对依据（仓库相对路径）：`README.md`、`docs/stage3_status.md`、`docs/testing/context_builder_v1_validation.md`、`docs/testing/module6_v1_validation.md`、`docs/mcp_host_integration/ARCHITECTURE.md`、`docs/testing/evaluation_a3.md`、`docs/testing/evaluation_a4.md`、`docs/observability_v1.md`、`docs/metrics_v1.md`、`docs/testing/module8_post_freeze_hardening.md`；定向核对 `backend/agent_runtime/`、`backend/tools/infrastructure/` 与 `backend/evaluation/` 的相关接口。未发现需靠猜测补齐的实现细节；尚未实现或未实测能力已标【知】。
