# Stage 3 Module 6 — Error Handling / Retry / Limits V1

日期：2026-10-29。状态：**COMPLETE (V1)**；Stage 3 整体仍为 IN PROGRESS，Module 7 尚未开始。
该结论针对下列 Module 6 验收项，不表示原始全仓 pytest 无条件通过。
独立审查发现的真实传输层原因链 P1 已完成定向修复与相关回归，等待再次 Review；
本轮没有重跑全仓，首次收尾的全仓数字保留为历史记录。
ReadError / WriteError 的可信包装与底层原因边界亦已补齐；首次 Read/Write 复核相关回归为
556 passed，泛化 ConnectionError 的最新微调相关回归为 289 passed。命令及范围分别记录如下。

## 基线与改动范围

开始和结束时均为 `main@98e791414c18ccb8e4406027cb72602f2e70bdb5`。
开始时仅有上一批已验收的五个文件修改，无额外用户修改或未跟踪文件。
保留上述全部有效实现；未创建 commit、tag 或执行 push。

| 文件 | 来源及目的 |
| --- | --- |
| `backend/agent_runtime/runtime.py` | 上一批：LLM 最多一次额外重试、180 秒逻辑调用预算、独立快照和脱敏日志；本轮未改 |
| `backend/agent_runtime/deepseek_adapter.py` | 上一批提供保守分类；P1 修复识别可信 httpcore2 连接包装并继续核验底层原因 |
| `backend/tools/implementations/weather.py` | 两个已审查 GET 边界显式授权有限重试；P1 修复传输层原因链分类 |
| `tests/agent_runtime/test_runtime.py` | 上一批：预算、取消、最终异常、快照和循环语义；本轮未改 |
| `tests/agent_runtime/test_deepseek_adapter.py` | 分类、等待提示、异常兼容性及 P1 真实映射原因链正反例 |
| `tests/agent_runtime/test_deepseek_weather_roundtrip.py` | 保留原矩阵；P1 将 HTTP 状态恢复扩展到真实依赖映射的连接拒绝和连接超时 |
| `tests/tools/test_weather.py` | 授权、次数、超时、取消、业务结果及 P1 原因链恢复与最终失败 |
| `README.md` | 同步 Current Stage、Stage 3 路线及 V1 行为；未改 Stage 4 |
| 本报告 | 汇总验收证据、实际命令、已知问题和范围限制 |

未修改 Dispatcher、Registry、共享异常、ChatService、llm_service 或客户端配置。

## 实际策略

### LLM 调用（沿用上一批）

- SDK 重试保持原配置；Runtime 在 SDK 最终失败后最多额外调用一次，不增加逻辑 iterations。
- 每个逻辑 LLM 调用使用事件循环单调时钟与 `asyncio.timeout_at` 建立 180 秒截止时间。
  首次尝试、等待及第二次尝试共享预算；Adapter 默认单次整体超时仍为 120 秒。
- 已确认瞬时连接故障、ExternalTimeoutError（包括 HTTP 408）和 HTTP 5xx 可恢复一次。
- HTTP 429 必须提供真实响应等待提示：秒数、HTTP 日期或 retry-after-ms；
  值须有限、非负、最多 120 秒，且剩余预算必须大于等待值。
- 409、其他 4xx、协议/截断、未知或无法分类的错误、主动取消不获 Runtime 额外重试。
- 不启动第二次调用便预算不足时保留首次异常；第二次失败传播最终异常；调用中的预算超时
  使用 ExternalTimeoutError。调用方 CancelledError 保持取消语义。
- 每次尝试深拷贝请求。脱敏日志通过逻辑调用 ID 关联首次和最终失败，不输出请求或凭据。

### 工具授权与有限重试（本轮）

实际注册表只有 `get_weather`。它的地理编码与当前天气接口均为固定地址的只读 GET。
授权来自 Handler 两个调用点显式传入 `_get_json(..., allow_retry=True)`，同时要求 URL
精确匹配这两个已审查地址。该私有参数默认 False，不来自 LLM 参数或 ToolErrorInfo。
未授权路径即使报告 retryable=True 也只有一次请求；Dispatcher 不负责重试。

每个授权 HTTP 路径最多两次尝试，整个成功 Weather Handler 最多四次请求：

| 失败类型 | 本轮自动重试 |
| --- | --- |
| ConnectError、ConnectTimeout；可信 httpcore2 包装须有原因，最终原因符合已有连接拒绝/重置/中止/超时策略 | 最多一次 |
| HTTP 408、500、502、503、504 | 最多一次 |
| HTTP 429、其他 4xx、501/505 等非白名单状态 | 不重试 |
| Read/Write/Pool timeout、其他网络/协议错误 | 不重试；保留既有结构化错误元数据 |
| JSON/编码/响应 Schema 错误、无匹配地点 | 不重试 |
| 直接抛出的未知程序异常、主动取消 | 原样传播，不重试 |
| 连接包装器内的未知异常、TLS/权限错误、循环原因链 | 不自动重试，保留原有 HTTP 异常转换边界 |

最终失败仍转换为原有 ToolExecutionError，保留最终 HTTP/网络异常作为原因；
Dispatcher 再转换为 ToolFailure 反馈给 LLM。未知程序异常不新增兜底业务包装。
不改变输入/输出 Schema、匹配地点选择或天气业务结果。没有写操作或未知远端副作用路径。

### 超时与终止语义

两条 GET 继续使用 `AsyncClient(timeout=10.0)`；每次尝试的 connect/read/write/pool
参数均为 10 秒，不禁用或重置为无限超时。读取超时也按原有工具错误契约处理。
这是 HTTP 分阶段/读取等待保护，不是整个工具严格 10 秒或 40 秒的总耗时承诺。
未发现其他已注册工具的超时缺口，因此未新增统一包装或额外总超时。

COMPLETED、MAX_ITERATIONS_REACHED、已知 ToolFailure、最终 LLM 异常、未知异常、
调用方取消及 ContextBudgetExceededError 的语义保持不变；不新增 FAILED 状态。

## 首次收尾验收记录（P1 修复前）

以下保留首次收尾证据。其模拟连接测试未覆盖真实 httpcore2 中间层，
不能独立证明真实传输层恢复；该缺口及本轮修复结果见后面的 P1 复核。

| 验收项 | 实际结果及证据 |
| --- | --- |
| 工具默认关闭、显式授权、retryable 不授予权限 | PASS：Weather 未授权/非白名单 URL 测试；既有 Dispatcher 两种 retryable 测试 |
| 授权瞬时失败恢复、连续失败最多两次、最终异常控制结果 | PASS：两个请求阶段的恢复/超时/HTTP 矩阵和 503 后 403 测试 |
| 确定性 4xx、未知/协议/数据错误不自动重试 | PASS：Weather 状态、编程异常、包装异常、JSON/响应测试 |
| 超时有效且重试不绕过保护 | PASS：每次请求 extensions.timeout 断言和四种超时子类的有限次数/最终原因测试 |
| 主动取消传播 | PASS：两个阶段、首次及第二次尝试的真实 task.cancel；既有 Runtime/Adapter 取消测试 |
| SDK/LLM/工具重试不重放工具或改变 iterations | PASS：8 种 SDK/Runtime/Weather 恢复组合，最多两次 _get_json、每路径最多两次 GET，iterations=2 |
| 180 秒预算、等待计时、嵌套超时、快照隔离 | PASS：保留并运行 test_runtime.py 的受控事件循环时钟与快照测试 |
| 终止、上下文、异常契约和普通 Chat | PASS：Agent/工具相关 420 项及 Chat 等相关 38 项 |
| 全仓无新增失败 | PASS（有既有例外）：最终全仓 792 passed、35 skipped、4 个已知 fixture errors，无测试失败 |

## 首次收尾实际测试命令及结果（历史记录）

所有命令从项目根目录运行，解释器为项目 `.venv` 的 Python 3.12.9。
异步测试在获准的沙箱外运行，避免已确认的 Windows asyncio socketpair 沙箱初始化阻塞。
不调用真实天气/LLM；真实数据库/模型测试保持既有 opt-in 关闭。
普通 Chat 及全仓命令在进程内设置测试凭据（没有写入 .env）：

```powershell
$env:DEEPSEEK_API_KEY='test-not-a-secret'
$env:DEEPSEEK_BASE_URL='https://example.invalid/v1'
```

以下各行均使用相同前缀 ` .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider `：

| 实际参数 | 结果（passed / skipped / failed / errors） |
| --- | --- |
| `tests/tools/test_weather.py -q` | 95 / 0 / 0 / 0（初版） |
| `tests/agent_runtime tests/tools -q` | 408 / 0 / 0 / 0（白名单最终补充测试前） |
| `tests/async tests/logging tests/memory/test_chat_service_memory_regression.py tests/frontend -q` | 38 / 0 / 0 / 0，15.53 秒 |
| `-q` | 780 / 35 / 0 / 4，45.05 秒（白名单最终补充测试前） |
| `tests/tools/test_weather.py tests/agent_runtime tests/tools/test_dispatcher.py tests/tools/test_registry.py tests/tools/test_package_layout.py -q` | 420 / 0 / 0 / 0，4.32 秒（最终代码） |
| `-q -rs` | 792 / 35 / 0 / 4，37.13 秒（最终代码） |
| `-q --ignore=tests/memory/test_memory_reader_v2_e2e.py` | 791 / 35 / 0 / 0，39.91 秒（最终代码，退出码 0） |

原始全仓退出码为 1；四个 setup errors 全在未修改的
`tests/memory/test_memory_reader_v2_e2e.py`，均为 `fixture 'reader' not found`：
test_langgraph_lexical、test_langgraph_semantic、test_no_relevant_memory、test_no_memory。
与已有 ContextBuilder / Phase 5 报告一致，没有删除、修改或弱化这些测试。
35 项 skip 均为既有 PostgreSQL / 真实 Qwen opt-in 测试；跳过不等于集成验证通过。
整文件排除也排除其中一项原本可运行的通过用例，因此通过数量比原始全仓少一项。
不能将这条有条件的回归结果表述为原始全仓无条件通过。

首次收尾时 `git diff --check` 通过。工作区共 8 个已跟踪修改文件和本报告 1 个预期未跟踪文件，
没有其他未跟踪项、暂存项或冲突；该次收尾保留了上一批四个非 roundtrip 文件的原样内容。

## P1 修复复核（2026-10-29）

本 P1 开始时为同一 main/HEAD，8 个已跟踪修改和 1 个新增报告，无额外用户文件。
保留全部先前改动；本次增量只涉及两个分类器、三个相关测试文件及 README/本报告。

### 缺陷与最小修复

独立审查实际复现：`httpx2.ConnectError → httpcore2.ConnectError → ConnectionRefusedError`
及 `httpx2.ConnectTimeout → httpcore2.ConnectTimeout → TimeoutError` 在中间 core 层
被旧分类条件拒绝，Weather 已授权路径只尝试一次，LLM 连接错误返回 unclassified_connection。

本次只调整 Weather 与 Adapter 现有原因链遍历条件：

- 接受可信 httpcore2.ConnectError/ConnectTimeout 作为中间包装，要求继续提供明确原因；
  包装没有原因时仍拒绝，不仅凭包装类型认定瞬时故障。
- 识别 AnyIO 的无 errno OSError 包装，只在其有明确 cause 时继续核验底层；
  裸未知 OSError、TLS、权限、未知程序异常仍拒绝。
- 补齐宽泛 APIConnectionError 包含 httpx2.ConnectTimeout 的分类条件。
- 保留原有 seen 集合，循环链拒绝；不改变 HTTP 白名单、授权、次数、超时或 Runtime。

测试使用真实 httpcore2 异常类及依赖的 `map_httpcore_exceptions` 映射，保留完整原因链。
Weather 覆盖两个阶段的恢复、连续两次失败的最终原因、未授权、取消及危险/未知原因拒绝。
Adapter 测试将完整映射原因链传至 Runtime，验证一次恢复、最多两次调用和最终 SDK 原因。
roundtrip 扩展为 24 种组合，覆盖 HTTP 状态、连接拒绝和连接超时；维持 iterations=2、
两个逻辑工具 HTTP 调用边界、每路径最多两次 GET，LLM 恢复不重放工具。

### 本轮实际命令和结果

相同前缀：`.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider`。
异步测试在获准的沙箱外执行；外部请求均为 Fake/Mock，无真实天气/LLM 请求。
相关回归使用前述进程内测试凭据，没有写入 .env。

| 实际参数 | 实际结果 |
| --- | --- |
| `tests/tools/test_weather.py tests/agent_runtime/test_deepseek_adapter.py -k core_connection_chain -q` | 初跑 48 passed、2 failed、216 deselected：发现 ConnectTimeout 还会被循环内部条件拒绝；补齐后复跑 50 passed、216 deselected（1.79 秒） |
| `tests/tools/test_weather.py -q` | 141 passed、0 skipped、0 failed（1.20 秒） |
| `tests/agent_runtime/test_deepseek_adapter.py tests/agent_runtime/test_runtime.py tests/agent_runtime/test_deepseek_weather_roundtrip.py -q` | 235 passed、0 skipped、0 failed（3.78 秒） |
| `tests/agent_runtime tests/tools tests/async tests/logging tests/memory/test_chat_service_memory_regression.py tests/frontend -q` | 528 passed、0 skipped、0 failed（21.29 秒） |

Runtime 的 180 秒预算、剩余时间、取消与快照测试全部保留并通过，没有修改 Runtime 文件。
README 仅补充可信传输层包装说明；没有修改共享 SDK、异常契约、Dispatcher 或普通 Chat。
没有重跑全仓；791/35 和原始 792/35/4 均为本节之前的历史证据，不能视为 P1 修复后结果。

另执行 PowerShell 内存脚本经标准输入传给 `.\.venv\Scripts\python.exe -B -`，
复核审查时的真实 AsyncHTTPTransport 路径：只替换连接池方法，使其抛出 core 连接异常，
不访问网络、不写文件。ConnectError 与 ConnectTimeout 两例均断言 Weather 恰好尝试
2 次，完整原因链的 Runtime 分类 delay_seconds=0.0，脚本退出码 0。

最终 `git diff --check` 通过，无暂存项或额外未跟踪文件。哈希核对确认 Runtime 及
test_runtime.py 与本 P1 开始时完全相同；工作区仍为 8 个已跟踪修改及 1 个新增报告。
修复没有展开 ExceptionGroup；多地址聚合中无法可靠分类的原因继续保守地不重试。

## Read/Write 原因链边界复核（2026-10-29）

### 范围与实际规则

开始时仍为 main@98e791414c18ccb8e4406027cb72602f2e70bdb5，8 个已跟踪修改和
1 个新增报告。本次只增量修改 Adapter 分类器、本文件和对应 Adapter 测试。
先前所有改动均保留，没有重置、暂存、提交、打 tag 或 push。

原实现对裸 httpx2.ReadError/WriteError 直接授权，却拒绝真实 httpcore2 中间包装。
本次保留既有遍历和 seen 集合，将 httpcore2.ReadError/WriteError 纳入可信中间包装，
要求继续核对底层原因；没有 cause 的可信 core 包装仍拒绝。

对于 Read/Write 入口，原因链必须最终到达既有策略允许的明确瞬时 OS 原因，
不能止于 ReadError、WriteError 或其他无证据的网络类型标签。原有 errno、TLS、
权限、未知异常和循环链检查保留；ConnectError/ConnectTimeout 既有策略不变。

正例经真实依赖的异常映射产生完整原因链：

- httpx2.ReadError → httpcore2.ReadError → ConnectionResetError。
- httpx2.WriteError → httpcore2.WriteError → BrokenPipeError。

测试通过 Adapter 和 Runtime 验证首次失败后成功、连续失败最多两次调用，
及最终 SDK 异常身份与原因链保留。反例包含 TLS 证书、握手、权限、认证、RuntimeError、
AttributeError、裸未知 OSError、缺失原因、循环链、包装未知程序错误及只有网络类型标签。
裸 httpx2.ReadError/WriteError 的已有分类断言已收紧为不重试，并新增 Runtime 单次调用断言。

### 本次实际测试

统一前缀：`.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider`。
在获准的沙箱外执行，使用真实依赖异常类型、Fake/Mock，无真实天气或 LLM 请求。
相关回归仍使用前述进程内测试凭据，未改 .env。

| 实际参数 | 实际结果 |
| --- | --- |
| `tests/agent_runtime/test_deepseek_adapter.py -k read_write -q --tb=short` | 修改生产代码前：22 passed、6 failed、125 deselected；证明可信链漏重试、裸类型误重试 |
| `tests/agent_runtime/test_deepseek_adapter.py -k read_write -q` | 修复后：28 passed、125 deselected（1.64 秒） |
| `tests/agent_runtime/test_deepseek_adapter.py -q` | 153 passed、0 skipped、0 failed（3.00 秒） |
| `tests/agent_runtime/test_runtime.py tests/agent_runtime/test_deepseek_weather_roundtrip.py -q` | 110 passed、0 skipped、0 failed（2.98 秒） |
| `tests/agent_runtime tests/tools tests/async tests/logging tests/memory/test_chat_service_memory_regression.py tests/frontend -q` | 556 passed、0 skipped、0 failed（19.48 秒） |

没有修改 Runtime、Weather、SDK 配置或 Dispatcher，未扩大 HTTP 白名单或重试授权。
哈希核对确认 Runtime、Weather、README、Runtime/Weather 测试及 roundtrip 文件与本轮开始时一致。
180 秒预算、主动取消、请求快照、逻辑 iterations、终止状态和工具不重放由保留的回归验证。
`git diff --check` 通过，仍为 8 个已跟踪修改及 1 个新增报告，无其他未跟踪文件或暂存项。
本轮没有重跑全仓；此前全仓数量、跳过项和 reader fixture 错误均为历史证据。
ExceptionGroup 等无法可靠分类的原因继续保守拒绝，不新增通用框架；停止等待 Review。

## 泛化 ConnectionError 微调复核（2026-10-29）

本次仍在同一 main/HEAD，保留全部未提交改动，仅增量修改 Adapter 分类条件、
对应测试和本报告；不改变 README、Runtime、Weather 或其他既有文件。

Read/Write 原因链中的泛化 ConnectionError 不再因继承 OSError 而跳过 errno 白名单。
只有 ConnectionResetError、ConnectionRefusedError、ConnectionAbortedError、BrokenPipeError、
明确 TimeoutError 可按既有类型证据处理；其他 OSError（包括泛化 ConnectionError）
必须满足已有 errno 白名单。无 errno 或非白名单 errno 的泛化 ConnectionError 拒绝重试。
ConnectError/ConnectTimeout 入口继续使用其原有判定，新增两个回归用例明确验证未改变。

完整原因链正例扩展到读/写两种包装、六类获准底层原因，分别验证恢复成功及连续两次失败。
反例新增 ConnectionError("unknown") 和带权限 errno 的泛化 ConnectionError，验证 Runtime
只调用一次；TLS、权限、认证、程序异常、缺失原因和循环链的既有断言全部保留。

本次实际命令仍使用前缀 `.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider`，
在获准的沙箱外运行，仅使用 Fake/Mock，不调用真实天气或 LLM API：

| 实际参数 | 实际结果 |
| --- | --- |
| `tests/agent_runtime/test_deepseek_adapter.py -k read_write -q` | 52 passed、125 deselected（1.89 秒；随后新增两个 Connect 策略回归用例由下一命令覆盖） |
| `tests/agent_runtime/test_deepseek_adapter.py tests/agent_runtime/test_runtime.py tests/agent_runtime/test_deepseek_weather_roundtrip.py -q` | 289 passed、0 skipped、0 failed（4.17 秒） |

`git diff --check` 通过。哈希核对确认除上述三个文件外，先前文件内容均保持原样；
仍为 8 个已跟踪修改和 1 个新增报告，没有其他未跟踪文件或暂存项。
未改变 SDK、HTTP/429 策略、180 秒预算、尝试次数、取消、快照、终止状态或工具授权。
本次未运行全仓；前述全仓数量及其他轮次结果均为历史记录。保留原有保守分类与
ExceptionGroup 限制；未提交、push 或打 tag，停止等待最终 Review，不启动 Module 7。

## 已知限制与后续

- 不建设通用重试框架，不增加 Run 总超时、独立工具次数上限或 Dispatcher 统一超时。
- HTTP 分阶段超时与 asyncio 协作取消不能强制中断阻塞事件循环或吞取消的自定义实现。
- 非白名单连接来源保守地不自动恢复；未来增加有副作用的工具须逐路径审查，不能复用
  retryable=True 作为执行授权。
- 全仓 reader fixture 缺失应在独立测试维护任务中处理，真实 PG/模型 opt-in 验证另行安排。
- Module 6 V1 功能验收完成；本轮停止，不启动 Module 7。
