# Module 8 Post-Freeze Hardening — F-01 / F-02

日期：2026-11-07（Asia/Shanghai）。两项 P2 已完成定点修复。

Module 8、Evaluation Offline V1、Observability B1 和 Metrics B2 继续保持
**COMPLETE / FROZEN**。本轮是模块完成后的可靠性加固，没有开发新功能、重构主循环
或开始其他 Stage。

## 1. Git 基线与历史记录保护

初始分支 main，HEAD 为 `c15f0c1980d320d76078a95c4c85fb3f31f9127d`。
检查工作区、Index 和 origin/main 后确认：本地/远端差异为 0/0；没有已跟踪文件
差异或预先暂存内容；仅有用户要求保留的未跟踪审查报告：

`docs/testing/module8_code_review.md`

已完整阅读该报告第 8、9、10 节，检查两项问题对应的生产代码和测试。
该报告未删除、未覆盖、未暂存、未进入任何提交。以下历史文档也原样保留：

- docs/testing/module8_final_acceptance.md
- docs/testing/observability_b1_validation.md
- docs/metrics_v1.md
- docs/metrics_catalog.md

实施前记录上述五个文件的 SHA-256，实施后逐一核实内容不变。审查报告的哈希为
`BE79EC16F93046789C89C5B8729AE725125B2E4CB5BB2B96250D62DE0DB7106D`。
本记录描述修复后的状态，不回写原审查或历史验收结论。

## 2. F-01：字符串元数据正规化与事件构造隔离

### 原始问题与根因

Tool Call 契约通过 isinstance(value, str) 接受合法 str 子类，ObservationEvent
属性却使用精确 type 判断。StrEnum 工具名称和字符串子类调用 ID 因此可能在观测
事件创建时触发 TypeError，导致原本可执行的 Runtime → Dispatcher 路径中断。
同时，Runtime._emit 在 safe_emit 之前创建事件，NoOpObserver 也不能隔离该错误。

### 最小修复

- 对 isinstance(value, str) 的元数据值调用 `str.__str__(value)`，提取内置 str
  值，兼容 StrEnum 和普通字符串子类，同时避免调用其自定义 __str__。
- 不对任意对象进行 str() 转换。字段白名单、复制冻结、非有限浮点数/嵌套容器拒绝
  保持；bool/int/float 等其他类型的精确校验保持，包括原先拒绝数值子类的行为。
- 新增 safe_emit_factory，将事件时间戳/校验/创建和 Observer.emit 放入同一安全
  旁路。既有 safe_emit 接口继续支持已经构造的事件。
- 共用原有 _safely 诊断机制，通过预先已知的 run_id/event_type 记录构造故障；
  普通异常及备用诊断异常被隔离，不捕获 BaseException，也不引入递归日志重试。
- Runtime 仅改 _emit 和相应 import。主循环、LLM 重试、Dispatcher 和真实业务
  异常边界没有改动。

只正规化事件副本，不更改 LLM 提供的原始 Tool Call ID、名称或业务 History 类型。
观测构造失败仍允许丢事件，但不会改变正常业务结果、掩盖原始异常或吞掉取消。

### 修改文件

| 文件 | 改动 |
| --- | --- |
| backend/agent_runtime/observability.py | 安全字符串正规化、事件工厂发送边界、复用故障诊断 |
| backend/agent_runtime/runtime.py | _emit 将构造与发送一起交给安全边界 |
| tests/agent_runtime/test_observability_hardening.py | 新增 28 项专项用例（含参数化） |

### 验证内容与结果

28 项专项覆盖：普通 str、StrEnum、覆盖 __str__ 的合法 str 子类；真实
Runtime → Dispatcher → Handler 成功执行；NoOp/Logging 两种 Observer；调用 ID
正规化且 History 保留原值/类型；只读事件与原字典隔离；拒绝任意字符串化对象、
非法字段、容器、NaN/Infinity 和数值子类；事件构造故障不影响正常返回、LLM/工具
原始异常；备用日志故障隔离；LLM/工具期间取消和构造中的三类 BaseException 传播。

专项首轮：**28 passed in 0.72s**。

格式化后阶段回归：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/agent_runtime tests/tools -q
```

结果：**647 passed in 1.90s**，包含全部 Runtime/Observability/Metrics 和 Tools。
Ruff E9/F 静态检查、该新增文件格式检查、Python compileall 和 git diff --check
均通过。自查确认该提交仅含上列三个文件，没有提前修改 F-02。

### 独立提交与推送

- Commit：`f726a7cf34759eab9a63ba63ce85e2172ac43a77`
- 标题：`fix(observability): isolate event construction and normalize string metadata`
- 首次 Push 因连接重置失败，随后普通重试和只读查询也遇到 GitHub 连接失败；没有
  将失败宣称为成功，也没有在此期间开始 F-02。
- 读取已启用的 Windows 系统代理后，以命令局部设置重试成功：
  `git -c http.proxy=http://127.0.0.1:12450 push origin main`。
- 实际远端更新：`c15f0c1..f726a7c main -> main`。确认本地/远端 0/0 后才开始 F-02。
- 没有修改系统代理或持久 Git 配置，没有 force push，没有绕过测试。

## 3. F-02：DeepSeekAdapter 诊断日志故障隔离

### 原始问题与根因

Adapter 的成功 info 日志在响应返回前直接执行，外部错误 warning 则在
translate_error 之前直接执行。共享 Logging Handler 抛出 OSError 时，成功响应
可能丢失，原 API 异常也可能无法转换，最终 EvaluationRunner 记录错误的 RAISED。

### 最小修复

在 Adapter 模块内增加一个只包围 logger.log 的小型安全辅助函数，替换两处诊断
调用。只捕获诊断写入的普通 Exception，不写递归备用日志，不捕获 BaseException。

SDK 调用、asyncio.timeout、响应转换和 translate_error 均在该辅助函数之外；
`raise self._translate_error(exc) from exc` 原样保留。错误映射、Runtime 重试和
取消语义没有改变，没有通过吞掉 complete 的异常来保护日志。

### 修改文件

| 文件 | 改动 |
| --- | --- |
| backend/agent_runtime/deepseek_adapter.py | 局部诊断写入隔离和两处日志调用替换 |
| tests/agent_runtime/test_deepseek_logging_hardening.py | 新增 20 项 Adapter / Runtime 专项 |
| tests/evaluation/test_adapter_logging_hardening.py | 新增 6 项真实 Runner 集成回归 |

### 验证内容与结果

使用 Mock SDK，不调用真实 DeepSeek API。测试通过根 Logging Handler 注入故障，
并非只 Mock logger 或一个辅助函数。

- 正常/故障 Handler 下 SDK 成功均返回成功。
- APIConnectionError 正常转换为同一个 ExternalServiceError 对象；translator
  收到同一个原始 API 异常，显式 cause 及其原有网络原因链保持。
- 直接 TimeoutError 和实际 Adapter 预算到期仍保持超时转换，挂起 SDK 调用会清理。
- 真 Runtime 的 HTTP 503 恢复仍为逻辑调用 1、实际尝试 2、额外重试 1，不消耗
  多余迭代，指标仍能在共享日志故障下记录正确结果。
- 非预期 SDK 业务异常原样传播；主动取消和诊断 Handler 的
  CancelledError/KeyboardInterrupt/SystemExit 不被吞掉。
- Mock SDK → 真 DeepSeekAdapter → 真 AgentRuntime → 真 EvaluationRunner 在正常/
  故障日志下，分别记录正确的 RETURNED 或 ExternalServiceError RAISED。
- Runner 取消时 Active Gauge 恢复原值，不开始下一 Case，translator 不误处理取消。

初次专项为 24 passed / 2 failed：两个断言误认为 pytest caplog 一定先于故障根
Handler 收到日志。修正测试注入器，让 Handler 在抛错前记录收到的 LogRecord，
继续验证诊断路径、元数据和敏感内容排除。没有为通过断言放宽生产故障边界。

修正并格式化后的专项：**26 passed in 0.66s**。

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/agent_runtime tests/evaluation -q --basetemp=D:/AI_Agent_Platform/.pytest_cache/m8-f02-precommit
```

阶段回归：**710 passed in 1.92s**。Ruff E9/F 静态检查、两个新增文件格式检查、
Python compileall 和 git diff --check 均通过。自查确认该提交仅含上列三个文件。

### 独立提交与推送

- Commit：`7e0917425b5e951fa57a43a3d4172ad166c9ed04`
- 标题：`fix(llm): isolate DeepSeek adapter diagnostic logging failures`
- 使用既有系统代理的命令局部设置 Push 成功：
  `f726a7c..7e09174 main -> main`，之后本地/远端差异 0/0。
- F-02 在 F-01 提交并成功推送之后开始，两项修复未混入同一个 Commit。

## 4. 最终整体回归

两个修复提交之后重新运行完整范围，记录新的 JUnit 到忽略的缓存目录：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/agent_runtime tests/mcp_host tests/evaluation tests/tools -q --basetemp=D:/AI_Agent_Platform/.pytest_cache/m8-hardening-final --junitxml=D:/AI_Agent_Platform/.pytest_cache/m8-hardening-final-tests.xml
```

结果：**1032 passed in 16.58s，0 failures/errors/skipped**。
从本次 JUnit 的 testcase 记录独立汇总：

| 范围 | 本次通过数 |
| --- | ---: |
| Runtime（含 Observability、Adapter，不含单列 Metrics） | 464 |
| Metrics | 31 |
| MCP Host（含真实本地 stdio） | 150 |
| Evaluation（含新集成测试） | 215 |
| Tools | 172 |
| 合计 | 1032 |

其中 Observability 专项 72 项（B1 原有 44 + F-01 新增 28），是 Runtime 的子集，
不重复相加。F-02 新增 Adapter 20、Evaluation 集成 6，共 26；本轮总新增 54 项。
完整运行覆盖原有 978 项及这 54 项，不沿用旧数字冒充当前执行结果。

其他最终检查：

```powershell
.\.venv\Scripts\python.exe -m pip check
uv --cache-dir D:/AI_Agent_Platform/.pytest_cache/m8b2-uv tool run ruff check --select E9,F backend/agent_runtime/observability.py backend/agent_runtime/runtime.py backend/agent_runtime/deepseek_adapter.py tests/agent_runtime/test_observability_hardening.py tests/agent_runtime/test_deepseek_logging_hardening.py tests/evaluation/test_adapter_logging_hardening.py
uv --cache-dir D:/AI_Agent_Platform/.pytest_cache/m8b2-uv tool run ruff format --check tests/agent_runtime/test_observability_hardening.py tests/agent_runtime/test_deepseek_logging_hardening.py tests/evaluation/test_adapter_logging_hardening.py
.\.venv\Scripts\python.exe -B -m compileall -q backend/agent_runtime tests/agent_runtime tests/evaluation
git diff --check
```

| 检查 | 结果 |
| --- | --- |
| pip check | No broken requirements found |
| Ruff E9/F，全部六个变更 Python 文件 | All checks passed |
| Ruff 格式，三个新增测试文件 | 3 files already formatted |
| Python 编译检查 | 通过，无错误输出 |
| git diff --check / 阶段暂存差异检查 | 通过 |
| 受保护历史文档 SHA-256 | 五个文件内容均未改变 |

沿用已有隔离 Ruff 工具环境，未安装项目依赖；格式化仅针对新增文件，保留存量生产
文件格式以控制差异。测试使用必要的临时目录/stdio 执行权限，未跳过测试。没有
发送真实 LLM/Judge 或公网工具请求，没有生成需提交的测试临时报告。

## 5. 架构与契约影响

| 范围 | 核实结果 |
| --- | --- |
| Runtime 执行语义 | 主循环不变；仅 _emit 观测旁路构造/发送隔离，正常结果、业务异常和取消保持 |
| Tool Contract / Dispatcher / Registry | 无修改；接受的字符串子类现在可通过观测路径 |
| B1 事件协议 | 字段、名称、关联字符串值不变；仅将合法字符串子类规范为事件内置 str 副本 |
| B2 指标口径 | MetricsState、Histogram、MetricsSnapshot 均未修改；标签、计数和耗时口径保持 |
| LLMPort / Bootstrap | 无修改 |
| DeepSeekAdapter | 只保护两处诊断日志；SDK 调用、错误映射、重试/超时/取消契约保持 |
| Evaluation V1 | 生产代码和 Engine 不变，增加真实 Runner 集成测试 |
| MCP Host | 无修改，完整既有回归保留 |
| 基础设施 | 没有新增服务、事件队列、线程模型、Lock 或全局异常吞噬机制 |

相对 c15f0c1 检查上述冻结组件的生产差异，只存在 F-01 的事件旁路与 F-02 的
Adapter 诊断写入变更。默认进程共享 Observer、指标 best-effort、真实 Judge
校准等既有设计取舍和未来事项没有扩大实施范围。

## 6. 收口结论

F-01 与 F-02 两项 P2 均已修复、测试并分别推送。修复实现基线为
`7e0917425b5e951fa57a43a3d4172ad166c9ed04`；本记录在最终回归后单独归档，
使两个修复 Commit 保持独立并记录其实际完整 SHA。

已跟踪代码和历史文档没有待提交改动；审查报告继续按要求保留为未跟踪文件，不以
“清理工作区”为由删除或提交它。Module 8 继续 COMPLETE / FROZEN，本轮到此结束。
