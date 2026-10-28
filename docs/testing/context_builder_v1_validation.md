# Stage 3 Module 5 — ContextBuilder V1 实现与验收报告

日期：2026-10-28。实现前 HEAD：`e0224b0`，工作区干净。
实现阶段未执行 Commit、Push、Tag 或 Merge；本次 Finalization 包括文档同步与 Commit/Push，
不创建 Tag 或 Release，不合并其他分支，不开发新功能。

**Stage 3 Module 5 — State / Context：COMPLETE / FROZEN**。
Stage 3 整体仍为 **IN PROGRESS**；Module 1–4 为 COMPLETE。
**Next — Stage 3 Module 6: Error Handling / Retry / Limits**。
后续仍为 Module 7 MCP 基础实践、Module 8 Agent Evaluation + Basic Observability。

## 1. 修改文件与职责

| 文件 | 职责 |
| --- | --- |
| `backend/agent_runtime/context_builder.py` | 预算配置、近似估算、History 协议校验、当前请求起点识别和完整 Turn 裁剪 |
| `backend/agent_runtime/errors.py` | 新增 ContextInputError 和 ContextBudgetExceededError |
| `backend/agent_runtime/runtime.py` | 固定本次请求起点，每次模型调用前构造 Context |
| `backend/agent_runtime/bootstrap.py` | 创建并注入 ContextBuilder，支持显式预算或构造器注入 |
| `tests/agent_runtime/test_context_builder.py` | 独立预算、裁剪、协议及深拷贝测试 |
| `tests/agent_runtime/test_runtime.py` | 多轮调用、动态裁剪、超额停止、Run 隔离及输入校验测试 |
| `tests/agent_runtime/test_bootstrap.py` | 默认组装、显式注入、预算配置和配置冲突测试 |
| 本报告 | 保存接口、实现决策、验证结果与限制 |
| `README.md`（Finalization） | 同步 Current Stage、Roadmap、Module 5 摘要与限制 |

## 2. 实际 Contract

```python
ContextBuilder(budget: ContextBudget | None = None)

ContextBuilder.build(
    history: list[LLMMessage],
    tools: list[LLMToolDefinition],
    current_turn_start: int,
) -> list[LLMMessage]
```

返回独立深拷贝，包括工具调用的嵌套 arguments。不修改 History 或 tools。
构造器只保存预算配置，不保存任何 Run 的 History、索引或裁剪结果。
不读取数据库、不检索、不调用模型、不执行工具。

`find_current_turn_start(history)` 在 Run 开始时校验输入并确定最后一个 User。
V1 要求输入以非空白 User Message 结束，因此该索引就是末尾索引。
拒绝没有 User、空白 User、末尾已有 Assistant/Tool/System、非法消息和残缺工具批次。
Builder 每轮还校验固定索引确实指向最后一个 User，防止错误或陈旧索引。

异常消息不回显用户文本、工具参数、结果或 schema。
ContextInputError 同时兼容 ValueError 和 LLMProtocolError；已有工具调用字段校验
复用 validate_tool_calls，其失败仍为 LLMProtocolError。
ContextBudgetExceededError 是独立上下文异常，不进入工具失败反馈路径。

## 3. Runtime 集成

```text
Bootstrap 创建/注入 ContextBuilder
  → run() deepcopy 输入到完整 History
  → 校验输入并固定 current_turn_start
  → 从 Registry 获取工具定义
  → 每轮 ContextBuilder.build(history, tools, current_turn_start)
  → LLMRequest 消息与 tools 深拷贝快照
  → LLMPort.complete()
  → 原有响应校验、工具执行、完整 History 追加和终止流程
```

Runtime 的前四个位置参数保持兼容，新增仅关键字 context_builder。
直接构造 Runtime 的旧调用方仍有默认构造器。
Bootstrap 增加 context_budget/context_builder 两个可选关键字参数，禁止同时指定。
FakeLLM 注入路径仍不导入旧服务或读取凭据。

max_iterations 继续统计 LLM 调用次数；最后一轮请求的工具批次不执行、不追加。
已知工具错误仍可反馈，未知异常和上下文异常直接向上抛出。
DeepSeekAdapter、LLMMessage、ToolContext、Dispatcher 的契约没有改变。

## 4. 裁剪与预算算法

先检查完整 History 的协议，再检查完整输入预算；足够时不裁剪。
旧 User Turn 从一个 User 开始，到下一个 User 之前结束。
最后一条非 System 消息为不含工具调用的 Assistant 时，才将旧 Turn 视为已完成。
从最早的已完成旧 Turn 开始删除，每删一轮重新估算，足够即停止。

所有 System Messages 保留且顺序不变；当前 User 及其后的全部消息受到保护。
没有 User 归属的完整前缀、没有最终 Assistant 回答的旧 Turn 均保守保留。
这种前缀兼容已有 Runtime 输入隔离测试，不会被静默裁剪。

工具调用批次必须紧随完整结果集合；允许批次内结果乱序。
拒绝孤立结果、重复结果、未知关联 ID、缺失结果、批次中间插入其他角色、
非 Assistant 携带 tool_calls、非 Tool 携带 tool_call_id 和其他非法字段结构。
同一批次调用 ID 必须唯一；不同批次沿用原协议，允许 ID 重用，结果只匹配当前批次。
即使异常位于可裁剪旧历史，也拒绝输入，不通过删除掩盖协议错误。

成本计算采用实际模型输入字段的 JSON UTF-8 字节数，每字节计一个估算单位：

- messages 包含角色、正文、原始工具参数、调用 ID、结果关联 ID；
- tools 包含 function 包装、名称、描述和完整 parameters schema；
- 额外每条消息 16 单位、每个工具 16 单位、每次请求 32 单位协议开销。

`available_input = input_budget - output_reserve - safety_margin`。
默认值分别为 32768、4096、1024，可用输入为 27648 估算单位。
这是本地应用策略，不表示任何模型的真实上下文上限。
input_budget 在该配置中表示扣除预留前的总额度；等于可用预算时允许发送。
值必须是非负整数（拒绝 bool），且可用空间必须为正。
输出预留只参与本地估算，不改变 Adapter，不设置实际输出 max_tokens。

删除全部可删除 Turn 后仍超额，抛出 ContextBudgetExceededError。
不会调用该轮 LLM、截断 JSON/工具结果、反馈 ToolFailure 或自动重试。
若前一轮工具已经执行，这些既有副作用不会回滚；本轮超额发生在下一次 LLM 之前。

配置示例：

```python
runtime = build_agent_runtime(
    llm=fake_llm,
    context_budget=ContextBudget(
        input_budget=16000,
        output_reserve=2000,
        safety_margin=1000,
    ),
)
```

## 5. 测试证据

### 初步验收（历史记录）

- ContextBuilder 独立测试覆盖预算边界、中文 UTF-8、原始参数、schema 成本、
  所有系统消息、最早完整 Turn、多工具/多轮链、输入校验和深拷贝。
- Runtime 测试覆盖每轮固定起点、完整结果 History、首轮及追加工具后的预算错误、
  首轮足够后续才裁剪、重复使用同一 Runtime/Builder 的跨 Run 隔离。
- 原有末轮工具跳过、工具已知/未知错误、取消、日志和请求快照测试继续通过。
- 新增 54 个参数化展开后的测试项（43 个 Builder、8 个 Runtime、3 个 Bootstrap）。

相关回归命令：

```powershell
.\.venv\Scripts\python.exe -u -m pytest tests/agent_runtime tests/tools -q
```

最终结果：**298 passed**，没有真实模型 API 调用。
首次回归曾出现未改动 Adapter 测试的 1 秒超时：SDK 首次获取 Windows 平台信息
耗时过长；复跑相关测试及全仓测试均未重现，未修改 Adapter 或测试超时。
沙箱内 asyncio 用例停在首次调用，最终异步回归在获准的沙箱外环境执行。

全仓原始命令 `python -u -m pytest -q`：
**668 passed, 35 skipped, 4 errors**。
该轮在最终新增用例完成前启动；最终版本由上述相关回归及下述全仓回归覆盖。
四个错误全部来自未改动的 `tests/memory/test_memory_reader_v2_e2e.py`，
原因均为 `fixture 'reader' not found`，与既有
`docs/testing/phase5_final_acceptance_report.md` 的记录一致。
35 项跳过为既有 opt-in 测试，不代表真实 PG/模型集成已经验证。

最终代码排除该已知手工测试文件后的全仓命令：

```powershell
.\.venv\Scripts\python.exe -u -m pytest -q --ignore=tests/memory/test_memory_reader_v2_e2e.py
```

结果：**669 passed, 35 skipped**，无失败或错误。

### Finalization 本轮复核（2026-10-28）

已读取当前 ContextBuilder、Runtime/Bootstrap/异常改动及相关测试，确认与上述策略一致。
开始收口时分支为 main，HEAD 和 fetch 后的 origin/main 均为
`e0224b0147d30098ffa79a2e7ef7debb4af3e945`，ahead/behind 为 0/0。
工作区仅有预期 8 个 Module 5 文件，无暂存项、冲突或无关修改。
未发现独立 Roadmap/Stage 进度文件，因此仅同步 README 中已有的进度与 Roadmap。
本轮未修改功能代码或测试代码。

本轮实际执行相关回归：

```powershell
.\.venv\Scripts\python.exe -u -m pytest tests/agent_runtime tests/tools -q
```

- 首跑：**297 passed, 1 failed**。未改动的
  `test_real_sdk_mock_transport_and_sdk_response_model` 在 SDK 获取 Windows 平台信息时
  超过其 1 秒超时，随后 Mock 错误翻译器导致 TypeError，与历史已记录的偶发问题一致。
- 原样复跑：**298 passed**（3.64 秒），没有修改 Adapter、测试超时或测试断言。
- 本轮实际执行排除已知手工测试文件的全仓回归：

  ```powershell
  .\.venv\Scripts\python.exe -u -m pytest -q --ignore=tests/memory/test_memory_reader_v2_e2e.py
  ```

  结果：**669 passed, 35 skipped**（46.02 秒），无失败或错误。35 项为既有 opt-in
  测试跳过，未开启真实 PG/模型验证。
- 原始不带 ignore 的全量 pytest 本轮未重跑；前述 4 个 reader fixture errors
  属历史验证记录，已知环境问题仍未解决，未修改 Memory 模块。

真实模型极限容量和既有 opt-in 集成仍不在本轮验证范围；不能将排除文件后的结果
表述为无条件全仓通过。

## 6. 已知限制与验收结论

- 近似估算不是精确 token 计数，不能严格保证模型 API 不拒绝超长输入。
  输出预留不是实际输出长度限制，部署时需按所用模型另行确认预算策略。
- 删除旧 Turn 可能让“那武汉呢？”这类省略式请求丢失背景；V1 不恢复语义，
  不做摘要或检索补全。
- 只支持标准新用户请求，不支持后台自主任务、工具链断点恢复或继续已完成请求。
  这是明确的输入兼容性收紧；仓库现有正常调用方和既有测试仍兼容。
- 无归属前缀、未完成旧 Turn 和所有 System 消息可能使上下文提前超额。
- 每次 build 校验全部 History，逐轮裁剪后重新估算；长历史可能有二次方成本，
  V1 优先可读性，没有增加复杂缓存或索引体系。
- 没有验证真实模型上下文边界或真实模型 token 使用量，自动化测试使用 Fake/Mock。

Module 5 实现限于 ContextBuilder V1、最小 Runtime/Bootstrap 接线、相关测试及本报告。
Finalization 仅更新本报告与 README，最终 Git Diff 涉及 9 个文件：
3 个新增（构造器、独立测试、本报告）、6 个增量修改（包含 README）。
`git diff --check` 通过；最终提交及推送状态见本次收口交付记录。
没有重构 Agent Loop，没有更改数据库、Memory、RAG、ChatService 或 Adapter，
没有新增第三方依赖。Module 5 V1 正式验收完成并冻结：**COMPLETE / FROZEN**。
全量 pytest 的既有手工测试错误、SDK 首次平台检测偶发超时与真实模型预算边界
保留为已知限制，不声称全仓原始命令完全通过。
