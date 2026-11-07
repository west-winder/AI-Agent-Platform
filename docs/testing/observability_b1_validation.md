# Stage 3 Module 8 Part B1 — Observability V1 验证记录

日期：2026-11-07（Asia/Shanghai）。状态：**B1 COMPLETE / Runtime Observability V1**。

## 改动范围

| 文件 | 作用 |
| --- | --- |
| backend/agent_runtime/observability.py | 不可变事件与 RunContext、Observer、JSON LoggingObserver、NoOpObserver、安全发送 |
| backend/agent_runtime/runtime.py | 局部 Run 状态、LLM/工具关联与事件、完整工具观测边界、安全诊断 |
| backend/agent_runtime/bootstrap.py | 可选 Observer 依赖注入 |
| tests/agent_runtime/test_observability.py | 44 项离线观测专项用例（含参数化） |
| docs/observability_v1.md | 架构、事件字段、计数规则及扩展点说明 |

未改动 Evaluation、MCP Host、Dispatcher、Registry、Tool Contract 或
LLMPort、ToolContext、AgentRunResult 公共契约。测试不调用真实 LLM 或远程工具。

## B1 验收映射

| 要求 | 专项测试 |
| --- | --- |
| 同一 Runtime 并发 Run，run_id 及关联隔离 | test_concurrent_runs_on_same_runtime_keep_all_associations_separate |
| 重试共用 llm_call_id，逻辑调用与实际尝试区分 | test_retries_share_logical_id_and_report_attempts |
| 多工具共用正确 parent_llm_call_id，保留模型原始 ID | test_multiple_tools_keep_raw_ids_and_share_parent |
| ToolFailure 正常完成，循环继续 | test_tool_failure_completes_and_loop_continues：解析、Schema、未知工具、业务异常四种来源 |
| 非预期工具异常保留原始对象，停止后续工具 | test_unexpected_tool_exception_retains_identity_and_stops_batch |
| 观测覆盖序列化和 History 写入异常 | test_tool_boundary_includes_serialization_and_history_write |
| Observer OSError 和备用诊断故障不影响返回或原异常 | test_observer_failure_and_failed_diagnostic_do_not_change_result |
| 默认 LoggingObserver 与共享 Handler 故障隔离 | test_logging_observer_handler_failure_is_isolated、test_shared_log_handler_failure_does_not_override_business_result |
| LLM/Tool 执行和重试等待中主动取消继续传播 | test_active_cancellation_has_cancel_events_and_propagates |
| 不捕获 BaseException | test_safe_emit_does_not_catch_base_exception |
| 预算超时与主动取消区分，无重复失败或虚增尝试 | test_budget_expiry_is_failure_not_cancellation_without_duplicate_attempts |
| 两种正常 Run 状态只记录一次 run.finished | test_normal_terminal_status_is_emitted_exactly_once |
| 输入校验在观测之前 | test_input_validation_precedes_observation |
| UTC 事件、属性复制冻结、拒绝业务字段/嵌套可变对象 | test_events_and_run_context_are_immutable_and_utc、test_event_rejects_non_metadata_and_mutable_values、test_event_rejects_naive_timestamp |
| 墙钟回拨不改变耗时，工具耗时包括序列化和 History | test_elapsed_time_uses_monotonic_even_if_wall_clock_moves_back |
| JSON 日志及故障诊断不含 Prompt/回答/Arguments/Result/原始异常消息 | test_json_logs_and_diagnostics_exclude_all_raw_content |
| Bootstrap 注入、默认 LoggingObserver 和显式 NoOpObserver | test_bootstrap_observer_injection_default_and_noop |

## 测试环境

初次运行发现主项目 .venv 缺少 requirements.txt 已声明的 mcp==1.26.0 和
jsonschema==4.26.0，已按固定版本安装。真实 stdio 用例引用的
mcp_servers/web_page_fetch/.venv 也缺失，需要按该 Server 既有 requirements.txt
恢复，最终使用已安装的 uv 成功安装。环境准备仅写入忽略的虚拟环境目录，
没有修改依赖声明或测试逻辑。

受限沙箱中 pytest 的默认临时目录和 mode=700 测试目录出现 WinError 5，
因此最终命令在获得执行权限后运行，并将 basetemp 指向本工作区 .pytest_cache。
该环境问题没有通过跳过测试或修改生产代码处理。

## 最终专项命令

```powershell
.\.venv\Scripts\python.exe -m pytest tests/agent_runtime tests/mcp_host tests/evaluation tests/tools -q --basetemp=D:/AI_Agent_Platform/.pytest_cache/m8b1-final
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

结果：**947 passed in 16.95s，0 failures/errors/skipped**。

| 测试域 | 通过数 |
| --- | ---: |
| Agent Runtime（原 372 项 + 新增观测 44 项） | 416 |
| MCP Host（包含真实 stdio） | 150 |
| Evaluation V1 | 209 |
| Tool System | 172 |
| 合计 | 947 |

pip check：No broken requirements found。git diff --check 通过。

准备环境期间的一次组合执行为 943 passed / 4 failed，四项均在缺失独立 Server
Python 可执行文件时失败；恢复其既有依赖后，最终同一完整测试集合全部通过。
未修改或跳过这些测试。

本次验收范围为上述相关回归；未重跑 Memory、RAG 等无关领域的全仓测试，
不把历史全仓问题计入本次结论。未执行真实 LLM、Judge 或公网 MCP 调用。
