# Stage 3 Roadmap / 当前状态

更新日期：2026-11-12（Asia/Shanghai）。阶段版本：**v0.5 — Tool Calling & Agent Runtime Foundation**。

Stage 0～2 及 Stage 3 均已 **COMPLETE / FROZEN**；用户已确认 Stage 3 整体收尾。

| 范围 | 状态 | 证据 / 边界 |
| --- | --- | --- |
| Module 1–4：Tool Contract、Function Calling、Registry / Dispatcher、Minimal Agent Loop | COMPLETE / FROZEN | 已有 Runtime 和 Tools 回归保留 |
| Module 5：State / Context | COMPLETE / FROZEN | [ContextBuilder V1](testing/context_builder_v1_validation.md) |
| Module 6：Error Handling / Retry / Limits | COMPLETE / FROZEN（V1） | [Module 6 验收](testing/module6_v1_validation.md) |
| Module 7：MCP Host Integration | COMPLETE / FROZEN | [Module 7 最终验收](testing/module7_mcp_host_final_acceptance.md) |
| Module 8 Part A：Evaluation V1 | COMPLETE / FROZEN（Offline V1） | [A4 验收](testing/evaluation_a4.md)；真实 Judge 校准不在已完成范围 |
| Module 8 Part B1：Logging / Tracing | COMPLETE / FROZEN | [B1 验收](testing/observability_b1_validation.md)，独立提交 ed614ec |
| Module 8 Part B2：Metrics V1 | COMPLETE / FROZEN | [指标目录](metrics_catalog.md)、[B2 设计](metrics_v1.md) |
| Module 8 整体 | COMPLETE / FROZEN | [最终验收](testing/module8_final_acceptance.md) |
| Stage 3 整体 | COMPLETE / FROZEN | Module 1～8 已交付；知识复习、文档同步与 v0.5 阶段收尾 |
| Stage 4：LangGraph + Agent Workflow | NEXT | 尚未完成开发；本轮不启动 |

Module 8 的工程范围已闭合：离线评测、事件追踪、进程内指标和相关回归。
这不表示已完成真实模型质量校准、数据库/远程监控、多进程指标汇总或旧 ChatService
到 AgentRuntime 的完整迁移。RAG Tool 与 LangGraph Workflow 尚未实现。

最近一次既有加固验收见 [Module 8 冻结后加固记录](testing/module8_post_freeze_hardening.md)：
Runtime、MCP Host、Evaluation 和 Tools 组合回归 **1032 passed，0 failures/errors/skipped**。
这是两项修复后的单次验收结果，不与较早的 978 项或专项结果累加；本轮没有重新运行测试。

既有阶段验收报告中的 NEXT/IN PROGRESS 是当时的历史状态；当前路线图以本文件和
README 为准。Evaluation、MCP Host 和其他已冻结模块的实现不因文档同步而修改。

阶段交付包含 [Stage 3 核心专业知识复习手册](stage3_knowledge_review.md)。手册记录了
2026-11-11 核对时的进度背景，当前整体状态以本文为准；`max_iterations` 默认值 5
已与 Runtime 和 Bootstrap 代码核对一致。

下一阶段为 **Stage 4 — LangGraph + Agent Workflow（NEXT）**：StateGraph / Node / Edge、
Conditional Routing、Workflow 编排、Checkpoint / 持久化、Interrupt / Human-in-the-loop
及复杂分支与多步骤任务处理。以上均为后续计划，本次仅结束 Stage 3 工程收尾。
