# Stage 3 Roadmap / 当前状态

更新日期：2026-11-07（Asia/Shanghai）。

| 范围 | 状态 | 证据 / 边界 |
| --- | --- | --- |
| Module 1–4：Tool Contract、Function Calling、Registry / Dispatcher、Minimal Agent Loop | COMPLETE | 已有 Runtime 和 Tools 回归保留 |
| Module 5：State / Context | COMPLETE / FROZEN | [ContextBuilder V1](testing/context_builder_v1_validation.md) |
| Module 6：Error Handling / Retry / Limits | COMPLETE (V1) | [Module 6 验收](testing/module6_v1_validation.md) |
| Module 7：MCP Host Integration | COMPLETE / FROZEN | [Module 7 最终验收](testing/module7_mcp_host_final_acceptance.md) |
| Module 8 Part A：Evaluation V1 | COMPLETE / FROZEN（Offline V1） | [A4 验收](testing/evaluation_a4.md)；真实 Judge 校准不在已完成范围 |
| Module 8 Part B1：Logging / Tracing | COMPLETE / FROZEN | [B1 验收](testing/observability_b1_validation.md)，独立提交 ed614ec |
| Module 8 Part B2：Metrics V1 | COMPLETE / FROZEN | [指标目录](metrics_catalog.md)、[B2 设计](metrics_v1.md) |
| Module 8 整体 | COMPLETE / FROZEN | [最终验收](testing/module8_final_acceptance.md) |
| Stage 3 整体 | IN PROGRESS / 待用户最终确认 | 本轮只结束 Module 8，不自动开始系统性复盘 |

Module 8 的工程范围已闭合：离线评测、事件追踪、进程内指标和相关回归。
这不表示已完成真实模型质量校准、数据库/远程监控、多进程指标汇总或旧 ChatService
到 AgentRuntime 的完整迁移，也不表示发布新版本或开始 Stage 4。

既有阶段验收报告中的 NEXT/IN PROGRESS 是当时的历史状态；当前路线图以本文件和
README 为准。Evaluation、MCP Host 和其他已冻结模块的实现不因文档同步而修改。

后续由用户确认 Module 8 交付后，另行开启 Stage 3 复盘；本轮到此停止。
