# AI Agent Platform

基于 FastAPI + SQLAlchemy + DeepSeek API 构建的 AI Agent 应用平台。

本项目目标是实现一个类似 ChatGPT 的多 Agent 对话系统，支持 Agent 管理、Conversation 管理、消息持久化以及基于 Agent 角色设定的智能对话。

当前版本：`v0.1`

---

# 项目简介

AI Agent Platform 是一个面向 AI 应用开发学习的全栈 Agent 项目。

项目重点不在模型训练，而是围绕已有大语言模型 API，构建完整的 Agent 应用系统，包括：

* Agent角色管理
* 多会话管理
* 消息历史记录
* Agent快照机制
* LLM调用封装
* 前后端分离架构

---

# 技术栈

## Backend

* Python
* FastAPI
* SQLAlchemy
* SQLite
* Pydantic

## LLM

* DeepSeek API

## Frontend

* Gradio

## Architecture

```
Frontend (Gradio)

        |

FastAPI Router

        |

Service Layer

        |

SQLAlchemy ORM

        |

Database
```

---

# 项目结构

```
AI_Agent_Platform

├── backend
│   ├── models        # 数据库ORM模型
│   ├── schemas       # Pydantic数据模型
│   ├── services      # 业务逻辑层
│   ├── routers       # API接口
│   ├── database      # 数据库配置
│   └── main.py       # FastAPI入口
│
├── frontend
│   └── app.py        # Gradio前端
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 已实现功能

## 1. Agent系统

支持创建和管理不同类型的 AI Agent。

Agent包含：

* 名称
* System Prompt
* 创建时间
* 更新时间

例如：

```
科研助手

你是一名科研领域专家，
负责帮助用户进行论文分析。
```

---

## 2. Conversation系统

实现独立会话管理。

每个Conversation关联：

* User
* Agent
* Message

支持：

* 创建新聊天
* 查看历史聊天
* 加载已有Conversation

---

## 3. Agent Snapshot机制

为了保证历史对话稳定性，引入 Agent 快照设计。

创建Conversation时保存：

```json
{
    "name": "科研助手",
    "system_prompt": "你是一名科研助手"
}
```

即使之后修改Agent：

历史Conversation仍保持原始Agent设定。

---

## 4. Message系统

实现消息持久化。

保存：

* 用户消息
* AI回复
* Token信息
* 创建时间

数据库结构：

```
Message

id

conversation_id

role

content

created_at
```

---

## 5. DeepSeek LLM接入

通过环境变量管理API配置。

示例：

```
.env

DEEPSEEK_API_KEY=your_key
```

代码通过环境变量读取，不直接保存密钥。

---

# 数据库设计

当前核心数据模型：

```
User

  |

  |--- Agent

  |

  |--- Conversation

              |

              |--- Message

```

核心思想：

* Agent负责定义AI角色
* Conversation负责一次聊天上下文
* Message负责具体消息记录

---

# 运行方式

## 1. 克隆项目

```bash
git clone your_repository_url

cd AI_Agent_Bootcamp
```

## 2. 创建虚拟环境

```bash
python -m venv venv
```

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 4. 配置环境变量

创建：

```
.env
```

添加：

```
DEEPSEEK_API_KEY=your_key
```

## 5. 启动FastAPI

```bash
uvicorn backend.main:app --reload
```

## 6. 启动Gradio

```bash
python frontend/app.py
```

---

# 当前限制

v0.1版本主要用于验证 Agent 应用架构。

当前暂未实现：

* 用户登录认证
* JWT权限系统
* PostgreSQL
* Redis缓存
* Memory系统
* RAG知识库
* LangGraph Agent工作流
* Docker部署

---

# Roadmap

## v0.2 Memory系统

计划实现：

* 用户长期记忆
* Memory存储
* Memory召回

## v0.3 RAG系统

计划实现：

* 文件上传
* Embedding
* 向量数据库
* 知识库问答

## v0.4 Agent Workflow

计划实现：

* LangGraph
* 多Agent协作
* 工具调用

## v1.0

完整AI Agent应用平台。

---

# License

This project is for learning and engineering practice.
