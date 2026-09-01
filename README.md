# AegisKB 企业知识助手

AegisKB 是一个面向企业制度问答的安全型知识库助手原型。它将授权检索、Qwen 问答、高风险操作审批和审计日志串在同一条请求链路中，支持无 API Key 的本地演示模式。

## 功能

- 企业制度问答：按部门和文档密级过滤检索结果，并返回引用片段。
- RBAC：内置 `employee`、`finance`、`admin` 角色及三档数据密级。
- 审批前置：导出、删除、发送/分享等高风险意图只创建审批单，批准前不会执行。
- 审批列表：管理员查看并处理全部审批单，普通用户只能查看自己提交的状态。
- 审计日志：记录登录、问答、文档入库和审批决策。
- Qwen 思考模式与工具调用参数兼容处理。
- Web 对话界面：新建、重命名、删除和持久化最近对话，并支持 API Key 连通性测试。

## 技术栈

Python 3.11、FastAPI、Uvicorn、SQLAlchemy 2.x、PyJWT、Pydantic Settings、OpenAI Python SDK，以及原生 HTML/CSS/JavaScript 前端。默认使用 SQLite，Docker Compose 使用 PostgreSQL 16。

## 快速开始

```bash
cd aegiskb
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux：source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env       # macOS/Linux：cp .env.example .env
uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000> 使用 Web 界面，API 文档位于 <http://127.0.0.1:8000/docs>。不配置 `DASHSCOPE_API_KEY` 时会使用本地检索回答。

### Docker

```bash
copy .env.example .env       # macOS/Linux：cp .env.example .env
docker compose up --build
```

## 演示账户

所有账户密码均为 `demo-password`：

| 邮箱 | 角色 | 部门 | 访问密级 |
| --- | --- | --- | --- |
| `employee@example.com` | employee | engineering | INTERNAL |
| `finance@example.com` | finance | finance | CONFIDENTIAL |
| `admin@example.com` | admin | security | CONFIDENTIAL |

空数据库首次启动时会自动创建上述账户和示例制度文档。

## 主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/auth/login` | 登录并获取 JWT |
| `GET` | `/api/v1/auth/me` | 当前用户信息 |
| `POST` | `/api/v1/agent/chat` | 问答或创建审批单 |
| `POST` | `/api/v1/settings/qwen/test` | 测试百炼 API Key |
| `GET` | `/api/v1/approvals` | 查看审批单 |
| `POST` | `/api/v1/approvals/{id}/decision` | 管理员批准/拒绝 |
| `POST` | `/api/v1/documents` | finance/admin 文档入库 |
| `GET` | `/api/v1/audit-logs` | 管理员查看审计日志 |

## 配置

复制 `.env.example` 为 `.env` 后按需设置：`DATABASE_URL`、`JWT_SECRET`、`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`QWEN_MODEL` 和 `QWEN_REASONING_EFFORT`。API Key 只应保存在本地 `.env` 或当前浏览器会话中，勿提交到 Git。

启用思考模式时工具调用使用 `tool_choice=auto`；关闭思考模式时才使用强制工具调用，以避免上游参数冲突。

## 测试

```bash
pytest -q
```

测试使用独立数据库并清空 API Key。手工测试步骤见 `测试.md`，从零复现说明见 `从零复现项目指南.md`。

## 安全说明

`.env`、数据库、虚拟环境和缓存已加入 `.gitignore`。示例 JWT 密钥和数据库密码只适用于本地演示；生产部署应补充密钥管理、数据库迁移、限流、日志脱敏和真实审批执行器。

## 目录结构

```text
app/                 # FastAPI 服务、Agent、数据模型、认证和前端
app/static/          # Web 界面
data/evals.jsonl     # 评测样例
tests/               # API 与安全行为测试
```
