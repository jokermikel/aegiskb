# AegisKB 企业知识助手

AegisKB 是一个面向企业制度问答的安全型知识库助手原型。它将授权检索、Qwen 问答、高风险操作审批和审计日志串在同一条请求链路中，支持无 API Key 的本地演示模式。

## 功能

- 企业制度问答：按部门和文档密级过滤检索结果，并返回引用片段；授权过滤在 SQL 与内存两侧各做一次，越权判定以内存侧的权威校验为准。
- RBAC：内置 `employee`、`finance`、`admin` 角色及三档数据密级；文档部门支持多部门与子树通配（如 `engineering,it`、`engineering/*`）。
- 审批前置：导出、删除、发送/分享等高风险**指令**只创建审批单，批准前不会执行；制度咨询（例如「导出数据的审批流程是怎样的？」）会被识别为提问而不是指令。
- 审批列表：管理员查看并处理全部审批单，普通用户只能查看自己提交的状态；列表分页返回，总数走响应头。
- 审计日志：记录登录、问答、文档入库和审批决策。
- Qwen 思考模式与工具调用参数兼容处理。
- Web 对话界面：新建、重命名、删除和持久化最近对话，并支持 API Key 连通性测试。

## 技术栈

Python 3.11+（Docker 镜像固定 `python:3.11-slim`）、FastAPI、Uvicorn、SQLAlchemy 2.x、Alembic、PyJWT、Pydantic Settings、OpenAI Python SDK，以及原生 HTML/CSS/JavaScript 前端。默认使用 SQLite，Docker Compose 使用 PostgreSQL 16。

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

## 演示数据

空数据库首次启动时会自动导入演示数据：**10 个账号、50 篇企业制度文档、100 条审批单样例**。
文档覆盖全员及八个部门、三档密级；审批单包含 55 条待处理、25 条已批准、20 条已拒绝。

| 邮箱 | 角色 | 部门 | 访问密级 | 可见制度 |
| --- | --- | --- | --- | --- |
| `employee@example.com` | employee | engineering | INTERNAL | 10 |
| `finance@example.com` | finance | finance | CONFIDENTIAL | 15 |
| `admin@example.com` | admin | security | CONFIDENTIAL | 13 |
| `hr@example.com` | employee | hr | INTERNAL | 11 |
| `legal@example.com` | employee | legal | CONFIDENTIAL | 12 |
| `it@example.com` | employee | it | CONFIDENTIAL | 10 |
| `sales@example.com` | employee | sales | INTERNAL | 8 |
| `ops@example.com` | employee | ops | INTERNAL | 9 |
| `auditor@example.com` | admin | security | INTERNAL | 9 |
| `controller@example.com` | finance | finance | INTERNAL | 8 |

所有账户密码均为 `demo-password`。数据集维护在 `app/seed_data.py`，审批单使用固定随机种子生成，
可重复复现；若要重新灌入数据，删除 `aegiskb.db` 后重启服务即可。

## 主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/auth/login` | 登录并获取 JWT |
| `GET` | `/api/v1/auth/me` | 当前用户信息 |
| `POST` | `/api/v1/agent/chat` | 问答或创建审批单 |
| `POST` | `/api/v1/settings/qwen/test` | 测试百炼 API Key |
| `GET` | `/api/v1/approvals` | 分页查看审批单（`page`、`page_size`，总数与待处理数见 `X-Total-Count` / `X-Total-Pending`） |
| `POST` | `/api/v1/approvals/{id}/decision` | 管理员批准/拒绝 |
| `POST` | `/api/v1/documents` | finance/admin 文档入库 |
| `GET` | `/api/v1/audit-logs` | 管理员查看审计日志（`limit` 取值范围 1–200） |

## 配置

复制 `.env.example` 为 `.env` 后按需设置：`DATABASE_URL`、`JWT_SECRET`、`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`QWEN_MODEL`、`QWEN_REASONING_EFFORT` 和 `AUTO_CREATE_SCHEMA`。API Key 只应保存在本地 `.env` 或当前浏览器会话中，勿提交到 Git。

启用思考模式时工具调用使用 `tool_choice=auto`；关闭思考模式时才使用强制工具调用，以避免上游参数冲突。

## 数据库迁移

开发与演示环境默认允许启动时自动建表（`AUTO_CREATE_SCHEMA=true`）。**生产环境应设为 `false`，改用 Alembic 管理结构：**

```bash
alembic upgrade head        # 应用迁移
alembic revision --autogenerate -m "描述本次结构变更"   # 生成新迁移
```

启动时会比对我模型定义与实际库结构（`detect_schema_drift()`）。一旦发现缺表或缺字段，服务会直接报错并提示执行 `alembic upgrade head`，不再出现「模型改了但库里静默不生效」的情况。

## 测试

```bash
pytest -q
```

测试使用独立数据库并清空 API Key，覆盖授权边界、审批流转、审批前置语义、数据规模、分页与 `data/evals.jsonl` 评测样例，共 22 个用例。

## 安全说明

`.env`、数据库、虚拟环境和缓存已加入 `.gitignore`，同时通过 `.dockerignore` 排除在镜像之外（`COPY . .` 不会再把真实 API Key 和 81 MB 虚拟环境打进镜像层）。

`.gitattributes` 将仓库行尾统一为 LF，并已执行 `git config core.autocrlf false` 由该文件接管，避免混合行尾让每次改动都表现为「整文件重写」。

示例 JWT 密钥和数据库密码只适用于本地演示；生产部署应补充密钥管理、限流、日志脱敏和真实审批执行器。审批前置目前基于关键词与语气判定，仍属于启发式规则，生产环境应改为由模型分类加人工复核双重确认。

## 目录结构

```text
app/                 # FastAPI 服务、Agent、数据模型、认证和前端
app/seed_data.py     # 演示数据集（10 账号 / 50 篇制度 / 100 条审批单）
app/static/          # Web 界面
data/evals.jsonl     # 评测样例（已在 tests/test_api.py 中接入自动断言）
migrations/          # Alembic 迁移脚本
tests/               # API、安全行为、授权矩阵与数据规模测试
```
