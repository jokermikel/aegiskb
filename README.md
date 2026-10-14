# AegisKB 企业知识助手

AegisKB 是一个面向企业制度问答的安全型知识库助手原型。它将授权检索、Qwen 问答、高风险操作审批和审计日志串在同一条请求链路中，支持无 API Key 的本地演示模式。

## 功能

- **混合检索**：语义向量召回 + 关键词召回，经 RRF 融合与精确词信号轻量重排；按句边界分块，不再从句子中间硬切。
- **可切换的向量后端**：faiss（有 faiss-cpu 时优先）/ NumPy / PostgreSQL pgvector；无 API Key 或上游失败时自动降级到确定性的本地向量，全新检出、离线也能跑。
- **企业制度问答**：按部门和文档密级过滤检索结果，并返回引用片段。授权裁剪发生在**相似度排序之前**的 SQL 层，是检索与索引共用的唯一权限入口。
- **RBAC**：内置 `employee`、`finance`、`admin` 角色及三档数据密级；文档部门支持多部门与子树通配（如 `engineering,it`、`engineering/*`），入库时会校验部门合法性。
- **审批前置**：导出、删除、发送/分享等高风险**指令**只创建审批单，批准前不会执行；制度咨询（例如「导出数据的审批流程是怎样的？」）会被识别为提问而不是指令。
- **审批列表**：管理员查看并处理全部审批单，普通用户只能查看自己提交的状态；支持分页与**服务端**状态/关键词筛选。
- **登录限流**：同一「邮箱 + 客户端 IP」在窗口内连续失败达到阈值后返回 429。
- **审计日志**：记录登录、问答、文档入库和审批决策；问答日志会带上检索后端、耗时、候选数等指标，便于回答「这次为什么没召回」。
- **离线评测与压测**：带标注的 `data/evals.jsonl` 可一键跑出 Recall@K / HitRate@K / MRR 并持久化，作为发布质量闸门。
- **回答来源可见**：每条回答标注「模型生成 / 本地检索兜底 / 已转人工审批」，用户能分辨信任级别。
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
| `GET` | `/api/v1/approvals` | 分页查看审批单（`page`、`page_size`、`status`、`keyword`；总数与待处理数见 `X-Total-Count` / `X-Total-Pending`） |
| `POST` | `/api/v1/approvals/{id}/decision` | 管理员批准/拒绝 |
| `POST` | `/api/v1/documents` | finance/admin 文档入库（会校验部门合法性并立即建立向量索引） |
| `GET` | `/api/v1/audit-logs` | 管理员查看审计日志（`limit` 取值范围 1–200） |
| `GET` | `/api/v1/retrieval/stats` | 管理员查看向量后端与索引新鲜度（不含任何文档正文） |
| `POST` | `/api/v1/retrieval/evaluations` | 管理员跑一次离线检索评测并持久化指标 |
| `GET` | `/api/v1/retrieval/evaluations` | 管理员查看历史评测记录 |

## 配置

复制 `.env.example` 为 `.env` 后按需设置：`DATABASE_URL`、`JWT_SECRET`、`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`QWEN_MODEL`、`QWEN_REASONING_EFFORT`、`QWEN_TIMEOUT_SECONDS`、`EMBEDDING_*`、`RETRIEVAL_*` 和 `AUTO_CREATE_SCHEMA`。API Key 只应保存在本地 `.env` 或当前浏览器会话中，勿提交到 Git。

启用思考模式时工具调用使用 `tool_choice=auto`；关闭思考模式时才使用强制工具调用，以避免上游参数冲突。

模型请求强制带超时与重试上限：上游卡住时会降级到本地检索结果，而不是让请求无限期挂起。

## 检索与评测

```bash
python scripts/run_eval.py --top-k 4      # 跑评测集，写入 output/evaluation/latest.json
python scripts/benchmark.py               # 对运行中的实例做端到端压测
```

`data/evals.jsonl` 中只有声明了 `relevant_titles` 的用例才进入 Recall@K 分母；安全与审批用例留在同一份数据集里做更广的回归，但不干扰检索指标。`tests/test_api.py` 会把这份数据集整体接入自动断言，因此它同时是回归套件与质量闸门。

## 数据库迁移

开发与演示环境默认允许启动时自动建表（`AUTO_CREATE_SCHEMA=true`）。**生产环境应设为 `false`，改用 Alembic 管理结构：**

```bash
alembic upgrade head        # 应用迁移
alembic revision --autogenerate -m "描述本次结构变更"   # 生成新迁移
```

启动时会比对我模型定义与实际库结构（`detect_schema_drift()`）。一旦发现缺表或缺字段，服务会直接报错并提示执行 `alembic upgrade head`，不再出现「模型改了但库里静默不生效」的情况。

## 测试与 CI

```bash
ruff check .        # 代码检查
pytest -q           # 单元与安全行为测试
```

测试使用独立数据库、清空 API Key，并强制离线向量后端，覆盖授权边界、审批流转、审批前置语义、入库校验、分块边界、分页与筛选、登录限流、迁移一致性与 `data/evals.jsonl` 评测样例，共 34 个用例。

`.github/workflows/ci.yml` 会在每次推送时依次跑 ruff、`alembic upgrade head && alembic check`、pytest 与评测质量闸门。

## 安全说明

`.env`、数据库、虚拟环境和缓存已加入 `.gitignore`，同时通过 `.dockerignore` 排除在镜像之外（`COPY . .` 不会再把真实 API Key 和 81 MB 虚拟环境打进镜像层）。

`.gitattributes` 将仓库行尾统一为 LF，并已执行 `git config core.autocrlf false` 由该文件接管，避免混合行尾让每次改动都表现为「整文件重写」。

示例 JWT 密钥和数据库密码只适用于本地演示；生产部署应补充密钥管理、限流、日志脱敏和真实审批执行器。审批前置目前基于关键词与语气判定，仍属于启发式规则，生产环境应改为由模型分类加人工复核双重确认。

## 目录结构

```text
app/                 # FastAPI 服务、Agent、数据模型、认证和前端
app/chunking.py      # 按句边界分块的文本切分
app/embeddings.py    # DashScope / 离线哈希两种向量提供方
app/vectorstore.py   # faiss / NumPy / pgvector 三种向量后端
app/retrieval.py     # 混合检索与唯一的权限裁剪入口
app/evaluation.py    # 离线检索评测
app/departments.py   # 部门白名单与写法校验
app/seed_data.py     # 演示数据集（10 账号 / 50 篇制度 / 100 条审批单）
app/static/          # Web 界面
data/evals.jsonl     # 带标注的评测集（已接入自动断言）
migrations/          # Alembic 迁移脚本
scripts/             # run_eval.py / benchmark.py
tests/               # API、检索、安全行为、授权矩阵与数据规模测试
.github/workflows/   # CI（ruff + 迁移检查 + pytest + 评测闸门）
```

## 已知限制

- 结构漂移守卫只比对**表名与字段名**，不比对字段类型与索引。实测直接比对类型（即使用方言解析）在 52 个字段里仍会产生 5 处假报（`JSON` 列与自定义 `TypeDecorator` 的物理类型无法按字符串直接比较），放进启动守卫会误拦截。因此类型级变更仍需靠 `alembic check` 把关。
- 向量索引用进程内缓存 + 数据库持久化，多实例部署需要外置索引服务或共享缓存。
- 登录限流是进程内滑动窗口，多实例部署应换成 Redis 等共享存储。
- 混合检索的语义召回会让「无依据」的问题也返回若干**有权阅读**的近似文档；可信度由相似度阈值与提示词共同约束，仍需真实语料调优。
- 审批前置基于关键词与语气判定，属于启发式规则；生产环境应改为模型分类 + 人工复核双确认。
- 首次检索需要为空语料建立向量（50 个切片约需数十次 embedding 调用），生产环境应在入库流程中预热索引。
