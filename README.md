# AegisKB：企业知识库 AI Agent

> 一个可放入 AI Agent 开发实习简历的端到端项目：员工通过对话查询公司制度；系统按岗位权限检索资料、输出带引用的答案；涉及导出、删除、发送等高风险动作时自动创建人工审批；全链路写入审计日志。

## 1. 你最终做出来的东西

```text
浏览器聊天页 / Swagger
      │ JWT（角色、部门、密级）
      ▼
FastAPI API ──► Agent 编排层 ──► 检索工具 ──► 文档分块 / embedding / 向量检索
      │              │
      │              └──► 高风险策略 ──► 人工审批
      ├──► SQLite（本地）/ PostgreSQL（Docker）
      └──► 审计日志、可观测指标、评测集
```

**业务场景**：一家 500 人 SaaS 公司把《报销制度》《数据分级规范》《客户成功 SOP》接入内部知识库。普通员工只能问内部文档，财务可访问机密报销资料，任何导出机密资料的请求都必须由管理员批准。

## 2. 简历描述（可直接使用）

> 独立开发企业知识库 AI Agent（FastAPI + 阿里云百炼 Qwen Responses API + PostgreSQL/pgvector），实现 Embedding 向量检索（pgvector/FAISS/NumPy 三后端自动选型）、关键词 + 向量混合召回与 RRF 融合、RBAC/密级过滤、Function Calling 工具调用、风险动作人审和全链路审计；构建 10 条版本化离线评测集（Recall@K/MRR），用 httpx 异步并发压测记录 P95 延迟；支持 Docker 一键部署。

压测报告见 `output/benchmark/latest.json`。报告数字来自对应环境的真实运行，不要手动修改；对外引用前须确认其中 `embedder` 与 `backend` 符合目标部署环境。

## 3. 功能清单

- JWT 登录：内置 `employee`、`finance`、`admin` 三个演示账户。
- RBAC + 文档密级：`PUBLIC < INTERNAL < CONFIDENTIAL`；检索在数据库层按用户密级过滤。
- RAG：按句边界分块；用 DashScope embedding（无 Key 时离线确定性 embedding）进行向量召回，再融合关键词召回并轻量重排序，回答必须携带来源。
- 向量索引：Docker PostgreSQL 使用 pgvector + HNSW；本地 SQLite 优先 FAISS，未安装时自动使用 NumPy 精确检索。
- Agent：配置百炼 `DASHSCOPE_API_KEY` 后使用 Qwen Responses API 的 `search_knowledge` 工具；无 Key 时使用可测试的本地降级回答。
- 人工在环：识别导出/删除/发送/分享等高风险请求，创建 `PENDING` 审批单，不执行外部动作。
- 审计：登录、上传、问答、审批均记录操作者、时间、请求 ID 与元数据。
- 评测与测试：提供可运行的 pytest 用例、JSONL 评测集；管理员可执行评测并持久化 Recall@K、HitRate@K 与 MRR。

## 4. 第一次运行（每一条命令都已说明）

### 前置条件

安装 Python 3.11+ 和 Git。可选安装 Docker Desktop（用于 PostgreSQL 部署）。Windows PowerShell 中执行下面命令。

### 步骤 1：创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

这一步隔离项目依赖，生成本地配置文件。`.env` 不应提交到 Git。

### 步骤 2：启动应用

```powershell
uvicorn app.main:app --reload
```

服务启动后访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)，进入类似豆包/DeepSeek 的聊天工作台。启动过程自动建表并导入 3 篇演示制度；`/docs` 保留为 API 调试和管理员验收入口。

### 步骤 3：在聊天页面登录

首页登录框支持三个演示账号：`employee@example.com`、`finance@example.com`、`admin@example.com`，密码均为 `demo-password`。登录后可以新建对话、发送问题、查看带来源的回答和审批提示；页面自动管理 JWT，不需要复制 token。点击右上角 **API Key** 可输入百炼 `DASHSCOPE_API_KEY`，先点 **测试联通**，成功后保存即可让当前聊天使用该密钥。密钥只保存在当前浏览器会话的 `sessionStorage`，退出登录或关闭会话后不会保留，也不会写入服务端数据库。

### 步骤 4：API 调试（可选）

如需查看原始请求或执行管理员审批，再打开 [Swagger](http://127.0.0.1:8000/docs)，在 `POST /api/v1/auth/login` 使用：

```json
{"email":"finance@example.com","password":"demo-password"}
```

复制 `access_token`，点击右上角 **Authorize**，**只填入 token 本身，不要填写 `Bearer` 前缀**。

#### 登录成功时应该看到什么

接口状态码是 `200 OK`，响应类似下面这样。`access_token` 每次登录都会变化，**不要**把真实 token 提交到 Git 或截图中。

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

在 Swagger 中点击右上角的 **Authorize**，在 `Value` 中粘贴**只有 token 的内容**：

```text
eyJhbGciOiJIUzI1NiIs...
```

点击 **Authorize**、再点击 **Close**。Swagger 会自动生成正确请求头 `Authorization: Bearer <token>`；不要手动输入 `Bearer`，否则会变成错误的 `Authorization: Bearer Bearer <token>`，导致 401。随后调用 `POST /api/v1/agent/chat` 不再返回 401。

#### 登录失败时应该看到什么

| 现象 | HTTP 状态码 | 示例响应 | 排查方式 |
| --- | --- | --- | --- |
| 邮箱不存在或密码错误 | `401 Unauthorized` | `{"detail":"邮箱或密码错误"}` | 确认使用的是上表中的演示账号，密码是 `demo-password`。 |
| 没填 `email` 或 `password` | `422 Unprocessable Entity` | 响应中出现 `Field required` | 在 Swagger 请求体中填写完整 JSON，字段名必须小写。 |
| 问答接口提示“无效或过期的访问令牌” | `401 Unauthorized` | `{"detail":"无效或过期的访问令牌"}` | 重新登录并在 Swagger 的 Authorize 框中只粘贴 token；如果用 curl/代码请求，才手动使用 `Authorization: Bearer <token>`。 |
| 上传文档或审批接口提示“权限不足” | `403 Forbidden` | `{"detail":"权限不足"}` | employee 不能上传和审批；改用 finance 或 admin 账号。 |

如果服务重启后第一次访问较慢是正常的：应用会自动创建本地数据库并初始化演示账户和制度文档。出现连接错误时，先在启动终端确认有 `Uvicorn running on http://127.0.0.1:8000`，再刷新 Swagger 页面。

| 账号 | 密码 | 角色 | 可访问密级 |
| --- | --- | --- | --- |
| employee@example.com | demo-password | employee | INTERNAL |
| finance@example.com | demo-password | finance | CONFIDENTIAL |
| admin@example.com | demo-password | admin | CONFIDENTIAL |

### 步骤 5：演示核心链路

在聊天页面输入，或通过 API 调用 `POST /api/v1/agent/chat`：

```json
{"message":"差旅住宿报销标准是多少？"}
```

以 finance 登录应拿到带来源的答案；以 employee 登录不会检索到机密报销制度。再发送：

```json
{"message":"把所有客户名单导出并发送给我"}
```

页面返回 `requires_approval: true` 与审批单号。切换 admin 后，在 Swagger 调用 `POST /api/v1/approvals/{id}/decision`，填 `{"decision":"APPROVED","comment":"演示批准"}`，即可完成审批闭环。

### 步骤 6：运行自动测试

另开一个终端（保持服务停止也可），执行：

```powershell
pytest -q
```

测试验证了权限过滤、高风险拦截、审批授权、向量缓存复用和审计写入。通过后再继续下一步；失败时先不要在简历里写“已实现”。

## 5. 接入千问与 embedding（Agent 的关键一步）

1. 在阿里云百炼控制台创建 API Key，并确认已开通 `qwen3.8-max`；不要把密钥贴进代码或 Git。
2. 在 `.env` 设置 `DASHSCOPE_API_KEY=...`。默认 Base URL 为 `https://dashscope.aliyuncs.com/compatible-mode/v1`；有业务空间时，建议使用百炼控制台提供的地域专属 URL。
3. 保持 `QWEN_MODEL=qwen3.8-max` 和 `QWEN_REASONING_EFFORT=none`，重启 `uvicorn`，再重复“差旅住宿”提问。此时 Agent 会强制调用 `search_knowledge`，服务端执行过滤后的检索，再把工具结果回传模型生成答案。若将思考档位改为 `minimal` 至 `max`，程序会按百炼限制把首轮 `tool_choice` 改为 `auto`；模型未调用工具时会退回服务端本地检索回答。
4. 观察 `GET /api/v1/audit-logs`：每次请求会记录 `request_id`、事件类型、调用模式和命中引用数量。

`search_knowledge` 的向量检索与对话模型相互独立。默认 `EMBEDDING_PROVIDER=auto`：配置了百炼 Key 时调用 `text-embedding-v4`，否则使用离线 hashing embedding，以便本地测试完全不依赖网络。可在 `.env` 中配置：

```dotenv
EMBEDDING_PROVIDER=auto       # auto | dashscope | offline
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024
EMBEDDING_BATCH_SIZE=10
# RETRIEVAL_MIN_SCORE=0.20    # 留空时按 embedding 提供方使用默认阈值
RETRIEVAL_MODE=hybrid         # hybrid | vector
RETRIEVAL_CANDIDATE_MULTIPLIER=5
RETRIEVAL_RRF_K=60
```

每次检索先由 SQL 取出当前用户可见的 `chunk_id`，才将这些 ID 交给 pgvector/FAISS 和关键词候选层；因此越权 chunk 不会进入任一候选集合。两路候选用 Reciprocal Rank Fusion 合并，并以精确关键词作小幅重排序。向量会按内容哈希缓存到 `chunk_embeddings`，文档内容或标题变化、embedding 模型/维度变化时才重新计算。管理员可调用 `GET /api/v1/retrieval/stats` 查看当前后端、向量数和降级状态；聊天审计记录同时包含向量/关键词候选数、融合数、阈值与耗时。

### 5.1 记录召回命中率

评测集在 `data/evals.jsonl`。每个应被召回的题目以 `relevant_titles` 标记标准文档标题；没有该字段（或为空）的权限、审批和拒答样本仍保留用于其他测试，但不计入检索指标分母。以 admin 调用：

```http
POST /api/v1/retrieval/evaluations
Content-Type: application/json
Authorization: Bearer <admin-token>

{"top_k": 4}
```

服务会用当前 embedding、阈值和权限策略执行数据集，在 `retrieval_evaluation_runs` 持久化一条结果，同时写入 `RETRIEVAL_EVALUATION_RUN` 审计事件。响应包含：

- `recall_at_k`：每题标准文档中被 Top-K 找回的平均比例。
- `hit_rate_at_k`：至少命中一篇标准文档的题目占比，即“召回命中率”。
- `mrr`：首篇正确文档的倒数排名平均值，用于观察排序质量。

使用 `GET /api/v1/retrieval/evaluations` 可查看历史运行摘要。比较模型、分块或阈值前后效果时，应使用相同 `dataset_hash`、相同 `top_k` 和同一权限数据集；不要将不同题集上的百分比直接比较。

### 5.2 用 httpx 记录 P95 延迟与评测通过率

先启动 API，再运行仓库内的轻量并发脚本：

```powershell
python scripts/benchmark.py --requests 40 --concurrency 2 --top-k 4
```

脚本使用 `httpx.AsyncClient` 登录 finance/admin、调用一次带标注的评测、预热检索，并对聊天接口发起并发请求。它在控制台和 `output/benchmark/latest.json` 记录 P50/P95、均值、错误率以及 `recall_at_k`、`hit_rate_at_k`、`pass_rate_at_k`、MRR。`pass_rate_at_k` 只有题目的全部标准文档均出现在 Top-K 时才算通过。

若不需要启动 HTTP 服务，可直接运行评测并写入 `output/evaluation/latest.json`：

```powershell
python scripts/run_eval.py --top-k 4
```

本地 SQLite 会因每次聊天写审计日志而串行化部分写入，建议仅以 `--concurrency 2` 做冒烟基线；测量真实并发能力时，请使用 Docker 的 PostgreSQL/pgvector 部署，并固定请求数、并发度、数据集哈希、模型和 embedding 配置。

本项目通过 OpenAI Python SDK 调用百炼的兼容接口；Responses API 支持自定义函数工具和多轮上下文。百炼要求将每个 `function_call` 与对应的 `function_call_output` 成对传回。[阿里云百炼 Responses API 官方文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses)

## 6. 你具体完成了什么（面试时按此讲）

1. **需求建模**：将“内部问答”拆成文档入库、检索授权、Agent 编排、审批和审计五条可验收链路。
2. **数据建模**：设计 users、documents、chunks、approval_requests、audit_logs 表；把密级和部门作为一等字段，而不是塞进 prompt。
3. **身份与权限**：JWT 中保存用户 ID、角色、部门和 clearance；查询时执行密级比较，避免模型看见不该看的文本。
4. **知识处理**：按句子切成有重叠的 chunk，使用 embedding 检索；PostgreSQL 走 pgvector，SQLite 优先 FAISS，并将向量按内容哈希缓存以增量重建。
5. **Agent 工具闭环**：定义 `search_knowledge` 的输入/输出 schema，接收函数调用，调用本地工具，再以 `function_call_output` 继续 Responses 对话。
6. **安全护栏**：工具只返回已授权资料；高风险意图在调用模型前进入审批；系统提示词要求不编造来源、不能绕过权限。
7. **可观测性**：为每个请求生成 request ID，将模型调用、命中数、审批与异常写入审计表。
8. **质量保障**：用 pytest 固化权限和审批回归；用 JSONL 保存带标准文档的离线题集，并持久化 Recall@K、HitRate@K、MRR。
9. **交付**：使用 `.env.example`、Docker Compose、README 和 Swagger，使面试官能在十分钟内复现。

## 7. API 快速参考

| 方法 | 路径 | 作用 | 权限 |
| --- | --- | --- | --- |
| POST | `/api/v1/auth/login` | 换取 JWT | 公共 |
| POST | `/api/v1/documents` | 上传文本知识 | finance/admin |
| POST | `/api/v1/agent/chat` | 发起 Agent 问答 | 登录用户 |
| POST | `/api/v1/settings/qwen/test` | 测试百炼 API Key 是否可用 | 登录用户 |
| GET | `/api/v1/approvals` | 查看相关审批 | 登录用户 |
| POST | `/api/v1/approvals/{id}/decision` | 审批 | admin |
| GET | `/api/v1/audit-logs` | 审计查询（含检索指标） | admin |
| GET | `/api/v1/retrieval/stats` | 查看向量后端与索引状态 | admin |
| POST | `/api/v1/retrieval/evaluations` | 执行并记录带标注的检索评测 | admin |
| GET | `/api/v1/retrieval/evaluations` | 查看历史 Recall@K / HitRate@K / MRR | admin |

### 7.1 `GET /api/v1/audit-logs` 返回示例

先用 `admin@example.com` 登录，并在 Swagger 的 **Authorize** 框中只填入返回的 token。这个接口用于追踪登录、文档入库、Agent 问答和审批操作；普通用户不能读取全量审计记录。

#### 管理员查询成功：有审计记录

请求：`GET /api/v1/audit-logs?limit=2`

状态码：`200 OK`

```json
[
  {
    "id": 8,
    "request_id": "6c6b9c6d-31fd-4c66-9d16-fc4a82920f13",
    "actor_id": 3,
    "event": "APPROVAL_DECIDED",
    "metadata": {
      "approval_id": 1,
      "decision": "APPROVED"
    },
    "created_at": "2026-08-17T05:50:20.123456"
  },
  {
    "id": 7,
    "request_id": "ad3535f5-0c71-4582-896c-26d85a841f26",
    "actor_id": 1,
    "event": "APPROVAL_CREATED",
    "metadata": {
      "approval_id": 1,
      "action": "EXPORT"
    },
    "created_at": "2026-08-17T05:49:57.123456"
  }
]
```

`limit` 默认是 `50`，最大实际返回 `200` 条；结果按 `created_at` 倒序排列。常见 `event` 包括：`LOGIN`、`DOCUMENT_INGESTED`、`AGENT_CHAT`、`APPROVAL_CREATED`、`APPROVAL_DECIDED`。

#### 管理员查询成功：没有记录

在一个尚未发生任何被审计操作的新数据库中，响应仍是 `200 OK`，但返回空数组：

```json
[]
```

#### 未登录或没有携带 Token

状态码：`403 Forbidden`

```json
{
  "detail": "Not authenticated"
}
```

这是 FastAPI 的 Bearer 认证层在路由进入业务代码前返回的结果。

#### Token 无效或已过期

状态码：`401 Unauthorized`

```json
{
  "detail": "无效或过期的访问令牌"
}
```

重新调用登录接口，复制新的 token；在 Swagger 的 Authorize 框中只粘贴 token 本体，不要附加 `Bearer`。

#### 已登录但不是 admin

例如以 `employee@example.com` 或 `finance@example.com` 调用，状态码为 `403 Forbidden`：

```json
{
  "detail": "权限不足"
}
```

这是接口的角色控制，不代表 Token 错误。只有 `role` 为 `admin` 的用户可以查看审计日志。

## 8. Docker + PostgreSQL 部署

```powershell
docker compose up --build
```

Docker Compose 会启动 `api` 和带 pgvector 扩展的 PostgreSQL `db`。首次启动时会启用 `vector` 扩展，并为当前 embedding 维度建立 HNSW 余弦索引；数据卷 `postgres_data` 会保留数据库。首次启动后依旧使用第 3 节的账号。生产环境应把 `JWT_SECRET`、数据库密码和 API Key 放进密钥管理服务，而不是 compose 文件。

## 9. 从作业升级为“企业级”的待办清单

按优先级逐项完成，并为每项附截图或测试证据：

1. 引入独立 Reranker，对现有“关键词 + 向量混合召回 + RRF”候选做重排序，并在同一评测集上记录相对纯向量检索的 Recall@K、MRR 变化。
2. 增加对象存储（MinIO/S3）与 PDF/DOCX 解析队列；对扫描件增加 OCR。
3. 用 Redis 加限流、会话缓存和审批异步任务；为工具调用设置超时和重试上限。
4. 接入 OIDC/企业 SSO，使用真实组织/部门同步，删除演示口令。
5. 用 OpenTelemetry + Prometheus/Grafana 输出延迟、错误率、token、工具调用和审批率。
6. 在 CI 中跑 pytest、ruff、迁移检查和 eval；对 prompt 与模型版本做版本化。
7. 加提示注入测试、敏感信息检测、文档删除权与数据保留策略。

## 10. 作品集录屏与答辩

录一段 3 分钟视频：

1. 以 employee 登录，询问机密报销额度，展示无法检索。
2. 以 finance 登录重复问题，展示带引用的正确回答。
3. 提交“导出客户名单”，展示系统创建审批单而非直接执行。
4. 以 admin 审批，并打开审计日志展示同一 `request_id` 的全过程。
5. 运行 `pytest -q`，展示质量门禁。

面试官常问：“为什么不是简单 RAG？”回答：简单 RAG 解决的是“找到文本”，这个项目还解决了企业落地必须面对的**身份、数据边界、危险动作、人审、审计与可验证性**。

## 11. 目录

```text
app/                 FastAPI 应用
  agent.py           工具调用与本地降级 Agent
  chunking.py        文档分块
  config.py          环境配置与 Embedding 参数
  db.py              SQLAlchemy 模型和演示数据
  embeddings.py      DashScope 在线 Embedding + 离线兜底
  evaluation.py      带标注的检索评测和指标持久化
  main.py            API、鉴权、审批、审计
  retrieval.py       向量/关键词混合检索 + RBAC 前置过滤 + 增量索引
  security.py        JWT 与密码哈希
  schemas.py         请求/响应数据结构
  vectorstore.py     pgvector / FAISS / NumPy 向量索引三后端
data/evals.jsonl     离线评测样本（10 条，4 条带 relevant_titles）
scripts/benchmark.py httpx 并发压测（P50/P95/P99/RPS）与 JSON 报告
scripts/run_eval.py  直接运行带标注评测并写入 JSON 报告
tests/               回归测试（7 个用例）
docker-compose.yml   API + pgvector/PostgreSQL
```

## 12. 边界说明

这是教学/作品集实现：演示账户、HMAC JWT、PBKDF2 密码哈希和 SQLite 向量后端只适合本地演示。它刻意不实现真实的“导出/发送/删除”副作用；生产环境应接入 SSO 和 Argon2id，并在审批后使用专用执行服务、幂等键、二次确认和告警。
