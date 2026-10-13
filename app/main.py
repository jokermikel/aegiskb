import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from .agent import risk_action, run_agent, test_qwen_connection
from .config import get_settings
from .db import ApprovalRequest, AuditLog, Base, Chunk, Document, User, engine, get_db
from .schemas import ApiKeyTestRequest, ApiKeyTestResponse, ApprovalOut, ApprovalDecision, ChatRequest, ChatResponse, DocumentCreate, LoginRequest, TokenResponse, UserOut
from .security import create_access_token, decode_access_token, hash_password, verify_password
from .seed_data import DEMO_DOCUMENTS, DEMO_PASSWORD, DEMO_USERS, build_approval_samples

@asynccontextmanager
async def lifespan(_: FastAPI):
    """服务启动时准备数据库：按配置建表、检查结构漂移、灌入演示数据。"""
    prepare_database()
    yield


app = FastAPI(title="Aegiskb Enterprise AI Agent", version="1.0.0", lifespan=lifespan)
bearer = HTTPBearer()  # OpenAPI/Swagger 由它生成 Bearer 认证框。
Db = Annotated[Session, Depends(get_db)]
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def audit(db: Session, request_id: str, actor_id: int | None, event: str, **metadata) -> None:
    """把安全相关行为写入可关联 request_id 的审计表。"""
    db.add(AuditLog(request_id=request_id, actor_id=actor_id, event=event, metadata_json=metadata))
    db.commit()


def make_chunks(text: str, size: int = 350, overlap: int = 60) -> list[str]:
    """文档分块并保留 60 字重叠，减少句子恰好在边界被截断的问题。"""
    clean = " ".join(text.split())
    return  [clean[i : i + size] for i in range(0, len(clean), size - overlap)]


def current_user(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)], db: Db) -> User:
    """所有需登录接口的统一身份入口。"""
    user = db.get(User, decode_access_token(credentials.credentials))
    if not user:
        raise HTTPException(status_code = 401, detail = "用户不存在")
    return user


def require_roles(*roles: str):
    """生成角色依赖，例如 require_roles('admin')。"""
    def dependency(user: Annotated[User, Depends(current_user)]) -> User:
        if user.role not in roles:
            raise HTTPException(status_code = 403, detail = "权限不足")
        return user
    return dependency


def seed_demo_data(db: Session) -> None:
    """仅空数据库时导入演示数据：10 个账号、50 篇制度文档、100 条审批单。

    数据本身维护在 app/seed_data.py，这里只负责落库，防止每次启动重复插入。
    """
    if db.scalar(select(User.id).limit(1)):
        return
    users = [
        User(
            email=email,
            password_hash=hash_password(DEMO_PASSWORD),
            role=role,
            department=department,
            clearance=clearance,
        )
        for email, role, department, clearance in DEMO_USERS
    ]
    db.add_all(users)
    db.flush()  # 先取到用户主键，供文档和审批单的外键使用。
    users_by_email = {user.email: user for user in users}
    admin = users_by_email["admin@example.com"]

    for title, department, sensitivity, body in DEMO_DOCUMENTS:
        document = Document(title=title, department=department, sensitivity=sensitivity, body=body, created_by=admin.id)
        db.add(document)
        db.flush()
        db.add_all([Chunk(document_id=document.id, ordinal=i, content=chunk) for i, chunk in enumerate(make_chunks(body))])

    # 高风险操作的审批单样例。已裁决的记录挂管理员作为审批人，并补上意见与裁决时间。
    for sample in build_approval_samples([user.email for user in users]):
        decided = sample["status"] != "PENDING"
        db.add(
            ApprovalRequest(
                requester_id=users_by_email[sample["requester_email"]].id,
                action=sample["action"],
                payload={"message": sample["message"]},
                status=sample["status"],
                reviewer_id=admin.id if decided else None,
                comment=sample["comment"],
                created_at=sample["created_at"],
                decided_at=sample["decided_at"],
            )
        )
    db.commit()


def detect_schema_drift() -> list[str]:
    """比对模型定义与数据库实际结构，返回差异描述（空列表表示一致）。"""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    problems: list[str] = []
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            problems.append(f"缺少表 {table_name}")
            continue
        actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
        for missing in sorted({column.name for column in table.columns} - actual_columns):
            problems.append(f"{table_name} 缺少字段 {missing}")
    return problems


def prepare_database() -> None:
    """准备数据库：按需建表、检查结构漂移、灌入演示数据。

    开发与演示环境允许自动建表（AUTO_CREATE_SCHEMA，默认开启）；
    建表后始终检查结构漂移 —— 若模型已改而库未迁移，直接抛错并提示
    `alembic upgrade head`，避免出现「改了模型却静默不生效」。
    """
    if get_settings().auto_create_schema:
        Base.metadata.create_all(bind=engine)
    problems = detect_schema_drift()
    if problems:
        raise RuntimeError(
            "数据库结构与模型定义不一致，请执行 `alembic upgrade head` 完成迁移，"
            "或删除数据库后重新初始化：\n  - " + "\n  - ".join(problems)
        )
    with Session(bind=engine) as db:
        seed_demo_data(db)


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Db, request: Request) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="邮箱或密码错误")
    audit(db, str(uuid.uuid4()), user.id, "LOGIN", ip=request.client.host if request.client else "unknown")
    return TokenResponse(access_token=create_access_token(user.id))


@app.get("/api/v1/auth/me", response_model=UserOut)
def authenticated_user(user: Annotated[User, Depends(current_user)]) -> UserOut:
    return UserOut(email=user.email, role=user.role, department=user.department, clearance=user.clearance)


@app.post("/api/v1/documents", status_code=201)
def create_document(payload: DocumentCreate, db: Db, user: Annotated[User, Depends(require_roles("finance", "admin"))]) -> dict:
    document = Document(**payload.model_dump(), created_by=user.id)
    db.add(document)
    db.flush()
    db.add_all([Chunk(document_id=document.id, ordinal=i, content=chunk) for i, chunk in enumerate(make_chunks(payload.body))])
    db.commit()
    audit(db, str(uuid.uuid4()), user.id, "DOCUMENT_INGESTED", document_id=document.id, sensitivity=document.sensitivity)
    return {"id": document.id, "chunks": len(document.chunks), "message": "文档已入库"}


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request, db: Db, user: Annotated[User, Depends(current_user)]) -> ChatResponse:
    request_id = str(uuid.uuid4())
    action = risk_action(payload.message)
    if action:
        # 人审优先：此处绝不调用模型，也绝不执行真实导出/删除/发送。
        approval = ApprovalRequest(requester_id=user.id, action=action, payload={"message": payload.message})
        db.add(approval)
        db.commit()
        db.refresh(approval)
        audit(db, request_id, user.id, "APPROVAL_CREATED", approval_id=approval.id, action=action)
        return ChatResponse(answer=f"该请求属于 {action} 高风险操作，已创建审批单 #{approval.id}。批准前不会执行。", requires_approval=True, approval_id=approval.id, request_id=request_id, mode="approval")
    api_key = request.headers.get("X-DASHSCOPE-API-KEY") or None
    answer, citations, mode = run_agent(db, user, payload.message, api_key=api_key)
    audit(db, request_id, user.id, "AGENT_CHAT", mode=mode, citation_count=len(citations))
    return ChatResponse(answer=answer, citations=citations, request_id=request_id, mode=mode)


@app.post("/api/v1/settings/qwen/test", response_model=ApiKeyTestResponse)
def test_qwen_api_key(payload: ApiKeyTestRequest, _: Annotated[User, Depends(current_user)]) -> ApiKeyTestResponse:
    connected, message = test_qwen_connection(payload.api_key.strip())
    return ApiKeyTestResponse(connected=connected, message=message, model=get_settings().qwen_model)


@app.get("/api/v1/approvals", response_model=list[ApprovalOut])
def list_approvals(
    db: Db,
    user: Annotated[User, Depends(current_user)],
    response: Response,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
) -> list[ApprovalRequest]:
    """分页返回审批单；总数与分页信息通过响应头回传。"""
    query = select(ApprovalRequest).order_by(ApprovalRequest.created_at.desc(), ApprovalRequest.id.desc())
    total_query = select(func.count(ApprovalRequest.id))
    pending_query = total_query.where(ApprovalRequest.status == "PENDING")
    # 普通用户只能看自己发起的审批；admin 才能看全量。
    if user.role != "admin":
        query = query.where(ApprovalRequest.requester_id == user.id)
        total_query = total_query.where(ApprovalRequest.requester_id == user.id)
        pending_query = pending_query.where(ApprovalRequest.requester_id == user.id)
    # 总数放在响应头，避免改变既有数组响应结构。
    response.headers["X-Total-Count"] = str(db.scalar(total_query) or 0)
    response.headers["X-Total-Pending"] = str(db.scalar(pending_query) or 0)
    response.headers["X-Page"] = str(page)
    response.headers["X-Page-Size"] = str(page_size)
    return list(db.scalars(query.offset((page - 1) * page_size).limit(page_size)))


@app.post("/api/v1/approvals/{approval_id}/decision", response_model=ApprovalOut)
def decide_approval(approval_id: int, payload: ApprovalDecision, db: Db, user: Annotated[User, Depends(require_roles("admin"))]) -> ApprovalRequest:
    approval = db.get(ApprovalRequest, approval_id)
    if not approval:
        raise HTTPException(status_code=404, detail="审批单不存在")
    if approval.status != "PENDING":
        raise HTTPException(status_code=409, detail="审批单已处理")
    approval.status, approval.comment, approval.reviewer_id = payload.decision, payload.comment, user.id
    approval.decided_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(approval)
    audit(db, str(uuid.uuid4()), user.id, "APPROVAL_DECIDED", approval_id=approval.id, decision=payload.decision)
    return approval


@app.get("/api/v1/audit-logs")
def audit_logs(db: Db, _: Annotated[User, Depends(require_roles("admin"))], limit: int = Query(50, ge=1, le=200)) -> list[dict]:
    # 用 Query(ge=1, le=200) 夹紧取值：负数会让 SQLite 的 LIMIT -1 变成不限行数，
    # 从而绕过 200 条上限，这里直接在参数校验层拒绝越界取值。
    rows = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)).all()
    return [{"id": row.id, "request_id": row.request_id, "actor_id": row.actor_id, "event": row.event, "metadata": row.metadata_json, "created_at": row.created_at} for row in rows]
