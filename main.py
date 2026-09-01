import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agent import risk_action, run_agent, test_qwen_connection
from .config import get_settings
from .db import ApprovalRequest, AuditLog, Base, Chunk, Document, User, engine, get_db
from .schemas import ApiKeyTestRequest, ApiKeyTestResponse, ApprovalOut, ApprovalDecision, ChatRequest, ChatResponse, DocumentCreate, LoginRequest, TokenResponse, UserOut
from .security import create_access_token, decode_access_token, hash_password, verify_password

app = FastAPI(title="Aegiskb Enterprise AI Agent", version="1.0.0")
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
    """仅空数据库时创建三位演示用户和三篇演示文档，防止每次启动重复插入。"""
    if db.scalar(select(User.id).limit(1)):
        return
    users = [
        User(email="employee@example.com", password_hash=hash_password("demo-password"), role="employee", department="engineering", clearance="INTERNAL"),
        User(email="finance@example.com", password_hash=hash_password("demo-password"), role="finance", department="finance", clearance="CONFIDENTIAL"),
        User(email="admin@example.com", password_hash=hash_password("demo-password"), role="admin", department="security", clearance="CONFIDENTIAL"),
    ]
    db.add_all(users)
    db.flush()  # 获取 users[2].id，供文档 created_by 外键使用。
    docs = [
        ("远程办公制度", "all", "INTERNAL", "员工每周可远程办公两天。跨城市长期办公需直属经理批准，并在 HR 系统登记。"),
        ("2026 差旅报销制度", "finance", "CONFIDENTIAL", "国内出差住宿标准：一线城市每晚不超过 800 元，其他城市每晚不超过 500 元。报销须在行程结束后 30 天内提交发票和审批单。"),
        ("数据分级与客户信息规范", "all", "INTERNAL", "客户联系人和合同属于内部数据。批量导出、跨系统发送、删除数据均为高风险操作，必须经过管理员审批并保留审计记录。"),
    ]
    for title, department, sensitivity, body in docs:
        document = Document(title=title, department=department, sensitivity=sensitivity, body=body, created_by=users[2].id)
        db.add(document)
        db.flush()
        db.add_all([Chunk(document_id=document.id, ordinal=i, content=chunk) for i, chunk in enumerate(make_chunks(body))])
    db.commit()


@app.on_event("startup")
def startup() -> None:
    # 教学原型直接建表；生产环境应使用 Alembic 数据库迁移。
    Base.metadata.create_all(bind=engine)
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
def list_approvals(db: Db, user: Annotated[User, Depends(current_user)]) -> list[ApprovalRequest]:
    query = select(ApprovalRequest).order_by(ApprovalRequest.created_at.desc())
    # 普通用户只能看自己发起的审批；admin 才能看全量。
    if user.role != "admin":
        query = query.where(ApprovalRequest.requester_id == user.id)
    return list(db.scalars(query))


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
def audit_logs(db: Db, _: Annotated[User, Depends(require_roles("admin"))], limit: int = 50) -> list[dict]:
    rows = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(min(limit, 200))).all()
    return [{"id": row.id, "request_id": row.request_id, "actor_id": row.actor_id, "event": row.event, "metadata": row.metadata_json, "created_at": row.created_at} for row in rows]
