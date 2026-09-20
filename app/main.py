import uuid
from contextlib import asynccontextmanager
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
from .chunking import chunk_text
from .config import get_settings
from .db import ApprovalRequest, AuditLog, Chunk, Document, SessionLocal, User, get_db, initialize_database
from .evaluation import list_evaluation_runs, run_retrieval_evaluation
from .retrieval import RetrievalMetrics, index_stats
from .schemas import ApiKeyTestRequest, ApiKeyTestResponse, ApprovalDecision, ApprovalOut, ChatRequest, ChatResponse, DocumentCreate, LoginRequest, RetrievalEvaluationRequest, TokenResponse
from .security import create_access_token, decode_access_token, hash_password, verify_password

@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    with SessionLocal() as db:
        seed_demo_data(db)
    yield


app = FastAPI(title="AegisKB Enterprise AI Agent", version="1.0.0", lifespan=lifespan)
bearer = HTTPBearer()
Db = Annotated[Session, Depends(get_db)]
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def audit(db: Session, request_id: str, actor_id: int | None, event: str, **metadata) -> None:
    db.add(AuditLog(request_id=request_id, actor_id=actor_id, event=event, metadata_json=metadata))
    db.commit()


def current_user(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)], db: Db) -> User:
    user = db.get(User, decode_access_token(credentials.credentials))
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


def require_roles(*roles: str):
    def dependency(user: Annotated[User, Depends(current_user)]) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="权限不足")
        return user
    return dependency


def seed_demo_data(db: Session) -> None:
    if db.scalar(select(User.id).limit(1)):
        return
    users = [
        User(email="employee@example.com", password_hash=hash_password("demo-password"), role="employee", department="engineering", clearance="INTERNAL"),
        User(email="finance@example.com", password_hash=hash_password("demo-password"), role="finance", department="finance", clearance="CONFIDENTIAL"),
        User(email="admin@example.com", password_hash=hash_password("demo-password"), role="admin", department="security", clearance="CONFIDENTIAL"),
    ]
    db.add_all(users)
    db.flush()
    docs = [
        ("远程办公制度", "all", "INTERNAL", "员工每周可远程办公两天。跨城市长期办公需直属经理批准，并在 HR 系统登记。"),
        ("2026 差旅报销制度", "finance", "CONFIDENTIAL", "国内出差住宿标准：一线城市每晚不超过 800 元，其他城市每晚不超过 500 元。报销须在行程结束后 30 天内提交发票和审批单。"),
        ("数据分级与客户信息规范", "all", "INTERNAL", "客户联系人和合同属于内部数据。批量导出、跨系统发送、删除数据均为高风险操作，必须经过管理员审批并保留审计记录。"),
    ]
    for title, department, sensitivity, body in docs:
        document = Document(title=title, department=department, sensitivity=sensitivity, body=body, created_by=users[2].id)
        db.add(document)
        db.flush()
        db.add_all([Chunk(document_id=document.id, ordinal=i, content=chunk) for i, chunk in enumerate(chunk_text(body))])
    db.commit()


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


@app.post("/api/v1/documents", status_code=201)
def create_document(payload: DocumentCreate, db: Db, user: Annotated[User, Depends(require_roles("finance", "admin"))]) -> dict:
    document = Document(**payload.model_dump(), created_by=user.id)
    db.add(document)
    db.flush()
    chunks = chunk_text(payload.body)
    db.add_all([Chunk(document_id=document.id, ordinal=i, content=chunk) for i, chunk in enumerate(chunks)])
    db.commit()
    audit(db, str(uuid.uuid4()), user.id, "DOCUMENT_INGESTED", document_id=document.id, sensitivity=document.sensitivity)
    return {"id": document.id, "chunks": len(chunks), "message": "文档已入库，向量将在首次检索时建立"}


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request, db: Db, user: Annotated[User, Depends(current_user)]) -> ChatResponse:
    request_id = str(uuid.uuid4())
    action = risk_action(payload.message)
    if action:
        approval = ApprovalRequest(requester_id=user.id, action=action, payload={"message": payload.message})
        db.add(approval)
        db.commit()
        db.refresh(approval)
        audit(db, request_id, user.id, "APPROVAL_CREATED", approval_id=approval.id, action=action)
        return ChatResponse(answer=f"该请求属于 {action} 高风险操作，已创建审批单 #{approval.id}。批准前不会执行。", requires_approval=True, approval_id=approval.id, request_id=request_id, mode="approval")
    api_key = request.headers.get("X-DASHSCOPE-API-KEY") or None
    metrics = RetrievalMetrics()
    answer, citations, mode = run_agent(db, user, payload.message, api_key=api_key, retrieval_metrics=metrics)
    audit(db, request_id, user.id, "AGENT_CHAT", mode=mode, citation_count=len(citations), **metrics.as_dict())
    return ChatResponse(answer=answer, citations=citations, request_id=request_id, mode=mode)


@app.get("/api/v1/retrieval/stats")
def retrieval_stats(db: Db, _: Annotated[User, Depends(require_roles("admin"))]) -> dict:
    """Show the selected vector backend and index freshness without exposing text."""
    return index_stats(db)


@app.post("/api/v1/retrieval/evaluations", status_code=201)
def run_retrieval_eval(
    payload: RetrievalEvaluationRequest,
    db: Db,
    user: Annotated[User, Depends(require_roles("admin"))],
) -> dict:
    """Run the checked-in labeled dataset and persist Recall@K/HitRate@K/MRR."""
    try:
        result = run_retrieval_evaluation(db, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    audit(
        db,
        str(uuid.uuid4()),
        user.id,
        "RETRIEVAL_EVALUATION_RUN",
        evaluation_id=result["id"],
        top_k=result["top_k"],
        recall_at_k=result["recall_at_k"],
        hit_rate_at_k=result["hit_rate_at_k"],
        mrr=result["mrr"],
    )
    return result


@app.get("/api/v1/retrieval/evaluations")
def retrieval_evaluations(
    db: Db,
    _: Annotated[User, Depends(require_roles("admin"))],
    limit: int = 20,
) -> list[dict]:
    return list_evaluation_runs(db, limit=limit)


@app.post("/api/v1/settings/qwen/test", response_model=ApiKeyTestResponse)
def test_qwen_api_key(payload: ApiKeyTestRequest, user: Annotated[User, Depends(current_user)]) -> ApiKeyTestResponse:
    connected, message = test_qwen_connection(payload.api_key.strip())
    return ApiKeyTestResponse(connected=connected, message=message, model=get_settings().qwen_model)


@app.get("/api/v1/approvals", response_model=list[ApprovalOut])
def list_approvals(db: Db, user: Annotated[User, Depends(current_user)]) -> list[ApprovalRequest]:
    query = select(ApprovalRequest).order_by(ApprovalRequest.created_at.desc())
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
