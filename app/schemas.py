from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Sensitivity = Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL"]


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DocumentCreate(BaseModel):
    title: str = Field(min_length=2, max_length=255)
    body: str = Field(min_length=20, max_length=100_000)
    department: str = "all"
    sensitivity: Sensitivity = "INTERNAL"


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=4_000)


class ApiKeyTestRequest(BaseModel):
    api_key: str = Field(min_length=8, max_length=500)


class ApiKeyTestResponse(BaseModel):
    connected: bool
    message: str
    model: str


class Citation(BaseModel):
    document_id: int
    title: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    requires_approval: bool = False
    approval_id: int | None = None
    request_id: str
    mode: Literal["qwen", "local", "approval"]


class ApprovalDecision(BaseModel):
    decision: Literal["APPROVED", "REJECTED"]
    comment: str = Field(min_length=2, max_length=1_000)


class ApprovalOut(BaseModel):
    id: int
    action: str
    status: str
    requester_id: int
    created_at: datetime
    comment: str | None


class RetrievalEvaluationRequest(BaseModel):
    top_k: int = Field(default=4, ge=1, le=20)
