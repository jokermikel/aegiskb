from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# 限定密级取值，避免客户端把密级写成任意字符串。
Sensitivity = Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL"]


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'


class UserOut(BaseModel):
    email: str
    role: str
    department: str
    clearance: str


class DocumentCreate(BaseModel):
    title: str = Field(min_length=2, max_length=255)
    body: str = Field(min_length=20, max_length=100_000)
    department: str = "all"
    sensitivity: Sensitivity = "INTERNAL"


class ChatRequest(BaseModel):
    # 限长阻止超大请求导致成本和延迟失控。
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
    payload: dict
    status: str
    requester_id: int
    requester_email: str
    created_at: datetime
    comment: str | None
