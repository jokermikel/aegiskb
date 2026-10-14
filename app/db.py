from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.types import TypeDecorator

from .config import get_settings


def utcnow() -> datetime:
    # 所有审计时间统一存 UTC，展示层再转换时区。
    return datetime.now(UTC)


class Base(DeclarativeBase):
    # SQLAlchemy 所有表模型的共同基类。
    pass


class EmbeddingStorage(TypeDecorator):
    """向量列：SQLite 存 float32 二进制，PostgreSQL 存 vector 字面量。

    PostgreSQL 的物理列由 Alembic 迁移建成原生 ``vector``；ORM 侧保持文本类型，
    这样即使数据库用户没有权限启用 pgvector，也能安全退回 FAISS/NumPy 后端。
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "sqlite":
            return dialect.type_descriptor(LargeBinary())
        return dialect.type_descriptor(Text())


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32))
    department: Mapped[str] = mapped_column(String(64))
    clearance: Mapped[str] = mapped_column(String(32), default="INTERNAL")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    body: Mapped[str] = mapped_column(Text)
    department: Mapped[str] = mapped_column(String(64), default="all")
    sensitivity: Mapped[str] = mapped_column(String(32), default="INTERNAL")
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # 删除文档时联动删除 chunk，避免遗留可检索的敏感文件
    chunks: Mapped[list["Chunk"]] = relationship(cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # 删除文档时联动删除向量缓存，避免残留可被检索的孤立向量。
    embedding: Mapped["ChunkEmbedding | None"] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, uselist=False
    )

    @property
    def embedding_chunk_id(self) -> int:
        return self.id


class ChunkEmbedding(Base):
    """单个 chunk 的向量缓存。

    主键刻意与 ``chunks.id`` 一致：更换 embedding 提供方或维度时原地更新，
    而 ``content_hash`` 保证内容未变的 chunk 不会被重复 embedding。
    """

    __tablename__ = "chunk_embeddings"
    __table_args__ = (Index("ix_chunk_embeddings_provider_dim", "provider", "dim"),)

    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    dim: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    embedding: Mapped[Any] = mapped_column(EmbeddingStorage(), nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class RetrievalEvaluationRun(Base):
    """一次可复现的离线检索评测结果。"""

    __tablename__ = "retrieval_evaluation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), index=True)
    top_k: Mapped[int] = mapped_column(Integer)
    total_cases: Mapped[int] = mapped_column(Integer)
    evaluated_cases: Mapped[int] = mapped_column(Integer)
    recall_at_k: Mapped[float] = mapped_column(Float)
    hit_rate_at_k: Mapped[float] = mapped_column(Float)
    mrr: Mapped[float] = mapped_column(Float)
    backend: Mapped[str] = mapped_column(String(32))
    embedder: Mapped[str] = mapped_column(String(64))
    dim: Mapped[int] = mapped_column(Integer)
    results: Mapped[list[dict]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"
    id: Mapped[int] = mapped_column(primary_key=True)
    requester_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(64))
    # JSON 保留用户原始意图；生产环境还应加脱敏和保留期策略。
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requester: Mapped["User"] = relationship(foreign_keys=[requester_id])

    @property
    def requester_email(self) -> str:
        return self.requester.email


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    actor_id: Mapped[int | None] = mapped_column(nullable=True, index=True)
    event: Mapped[str] = mapped_column(String(64), index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


settings = get_settings()
engine = create_engine(
    settings.database_url,
    # SQLite 的请求线程需要这项；PostgreSQL 不需要
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI 依赖：一请求一会话，完成后保证关闭连接。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
