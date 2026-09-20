from datetime import datetime, timezone

from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, LargeBinary, String, Text, create_engine, text
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class EmbeddingStorage(TypeDecorator):
    """Store vectors as bytes in SQLite and as vector literals in PostgreSQL.

    PostgreSQL's physical column is created as native ``vector`` by
    :func:`initialize_database`; keeping the ORM type textual makes it possible
    to retain a safe FAISS/NumPy fallback when the database user cannot enable
    pgvector. SQLite uses a compact float32 blob.
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
    chunks: Mapped[list["Chunk"]] = relationship(cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)


class ChunkEmbedding(Base):
    """Cached embedding for one chunk.

    The primary key deliberately matches ``chunks.id``. Changing an embedding
    provider or dimension updates the row in place, while ``content_hash``
    prevents unchanged chunks from being embedded again.
    """

    __tablename__ = "chunk_embeddings"
    __table_args__ = (Index("ix_chunk_embeddings_provider_dim", "provider", "dim"),)

    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    dim: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[Any] = mapped_column(EmbeddingStorage(), nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class RetrievalEvaluationRun(Base):
    """One reproducible offline retrieval evaluation result."""

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
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def initialize_database() -> None:
    """Create application tables and, when possible, a native pgvector table.

    ``create_all`` is retained for the demo project's zero-config SQLite setup.
    PostgreSQL is initialized first so a deployment with the pgvector extension
    gets a real ``vector`` column. If the extension is unavailable, the table
    is created with text storage and retrieval safely falls back to FAISS/NumPy.
    """

    if engine.dialect.name != "postgresql":
        Base.metadata.create_all(bind=engine)
        return

    # The vector table has to be held back until the extension exists, while
    # its foreign-key target (chunks) must exist before native DDL runs.
    tables_without_embeddings = [
        table for table in Base.metadata.sorted_tables if table.name != ChunkEmbedding.__tablename__
    ]
    Base.metadata.create_all(bind=engine, tables=tables_without_embeddings)

    if engine.dialect.name == "postgresql":
        has_pgvector = False
        try:
            with engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            has_pgvector = True
        except Exception:
            # The retrieval backend will report the fallback. Do not make the
            # entire API unavailable merely because this database role cannot
            # create extensions.
            pass

        if has_pgvector:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS chunk_embeddings ("
                        "chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE, "
                        "provider VARCHAR(64) NOT NULL, dim INTEGER NOT NULL, "
                        "content_hash VARCHAR(64) NOT NULL, embedding vector NOT NULL, "
                        "meta JSONB NOT NULL DEFAULT '{}'::jsonb, "
                        "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                    )
                )
                connection.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_provider_dim "
                        "ON chunk_embeddings (provider, dim)"
                    )
                )
                # A partial expression index keeps a dimension change from
                # invalidating old rows. The query uses the same expression;
                # if the installed pgvector version lacks HNSW, retrieval
                # still works and simply falls back to a sequential scan.
                try:
                    dimension = int(settings.embedding_dim)
                    with connection.begin_nested():
                        connection.execute(
                            text(
                                f"CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_hnsw_{dimension} "
                                "ON chunk_embeddings USING hnsw "
                                f"((embedding::vector({dimension})) vector_cosine_ops) WHERE dim = {dimension}"
                            )
                        )
                except Exception:
                    pass

    # If pgvector was unavailable this creates the generic text-backed table;
    # if it was available the native table already exists and SQLAlchemy leaves
    # it untouched.
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
