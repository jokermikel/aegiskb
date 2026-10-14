"""新增向量缓存表与检索评测记录表

Revision ID: adb7c842a572
Revises: 0001_initial_schema
Create Date: 2026-10-14 10:21:11.016198

- ``chunk_embeddings``：chunk 的向量缓存，主键与 ``chunks.id`` 一致；
  ``embedding`` 列由 ``app.db.EmbeddingStorage`` 定义 —— SQLite 落 float32 二进制，
  PostgreSQL 落 pgvector 可读的文本字面量。
- ``retrieval_evaluation_runs``：一次可复现的离线检索评测结果。

注意：本文件用到了应用侧的自定义列类型，因此必须导入 app.db；
autogenerate 只生成 ``app.db.EmbeddingStorage()`` 而不会补这行 import。
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

import app.db

revision: str = "adb7c842a572"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "retrieval_evaluation_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("total_cases", sa.Integer(), nullable=False),
        sa.Column("evaluated_cases", sa.Integer(), nullable=False),
        sa.Column("recall_at_k", sa.Float(), nullable=False),
        sa.Column("hit_rate_at_k", sa.Float(), nullable=False),
        sa.Column("mrr", sa.Float(), nullable=False),
        sa.Column("backend", sa.String(length=32), nullable=False),
        sa.Column("embedder", sa.String(length=64), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("results", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("retrieval_evaluation_runs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_retrieval_evaluation_runs_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_retrieval_evaluation_runs_dataset_hash"), ["dataset_hash"], unique=False)

    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", app.db.EmbeddingStorage(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    with op.batch_alter_table("chunk_embeddings", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_chunk_embeddings_content_hash"), ["content_hash"], unique=False)
        batch_op.create_index(batch_op.f("ix_chunk_embeddings_provider"), ["provider"], unique=False)
        batch_op.create_index("ix_chunk_embeddings_provider_dim", ["provider", "dim"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("chunk_embeddings", schema=None) as batch_op:
        batch_op.drop_index("ix_chunk_embeddings_provider_dim")
        batch_op.drop_index(batch_op.f("ix_chunk_embeddings_provider"))
        batch_op.drop_index(batch_op.f("ix_chunk_embeddings_content_hash"))
    op.drop_table("chunk_embeddings")

    with op.batch_alter_table("retrieval_evaluation_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_retrieval_evaluation_runs_dataset_hash"))
        batch_op.drop_index(batch_op.f("ix_retrieval_evaluation_runs_created_at"))
    op.drop_table("retrieval_evaluation_runs")
