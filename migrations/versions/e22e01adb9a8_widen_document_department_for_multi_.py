"""加宽 documents.department 以容纳多部门选择器

Revision ID: e22e01adb9a8
Revises: adb7c842a572
Create Date: 2026-10-14 11:50:28.967319

文档部门支持逗号分隔的多部门与子树通配（如 ``engineering,it``、``engineering/*``），
8 个已知部门全部用子树写法时长度为 66 字符，已超过原先的 VARCHAR(64)。
SQLite 不校验长度所以不会报错，但 PostgreSQL 会直接拒绝写入，因此必须加宽。

注：SQLite 不支持直接 ALTER COLUMN，这里使用 batch 模式（重建表方式）完成类型变更。
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e22e01adb9a8"
down_revision: Union[str, None] = "adb7c842a572"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.alter_column(
            "department",
            existing_type=sa.VARCHAR(length=64),
            type_=sa.String(length=255),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.alter_column(
            "department",
            existing_type=sa.String(length=255),
            type_=sa.VARCHAR(length=64),
            existing_nullable=False,
        )
