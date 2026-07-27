"""add users table

Revision ID: 3f069886db2e
Revises: 0003
Create Date: 2026-07-26

⚠️ 注意:autogenerate 原本还生成了 drop checkpoint* 四张表的语句 —— 那是 LangGraph
的会话记忆表,由 PostgresSaver.setup() 在运行时创建、不在我们的 SQLAlchemy 模型里,
Alembic 便误判成"多余的表"。**已手工删除那些 drop 语句**,否则升级会清空所有对话历史。

教训:autogenerate 只是草稿,必须人工审阅 —— 尤其当它生成 drop 的时候。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3f069886db2e"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_table("users")
