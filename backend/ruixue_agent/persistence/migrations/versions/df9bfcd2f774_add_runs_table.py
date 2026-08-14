"""add runs table

Revision ID: df9bfcd2f774
Revises: 3f069886db2e
Create Date: 2026-08-06

⚠️ 注意:autogenerate 原本还生成了 drop checkpoint* 四张表的语句 —— 那是 LangGraph
的会话记忆表,由 PostgresSaver.setup() 在运行时创建、不在我们的 SQLAlchemy 模型里,
Alembic 便误判成"多余的表"。**已手工删除那些 drop 语句**,否则升级会清空所有对话历史。

这是第二次踩同一个坑(users 表那次也是)。教训:autogenerate 只是草稿,
凡是生成了 drop 的,一律先人工审阅再执行。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "df9bfcd2f774"
down_revision: str | None = "3f069886db2e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("thread_id", sa.String(length=160), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="running", nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("error", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_runs_created_at"), "runs", ["created_at"], unique=False)
    op.create_index(op.f("ix_runs_run_id"), "runs", ["run_id"], unique=True)
    op.create_index(op.f("ix_runs_user_id"), "runs", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_runs_user_id"), table_name="runs")
    op.drop_index(op.f("ix_runs_run_id"), table_name="runs")
    op.drop_index(op.f("ix_runs_created_at"), table_name="runs")
    op.drop_table("runs")
