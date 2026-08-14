"""add memories table

长期记忆表:跨会话记住某个用户的事实。设计取舍见 models.py 的 MemoryRow 文档。

注意:注意:手工删掉了 autogenerate 生成的 checkpoint 表 DROP 语句 —— 这是第【三】次踩到。
    checkpoints / checkpoint_blobs / checkpoint_writes / checkpoint_migrations
    是 LangGraph 的 PostgresSaver 【自己建、自己管】的,不在我们的 SQLAlchemy
    模型里。alembic 比对时发现"库里有、模型里没有",就以为是要删的残留表。
    照它生成的跑一次 = 把所有用户的会话历史清空。

    **规矩:每次 autogenerate 之后必须人工过一遍,凡是 drop_table 都要先问
    "这张表是我们建的吗"。** 不是我们建的,一律删掉那几行。

Revision ID: 544280bdbc5c
Revises: df9bfcd2f774
Create Date: 2026-08-10 09:53:24.673617
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "544280bdbc5c"
down_revision: str | None = "df9bfcd2f774"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("memory_id", sa.String(length=16), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column(
            "kind",
            sa.String(length=24),
            nullable=False,
            comment="plot=地块 / crop=作物 / preference=偏好 / other",
        ),
        sa.Column("text", sa.Text(), nullable=False, comment="抽取出的事实,一句话"),
        sa.Column("source_run_id", sa.String(length=36), nullable=True),
        sa.Column("confidence", sa.String(length=8), server_default="high", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted", sa.Boolean(), server_default="false", nullable=False),
        sa.PrimaryKeyConstraint("id"),
        comment="长期记忆。一行 = 关于某用户的一条事实",
    )
    # 检索主路径:按用户 + 未删除。放在一个联合索引里,选择性高的 user_id 在前。
    op.create_index("idx_memories_user_alive", "memories", ["user_id", "deleted"], unique=False)
    op.create_index(op.f("ix_memories_memory_id"), "memories", ["memory_id"], unique=True)
    op.create_index(op.f("ix_memories_source_run_id"), "memories", ["source_run_id"], unique=False)
    op.create_index(op.f("ix_memories_user_id"), "memories", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_memories_user_id"), table_name="memories")
    op.drop_index(op.f("ix_memories_source_run_id"), table_name="memories")
    op.drop_index(op.f("ix_memories_memory_id"), table_name="memories")
    op.drop_index("idx_memories_user_alive", table_name="memories")
    op.drop_table("memories")
