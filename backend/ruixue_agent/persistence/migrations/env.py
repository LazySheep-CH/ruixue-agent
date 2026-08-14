"""Alembic 运行环境:告诉 Alembic「连哪个库」和「模型长什么样」。

Alembic 干的事:
    比对【models.py 里的模型】和【数据库里的实际表】,把差异生成一个 migration 文件。
    每个 migration 有 revision(自己的版本号)和 down_revision(上一个版本号),
    串成一条链。数据库里有张 alembic_version 表记着「我现在在哪个版本」。

    这就是为什么它能做到手写 schema.sql 做不到的三件事:
      1) 升级:alembic upgrade head —— 从当前版本一路跑到最新
      2) 回滚:alembic downgrade -1 —— 每个 migration 都有 downgrade()
      3) 追溯:git log 里能看到表结构的完整演化史
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from ruixue_agent.persistence.engine import database_url
from ruixue_agent.persistence.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 运行时才注入真实 URL(带密码),所以 alembic.ini 里可以留空 → 密码不进 git
config.set_main_option("sqlalchemy.url", database_url())

# Alembic 靠这个知道「模型应该长什么样」。models.py import 进来时,
# 每个 class 会自动把自己注册到 Base.metadata 里。
target_metadata = Base.metadata


# ── 不归我们管的表:autogenerate 必须无视它们 ─────────────────────
#
# ## 不加这个会出什么事(2026-08-13 实测,差点酿成事故)
#
# LangGraph 的 PostgresSaver 会在【同一个库】里自己建四张 checkpoint 表
# (见 ruixue_agent/checkpointer.py 的 saver.setup())。它们不在 Base.metadata 里,
# 于是 autogenerate 认为"库里有、模型里没有 → 应该删掉",生成的迁移里赫然写着:
#
#     op.drop_table('checkpoints')
#     op.drop_table('checkpoint_writes')
#     op.drop_table('checkpoint_blobs')
#     op.drop_table('checkpoint_migrations')
#
# 跑下去就是清空所有会话的执行状态 —— 而且迁移本身会"成功",
# 没有任何报错,只有用户发现"我的对话怎么接不上了"。
#
# 这类风险的共性:多个组件共用一个库,而只有一个组件持有"完整视图"。
# 解法不是"记得每次审一遍迁移"(迟早会忘),而是在生成期就把它们排除。
_FOREIGN_TABLE_PREFIXES = ("checkpoint",)


def include_object(obj, name, type_, reflected, compare_to):
    """只让 autogenerate 看见【我们自己的】表。

    reflected=True 表示"这个对象是从数据库里反射出来的";配合前缀判断,
    就能把别的组件建的表排除在比对之外 —— 既不会被误删,也不会被误建。
    """
    if type_ == "table" and name and name.startswith(_FOREIGN_TABLE_PREFIXES):
        return False
    return True


def run_migrations_offline() -> None:
    """离线模式:不连库,只把 SQL 打印出来(给 DBA 审核用 —— 大厂常见流程)。"""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式:真连库执行。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # 迁移是一次性的,不需要连接池
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )
        # PG 的 DDL 是【事务性】的(MySQL 不是!):
        # 一个 migration 里改 5 张表,中间失败 → 全部回滚,不会留下半吊子状态。
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
