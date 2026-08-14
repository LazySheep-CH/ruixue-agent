"""持久化层:PostgreSQL(数据的 source of truth)。

    models.py       表结构(SQLAlchemy 声明式模型)
    engine.py       连接管理(连接池)
    repository.py   存/取
    migrations/     Alembic 版本化迁移 —— 表结构的每一次变更都是一个可回滚的版本

为什么独立成模块(不放 ingestion 里):表是 ingestion(写)和 rag(读)共用的。
"""

from ruixue_agent.persistence.engine import database_url, get_engine
from ruixue_agent.persistence.models import Base, ChunkRow, DocumentRow

__all__ = ["Base", "ChunkRow", "DocumentRow", "database_url", "get_engine"]
