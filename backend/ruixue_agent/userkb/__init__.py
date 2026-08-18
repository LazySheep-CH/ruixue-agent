"""用户自有知识库:上传 → 切块 → 向量化 → 隔离检索,并入 search_knowledge。"""

from ruixue_agent.userkb.store import (
    MAX_DOC_BYTES,
    MAX_DOCS_PER_USER,
    UserKbError,
    delete_doc,
    ingest,
    list_docs,
    search_as_hits,
)

__all__ = [
    "MAX_DOC_BYTES",
    "MAX_DOCS_PER_USER",
    "UserKbError",
    "delete_doc",
    "ingest",
    "list_docs",
    "search_as_hits",
]
