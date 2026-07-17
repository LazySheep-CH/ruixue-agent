"""管道的【中间产物缓存】:把 Document / Chunk 存成 JSON,下次直接读,不用重跑贵的步骤。

⚠ 和 ruixue_agent/persistence/ 的区别(别搞混,所以从 persistence.py 改名成了 cache.py):
    cache.py(这里)  ingestion 内部的临时落盘。可以随时删,删了重跑管道就有。
                      属于 ingestion,因为只有解析管道用它。
    persistence/      数据库(PostgreSQL)。是 source of truth,删了就没了。
                      独立模块,因为 ingestion(写)和 rag(读)【共用】。


分层缓存:
  data/parsed/<document_id>.json   Document(parse+clean 的结果)
  data/chunks/<document_id>.json   Chunk 列表(chunk 的结果)
  data/failed/<document_id>.json   质量门禁毙掉的(留账,不悄悄丢)

靠 schema.py 选 Pydantic 时就备好的两把刀:
  doc.model_dump_json()            对象 → JSON 字符串
  Document.model_validate_json(s)  JSON 字符串 → 对象(嵌套 Element 自动重建+校验)
"""

from __future__ import annotations

from pathlib import Path

from ruixue_agent.ingestion.schema import Chunk, Document

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
PARSED = _DATA / "parsed"
CHUNKS = _DATA / "chunks"
FAILED = _DATA / "failed"


def save_document(doc: Document, folder: Path = PARSED) -> Path:
    """把一个 Document 存成 data/parsed/<document_id>.json,返回文件路径。

    轮到你写(三步):
      1. folder.mkdir(parents=True, exist_ok=True)      确保目录在
      2. path = folder / f"{doc.document_id}.json"       文件名 = 它的 id
      3. path.write_text(doc.model_dump_json(), encoding="utf-8")
         return path
    """
    # 你写
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{doc.document_id}.json"
    path.write_text(doc.model_dump_json(), encoding="utf-8")
    return path


def load_document(document_id: str, folder: Path = PARSED) -> Document | None:
    """读回一个 Document;不存在返回 None(让调用方决定要不要重新解析)。

    轮到你写(三步):
      1. path = folder / f"{document_id}.json"
      2. 如果 not path.exists(): return None
      3. return Document.model_validate_json(path.read_text(encoding="utf-8"))
    """
    # 你写
    path = folder / f"{document_id}.json"
    if not path.exists():
        return None
    return Document.model_validate_json(path.read_text(encoding="utf-8"))


# ── 下面两个我写(和上面同构,只是 Chunk 是个列表,要多一层包装)──


def save_chunks(document_id: str, chunks: list[Chunk], folder: Path = CHUNKS) -> Path:
    """一篇文档的所有 Chunk 存成一个 JSON 数组。"""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{document_id}.json"
    body = "[" + ",".join(c.model_dump_json() for c in chunks) + "]"
    path.write_text(body, encoding="utf-8")
    return path


def load_chunks(document_id: str, folder: Path = CHUNKS) -> list[Chunk] | None:
    """读回一篇文档的所有 Chunk;不存在返回 None。"""
    import json

    path = folder / f"{document_id}.json"
    if not path.exists():
        return None
    return [
        Chunk.model_validate(o) for o in json.loads(path.read_text(encoding="utf-8"))
    ]


def is_parsed(document_id: str) -> bool:
    """这篇解析过了吗 —— 断点续跑靠它跳过已完成的。"""
    return (PARSED / f"{document_id}.json").exists()
