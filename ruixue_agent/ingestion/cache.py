"""管道中间产物的 JSON 缓存,避免重复执行昂贵的解析步骤。

与 ruixue_agent/persistence/ 的职责区分(改名自 persistence.py,避免混淆):
    cache.py      ingestion 内部的临时落盘,可随时删除、重跑管道即可再生;
    persistence/  PostgreSQL,数据的 source of truth,ingestion 与 rag 共用。

目录布局:
  data/parsed/<document_id>.json   Document(parse+clean 结果)
  data/chunks/<document_id>.json   Chunk 列表(chunk 结果)
  data/failed/<document_id>.json   质量门禁未通过的文档(留档,不静默丢弃)

序列化直接用 Pydantic:model_dump_json 落盘,model_validate_json 读回,
嵌套的 Element 自动重建并再次校验。
"""

from __future__ import annotations

from pathlib import Path

from ruixue_agent.ingestion.schema import Chunk, Document

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
PARSED = _DATA / "parsed"
CHUNKS = _DATA / "chunks"
FAILED = _DATA / "failed"


def save_document(doc: Document, folder: Path = PARSED) -> Path:
    """Document 存为 <folder>/<document_id>.json,返回文件路径。"""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{doc.document_id}.json"
    path.write_text(doc.model_dump_json(), encoding="utf-8")
    return path


def load_document(document_id: str, folder: Path = PARSED) -> Document | None:
    """读回 Document;不存在返回 None,由调用方决定是否重新解析。"""
    path = folder / f"{document_id}.json"
    if not path.exists():
        return None
    return Document.model_validate_json(path.read_text(encoding="utf-8"))


def save_chunks(document_id: str, chunks: list[Chunk], folder: Path = CHUNKS) -> Path:
    """一篇文档的全部 Chunk 存为一个 JSON 数组。"""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{document_id}.json"
    body = "[" + ",".join(c.model_dump_json() for c in chunks) + "]"
    path.write_text(body, encoding="utf-8")
    return path


def load_chunks(document_id: str, folder: Path = CHUNKS) -> list[Chunk] | None:
    """读回一篇文档的全部 Chunk;不存在返回 None。"""
    import json

    path = folder / f"{document_id}.json"
    if not path.exists():
        return None
    return [
        Chunk.model_validate(o) for o in json.loads(path.read_text(encoding="utf-8"))
    ]


def is_parsed(document_id: str) -> bool:
    """该文档是否已解析,批处理断点续跑时用于跳过。"""
    return (PARSED / f"{document_id}.json").exists()
