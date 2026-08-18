"""用户自有知识库:上传文档 → 切块 → 向量化 → 按用户隔离检索。

平台知识库回答不了的领域(用户自己的规程、当地农技手册),让用户自己带资料。
能力边界 = 知识库边界,这里把边界的扩展权交给用户,而不是靠放宽提示词
让模型凭训练语料泛答(那是幻觉的来路)。

隔离纪律照抄 memory/datasets 那套,一条都不能少:
  - Milvus 侧 user_id 标量过滤 + PG 侧再验一次归属(任一层写错都不该泄露)
  - doc_id 用 uuid(不可枚举),删除按 user_id + doc_id 双条件
  - 解析在内存完成,原文只进 PG,不落文件系统(没有路径就没有路径逃逸)

解析用 PyMuPDF(镜像里本来就有,report.py 在用),不引新依赖;
MinerU 那条重管线是给平台语料的离线批处理用的,在线上传等不起它。
"""

from __future__ import annotations

import logging
import re
import uuid

from pymilvus import DataType
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.models import UserDocChunkRow, UserDocRow
from ruixue_agent.rag.embedding import embed
from ruixue_agent.rag.milvus_store import _DIM, _URI
from ruixue_agent.rag.retriever import Hit

logger = logging.getLogger("ruixue.userkb")

COLLECTION = "user_kb"

# 配额:知识库不是网盘。20 份 × 10MB 足够个人资料,挡得住把整个硬盘倒进来。
MAX_DOCS_PER_USER = 20
MAX_DOC_BYTES = 10 * 1024 * 1024
MAX_CHUNKS_PER_DOC = 600

# 切块:目标 500 字上下,段落边界优先,相邻块重叠 80 字防句子被切断。
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# 召回条数与相关度下限。用户库的块和平台库走同一个嵌入模型,量纲一致。
RECALL_TOP_K = 3
RECALL_MIN_SCORE = 0.35


class UserKbError(ValueError):
    """面向用户的错误消息(配额、格式、解析失败)。"""


# ── 解析与切块 ────────────────────────────────────────────────


def parse_text(filename: str, raw: bytes) -> str:
    """PDF 走 PyMuPDF,txt/md 按常见编码解。返回全文;认不出的格式报人话。"""
    name = filename.lower()
    if name.endswith(".pdf"):
        import fitz

        try:
            with fitz.open(stream=raw, filetype="pdf") as doc:
                text = "\n\n".join(page.get_text() for page in doc)
        except Exception as e:
            raise UserKbError("PDF 解析失败,文件可能已损坏或是扫描件(暂不支持 OCR)。") from e
        if not text.strip():
            raise UserKbError("这份 PDF 提取不到文字 —— 可能是纯扫描件,暂不支持 OCR。")
        return text
    if name.endswith((".txt", ".md")):
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        raise UserKbError("文本编码无法识别,请另存为 UTF-8 后重新上传。")
    raise UserKbError("暂只支持 PDF / TXT / Markdown 文档。")


def chunk_text(text: str) -> list[str]:
    """段落优先的滑窗切块。

    不复用平台语料那套父子分块:那套依赖 MinerU 的结构化输出(标题层级、
    表格标注),用户的裸文本没有这些。对个人资料,定长滑窗够用 ——
    检索质量的大头在嵌入模型,不在切块的花样。
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= CHUNK_SIZE:
            buf = f"{buf}\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
            buf = buf[-CHUNK_OVERLAP:] + "\n" + p if CHUNK_OVERLAP else p
        else:
            buf = p
        # 单段超长:硬切
        while len(buf) > CHUNK_SIZE:
            chunks.append(buf[:CHUNK_SIZE])
            buf = buf[CHUNK_SIZE - CHUNK_OVERLAP :]
    if buf.strip():
        chunks.append(buf)
    return chunks[:MAX_CHUNKS_PER_DOC]


# ── Milvus ────────────────────────────────────────────────────


def _client():
    from pymilvus import MilvusClient

    return MilvusClient(uri=_URI)


def ensure_collection() -> None:
    """建库并 load(幂等,搜索前兜底调用)。load 自愈的理由见 memory/store.py。"""
    c = _client()
    if not c.has_collection(COLLECTION):
        schema = c.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=36)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=_DIM)
        # user_id 标量过滤是安全边界,不是优化 —— 漏了会召回别人的资料
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        c.create_collection(COLLECTION, schema=schema)
        idx = c.prepare_index_params()
        idx.add_index("vector", index_type="FLAT", metric_type="COSINE")
        c.create_index(COLLECTION, idx)
        logger.info("已创建用户知识库 collection")
    c.load_collection(COLLECTION)


# ── 入库 / 检索 / 管理 ────────────────────────────────────────


def ingest(user_id: str, filename: str, raw: bytes) -> dict:
    """解析、切块、向量化、双写。返回 {doc_id, n_chunks}。"""
    if len(raw) > MAX_DOC_BYTES:
        raise UserKbError(f"文件超过 {MAX_DOC_BYTES // 1024 // 1024}MB 上限。")
    with Session(get_engine()) as s:
        n_docs = s.scalar(
            select(func.count()).select_from(UserDocRow).where(UserDocRow.user_id == user_id)
        )
    if (n_docs or 0) >= MAX_DOCS_PER_USER:
        raise UserKbError(f"最多保留 {MAX_DOCS_PER_USER} 份资料,请先删除不用的。")

    text = parse_text(filename, raw)
    chunks = chunk_text(text)
    if not chunks:
        raise UserKbError("文档内容为空。")

    doc_id = str(uuid.uuid4())
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    vectors = embed(chunks)

    with Session(get_engine()) as s:
        s.add(
            UserDocRow(
                doc_id=doc_id, user_id=user_id, filename=filename[:255], n_chunks=len(chunks)
            )
        )
        for cid, seq, body in zip(chunk_ids, range(len(chunks)), chunks, strict=True):
            s.add(UserDocChunkRow(chunk_id=cid, doc_id=doc_id, user_id=user_id, seq=seq, text=body))
        s.commit()

    ensure_collection()
    _client().upsert(
        COLLECTION,
        [
            {"chunk_id": cid, "vector": vec, "user_id": user_id}
            for cid, vec in zip(chunk_ids, vectors, strict=True)
        ],
    )
    logger.info("用户 %s 入库文档 %s(%d 块)", user_id, doc_id, len(chunks))
    return {"doc_id": doc_id, "filename": filename, "n_chunks": len(chunks)}


def search_as_hits(user_id: str, query: str, k: int = RECALL_TOP_K) -> list[Hit]:
    """检索该用户的资料,直接返回生成层可用的 Hit(title 标明是用户资料)。

    失败一律返回空:用户库是增益,不能因为它把主回答拖垮 —— 和记忆召回同一条。
    """
    try:
        ensure_collection()
        res = _client().search(
            COLLECTION,
            data=[embed([query])[0]],
            limit=k,
            filter=f'user_id == "{user_id}"',  # 安全边界
            output_fields=["chunk_id"],
        )
    except Exception:
        logger.warning("用户知识库检索失败,本次不合并", exc_info=True)
        return []

    scored = {
        h["entity"]["chunk_id"]: h["distance"]
        for h in (res[0] if res else [])
        if h["distance"] >= RECALL_MIN_SCORE
    }
    if not scored:
        return []

    with Session(get_engine()) as s:
        rows = s.scalars(
            select(UserDocChunkRow).where(
                UserDocChunkRow.chunk_id.in_(list(scored)),
                UserDocChunkRow.user_id == user_id,  # PG 侧再验一次归属
            )
        ).all()
        names = {
            d.doc_id: d.filename
            for d in s.scalars(
                select(UserDocRow).where(UserDocRow.doc_id.in_({r.doc_id for r in rows}))
            )
        }

    return [
        Hit(
            chunk_id=r.chunk_id,
            text=r.text,
            score=scored[r.chunk_id],
            document_id=r.doc_id,
            section_path=[],
            page_start=0,
            page_end=0,
            # 标题直接声明来源身份 —— 生成层拼上下文、答案标出处都用它,
            # 用户一眼能分清哪条结论出自自己的资料
            title=f"您上传的资料 · {names.get(r.doc_id, '文档')}",
            year=None,
        )
        for r in sorted(rows, key=lambda r: -scored[r.chunk_id])
    ]


def list_docs(user_id: str) -> list[UserDocRow]:
    with Session(get_engine()) as s:
        return list(
            s.scalars(
                select(UserDocRow)
                .where(UserDocRow.user_id == user_id)
                .order_by(UserDocRow.created_at.desc())
            )
        )


def delete_doc(user_id: str, doc_id: str) -> bool:
    """删文档及其块;不属于该用户返回 False(不透露存在性)。"""
    with Session(get_engine()) as s:
        chunk_ids = list(
            s.scalars(
                select(UserDocChunkRow.chunk_id).where(
                    UserDocChunkRow.doc_id == doc_id, UserDocChunkRow.user_id == user_id
                )
            )
        )
        if not chunk_ids:
            return False
        s.execute(
            delete(UserDocChunkRow).where(
                UserDocChunkRow.doc_id == doc_id, UserDocChunkRow.user_id == user_id
            )
        )
        res = s.execute(
            delete(UserDocRow).where(UserDocRow.doc_id == doc_id, UserDocRow.user_id == user_id)
        )
        s.commit()
    try:
        _client().delete(COLLECTION, filter=f'chunk_id in {chunk_ids!r} and user_id == "{user_id}"')
    except Exception:
        # 向量删失败只是残留索引项,PG 归属校验保证它们永远查不出正文
        logger.warning("用户库向量删除失败(不影响数据安全)", exc_info=True)
    return bool(res.rowcount)
