"""用户自有知识库的机制测试。

不测检索效果(那由嵌入模型决定,不是这层代码的责任),只测机制:
切块契约、解析报错的可读性、双层归属隔离、配额、来源标注。
向量层(embed / Milvus)全部打桩 —— 真嵌入要花钱走网络,而这里要验的
没有一条依赖向量算得准不准。
"""

from __future__ import annotations

import io

import pytest

from ruixue_agent.userkb import store as ukb
from ruixue_agent.userkb.store import UserKbError, chunk_text, parse_text

# ── 纯函数:切块与解析(无外部依赖)──────────────────────────


def test_chunks_respect_size_and_paragraphs():
    text = "\n\n".join(f"第{i}段。" + "内容" * 60 for i in range(6))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= ukb.CHUNK_SIZE for c in chunks)
    # 段落边界优先:每块开头应落在某段的行首,而不是句子中间随机切
    assert chunks[0].startswith("第0段")


def test_oversized_single_paragraph_is_hard_split():
    chunks = chunk_text("字" * (ukb.CHUNK_SIZE * 3))
    assert len(chunks) >= 3
    assert all(len(c) <= ukb.CHUNK_SIZE for c in chunks)


def test_chunk_count_is_capped(monkeypatch):
    monkeypatch.setattr(ukb, "MAX_CHUNKS_PER_DOC", 5)
    chunks = chunk_text("\n\n".join("段" * 400 for _ in range(50)))
    assert len(chunks) == 5


def test_parse_txt_handles_gbk():
    assert "地膜" in parse_text("笔记.txt", "地膜覆盖要点".encode("gbk"))


def test_parse_rejects_unknown_format():
    with pytest.raises(UserKbError, match="PDF / TXT / Markdown"):
        parse_text("photo.jpg", b"...")


def test_parse_rejects_broken_pdf():
    with pytest.raises(UserKbError, match="解析失败"):
        parse_text("bad.pdf", b"not a pdf at all")


# ── 存储与隔离(需要 PostgreSQL;向量层打桩)─────────────────


def _deps_up() -> bool:
    try:
        from sqlalchemy import text

        from ruixue_agent.persistence.engine import get_engine

        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


needs_pg = pytest.mark.skipif(not _deps_up(), reason="需要 PostgreSQL")

_A, _B = "ukb-alice", "ukb-bob"


class FakeMilvus:
    """记录 upsert、按预设返回 search —— 只验证调用契约,不验证向量数学。"""

    def __init__(self):
        self.upserted: list[dict] = []
        self.search_result: list[list[dict]] = [[]]

    def has_collection(self, name):
        return True

    def load_collection(self, name):
        pass

    def upsert(self, name, rows):
        self.upserted.extend(rows)

    def search(self, name, **kw):
        return self.search_result

    def delete(self, name, filter=""):
        pass


@pytest.fixture
def fake_vec(monkeypatch):
    fm = FakeMilvus()
    monkeypatch.setattr(ukb, "_client", lambda: fm)
    monkeypatch.setattr(ukb, "embed", lambda texts: [[0.1] * 4 for _ in texts])
    yield fm
    # 清理两位测试用户的数据,不影响真实库
    from sqlalchemy import delete as sqldelete
    from sqlalchemy.orm import Session

    from ruixue_agent.persistence.engine import get_engine
    from ruixue_agent.persistence.models import UserDocChunkRow, UserDocRow

    with Session(get_engine()) as s:
        s.execute(sqldelete(UserDocChunkRow).where(UserDocChunkRow.user_id.in_([_A, _B])))
        s.execute(sqldelete(UserDocRow).where(UserDocRow.user_id.in_([_A, _B])))
        s.commit()


@needs_pg
def test_ingest_writes_chunks_and_vectors_with_owner(fake_vec):
    info = ukb.ingest(_A, "规程.txt", ("大蒜覆膜规程。" * 30).encode("utf-8"))
    assert info["n_chunks"] >= 1
    assert len(fake_vec.upserted) == info["n_chunks"]
    # 每条向量都必须带 user_id —— 这是 Milvus 侧过滤的前提
    assert all(r["user_id"] == _A for r in fake_vec.upserted)


@needs_pg
def test_search_rechecks_ownership_in_pg(fake_vec):
    """Milvus 结果被污染(返回了别人的 chunk_id)时,PG 归属复核必须拦住。

    这是双层隔离里第二层的存在意义:任何一层出 bug(过滤条件写错、
    collection 被误写),另一层仍保证查不出别人的正文。
    """
    ukb.ingest(_A, "甲的资料.txt", ("甲的机密内容。" * 30).encode("utf-8"))
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from ruixue_agent.persistence.engine import get_engine
    from ruixue_agent.persistence.models import UserDocChunkRow

    with Session(get_engine()) as s:
        a_chunk = s.scalar(select(UserDocChunkRow.chunk_id).where(UserDocChunkRow.user_id == _A))
    # 模拟 Milvus 侧过滤失效:把甲的块当成乙的检索结果返回
    fake_vec.search_result = [[{"entity": {"chunk_id": a_chunk}, "distance": 0.9}]]
    assert ukb.search_as_hits(_B, "机密") == []


@needs_pg
def test_hits_are_labeled_as_user_material(fake_vec):
    ukb.ingest(_A, "本地手册.md", ("覆膜时间建议。" * 30).encode("utf-8"))
    cid = fake_vec.upserted[0]["chunk_id"]
    fake_vec.search_result = [[{"entity": {"chunk_id": cid}, "distance": 0.9}]]
    hits = ukb.search_as_hits(_A, "覆膜时间")
    assert hits and hits[0].title == "您上传的资料 · 本地手册.md"


@needs_pg
def test_low_score_hits_are_dropped(fake_vec):
    ukb.ingest(_A, "a.txt", ("正文。" * 30).encode("utf-8"))
    cid = fake_vec.upserted[0]["chunk_id"]
    fake_vec.search_result = [[{"entity": {"chunk_id": cid}, "distance": 0.05}]]
    assert ukb.search_as_hits(_A, "无关问题") == []


@needs_pg
def test_delete_requires_ownership(fake_vec):
    doc_id = ukb.ingest(_A, "a.txt", ("正文。" * 30).encode("utf-8"))["doc_id"]
    assert ukb.delete_doc(_B, doc_id) is False  # 乙删不掉甲的
    assert ukb.delete_doc(_A, doc_id) is True
    assert ukb.list_docs(_A) == []


@needs_pg
def test_doc_quota(fake_vec, monkeypatch):
    monkeypatch.setattr(ukb, "MAX_DOCS_PER_USER", 1)
    ukb.ingest(_A, "a.txt", ("正文。" * 30).encode("utf-8"))
    with pytest.raises(UserKbError, match="最多保留"):
        ukb.ingest(_A, "b.txt", ("正文。" * 30).encode("utf-8"))


# ── 接口层(/kb/docs)────────────────────────────────────────


@pytest.fixture
def api(monkeypatch, fake_vec):
    from fastapi.testclient import TestClient

    import ruixue_app.auth as auth
    import ruixue_app.main as main_mod

    monkeypatch.setattr(auth, "API_KEYS", {_A: "alice-ukb", _B: "bob-ukb"})
    return TestClient(main_mod.app)


def _post(api, key, name="资料.txt", body="覆膜规程。" * 30):
    return api.post(
        "/kb/docs",
        files={"file": (name, io.BytesIO(body.encode("utf-8")), "text/plain")},
        headers={"X-API-Key": key},
    )


@needs_pg
def test_api_upload_list_delete_roundtrip(api):
    doc_id = _post(api, _A).json()["doc_id"]
    names = [d["filename"] for d in api.get("/kb/docs", headers={"X-API-Key": _A}).json()["docs"]]
    assert "资料.txt" in names
    assert api.delete(f"/kb/docs/{doc_id}", headers={"X-API-Key": _B}).status_code == 404
    assert api.delete(f"/kb/docs/{doc_id}", headers={"X-API-Key": _A}).status_code == 200


@needs_pg
def test_api_unparseable_file_is_422_with_readable_detail(api):
    r = _post(api, _A, name="照片.jpg")
    assert r.status_code == 422
    assert "PDF / TXT / Markdown" in r.json()["detail"]
