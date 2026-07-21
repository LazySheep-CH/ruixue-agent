"""PgRepository 的规格说明书。

这些测试就是 repository.py 的【需求】—— 先读懂每个测试要什么,再去实现。
测试全绿 = 功能做完了。这就是 TDD:测试是规格,不是事后补的作业。

每个测试跑在事务里,测完自动回滚(见 conftest.py),所以可以放心往库里插。
"""

from __future__ import annotations

import pytest

from ruixue_agent.ingestion.schema import Chunk, Document, Element
from ruixue_agent.persistence.models import ChunkRow, DocumentRow
from ruixue_agent.persistence.repository import PgRepository


def _doc(document_id: str = "aaaa000000000001", **meta) -> Document:
    return Document(
        document_id=document_id,
        filename="李帅_PBAT地膜.pdf",
        file_type="pdf",
        sha256=document_id * 4,  # 16位 × 4 = 64位,凑个合法长度
        source="期刊论文",
        parser="mineru",
        elements=[Element(type="heading", text="1 实验部分", page=0, level=2)],
        meta={
            "title": "PBAT复合薄膜的降解性能",
            "year": 2025,
            "keywords": ["PBAT", "地膜", "降解"],
            "abstract": "本文研究了…",
            **meta,
        },
    )


def _chunks(document_id: str = "aaaa000000000001") -> list[Chunk]:
    """一父两子 —— 真实结构的最小样本。"""
    return [
        Chunk(
            chunk_id=f"{document_id}_s0",
            document_id=document_id,
            text="1.1 主要原料。PBAT 牌号 TH801T,由蓝山屯河提供。密度 1.26 g/cm³。",
            section_path=["实验部分", "1.1 主要原料"],
            page_start=1,
            page_end=1,
            parent_id=None,
            meta={"kind": "parent"},
        ),
        Chunk(
            chunk_id=f"{document_id}_s0_c0",
            document_id=document_id,
            text="PBAT 牌号 TH801T,由蓝山屯河提供。",
            section_path=["实验部分", "1.1 主要原料"],
            page_start=1,
            page_end=1,
            parent_id=f"{document_id}_s0",
            meta={"kind": "child", "element_type": "paragraph"},
        ),
        Chunk(
            chunk_id=f"{document_id}_s0_c1",
            document_id=document_id,
            text="密度 1.26 g/cm³。",
            section_path=["实验部分", "1.1 主要原料"],
            page_start=1,
            page_end=1,
            parent_id=f"{document_id}_s0",
            meta={"kind": "child", "element_type": "paragraph"},
        ),
    ]


@pytest.fixture
def repo(session) -> PgRepository:
    return PgRepository(session)


# ────────────────────── 1. 文档:存得进,读得出 ──────────────────────


def test_save_document_maps_meta_to_columns(repo, session):
    """Document.meta 里的字段要落到【真正的列】上,不是塞进一个 JSON 大字段。

    为什么:列才能建索引、才能 WHERE year > 2020、才有类型检查。
    塞 JSON = 把数据库当文件用,那还不如不用数据库。
    """
    repo.save_document(_doc())
    row = session.get(DocumentRow, "aaaa000000000001")
    assert row is not None
    assert row.title == "PBAT复合薄膜的降解性能"
    assert row.year == 2025
    assert row.keywords == ["PBAT", "地膜", "降解"]  # PG 原生数组,不是逗号拼的字符串
    assert row.parser == "mineru"  # 数据血缘


def test_save_document_tolerates_missing_meta(repo, session):
    """元数据抽不全是【常态】,不是异常。

    实测覆盖率:title 97.6% / year 84.9% / doi 34.4% —— 意味着
    238 篇没有 year。没有就是 NULL,不能因此拒绝入库、更不能崩。
    """
    doc = Document(
        document_id="bbbb000000000002",
        filename="无元数据.pdf",
        file_type="pdf",
        sha256="b" * 64,
        source="标准规范",
        parser="mineru",
        meta={},  # 什么都没抽到
    )
    repo.save_document(doc)
    row = session.get(DocumentRow, "bbbb000000000002")
    assert row.title is None and row.year is None and row.keywords is None


# ────────────────────── 2. 幂等:重跑不炸、不重复 ──────────────────────


def _count(session, model, **where) -> int:
    """只数【我们关心的那些行】,不数全表。

    为什么不用 session.query(X).count():那是在数全表,等于假设"库里只有我的测试数据"。
    库里一旦有别的东西(比如真实数据、别的测试留下的),断言就红 ——
    而那根本不是被测代码的错。测试要盯住自己那一亩三分地。
    (踩过:这个文件第一版就是这么写的,库里灌了 1578 篇真数据后集体翻车)
    """
    return session.query(model).filter_by(**where).count()


def test_save_document_twice_is_idempotent(repo, session):
    """★ 核心需求:同一篇存两次 = 库里还是 1 行,且不报错。

    为什么这条最重要:灌 1578 篇的过程中挂了(网断/OOM/手滑 Ctrl-C),
    你必须能【直接重跑】。如果重跑会因为主键冲突炸掉,你就只能
    "先全删再重来" —— 那 26 万条就没有断点续传,每次失败都从零开始。

    这就是 document_id = 内容寻址(sha256前16位)真正兑现价值的地方:
    同样的文件永远同样的 ID → 天然幂等。
    """
    repo.save_document(_doc())
    repo.save_document(_doc())  # 再来一次,不许炸
    assert _count(session, DocumentRow, document_id="aaaa000000000001") == 1


def test_save_document_twice_updates_content(repo, session):
    """幂等 ≠ 忽略第二次。第二次的内容要【覆盖】掉第一次。

    场景:metadata 抽取算法改进了,重跑管道 —— 新抽出来的 title 得生效,
    而不是"已经有了就跳过"(那样你的改进永远进不了库)。
    """
    repo.save_document(_doc(title="旧标题"))
    repo.save_document(_doc(title="新标题"))
    row = session.get(DocumentRow, "aaaa000000000001")
    assert row.title == "新标题"
    assert _count(session, DocumentRow, document_id="aaaa000000000001") == 1


# ────────────────────── 3. 块:父子关系 ──────────────────────


def test_save_chunks_writes_parent_and_children(repo, session):
    repo.save_document(_doc())
    repo.save_chunks(_chunks())

    assert _count(session, ChunkRow, document_id="aaaa000000000001") == 3
    parent = session.get(ChunkRow, "aaaa000000000001_s0")
    child = session.get(ChunkRow, "aaaa000000000001_s0_c0")
    assert parent.kind == "parent" and parent.parent_id is None
    assert child.kind == "child" and child.parent_id == parent.chunk_id
    assert child.element_type == "paragraph"
    assert child.section_path == ["实验部分", "1.1 主要原料"]


def test_save_chunks_handles_child_before_parent(repo, session):
    """★ 核心需求:哪怕子块【排在父块前面】传进来,也要能存进去。

    为什么会炸:chunks 表有自引用外键 parent_id → chunks.chunk_id。
    先插子块时,它爹还不存在 → PG 直接拒绝(违反外键)。

    chunk_document() 现在恰好是父在前,但 repository 不该【依赖调用方的顺序】——
    那是隐形契约,哪天有人改了 chunk 的实现,这里就神秘炸掉。
    自己保证顺序,是这个方法的职责。
    """
    reversed_chunks = list(reversed(_chunks()))  # 子块跑到最前面了
    repo.save_document(_doc())
    repo.save_chunks(reversed_chunks)  # 不许炸
    assert _count(session, ChunkRow, document_id="aaaa000000000001") == 3


def test_save_chunks_twice_is_idempotent(repo, session):
    """同上:26 万条的灌库过程必须可重跑。"""
    repo.save_document(_doc())
    repo.save_chunks(_chunks())
    repo.save_chunks(_chunks())
    assert _count(session, ChunkRow, document_id="aaaa000000000001") == 3


def test_save_empty_chunks_is_noop(repo, session):
    """空列表不许炸 —— 有的文档质量差,切完一个块都不剩。

    边界情况:很多"批量插入"的写法(比如手拼 SQL 的 VALUES)
    遇到空列表会生成 `INSERT INTO chunks VALUES` 这种语法错误的 SQL。
    """
    repo.save_document(_doc())
    repo.save_chunks([])  # 不许炸
    assert _count(session, ChunkRow, document_id="aaaa000000000001") == 0


# ────────────────────── 4. 触发器要真的生效 ──────────────────────


def test_tsv_is_populated_by_trigger(repo, session):
    """text_tsv 由数据库触发器从 text_tokens 生成 —— repository 不许自己填 tsv。

    ⚠ 分工在 migration 0002 变了:
        0001: 触发器读 text      —— 但 simple 配置不切中文,整句一个 token,索引是废的
        0002: 触发器读 text_tokens —— Python 侧 jieba 分好词写进来,PG 只负责转 tsvector
      分词是 Python 的活(能加领域词典、能测),转 tsvector 是数据库的活。

    所以 repository 存的 chunk 如果没有 text_tokens,tsv 就是空的 —— 这是对的:
    分词由 scripts/backfill_tokens.py 统一做,不该混进写库路径。
    """
    from sqlalchemy import text as sa_text

    repo.save_document(_doc())
    repo.save_chunks(_chunks())
    session.flush()

    # 模拟分词管道写入 tokens(真实路径是 backfill_tokens.py)
    session.execute(
        sa_text("UPDATE chunks SET text_tokens = :tok WHERE chunk_id = :cid"),
        {"tok": "pbat 牌号 th801t 蓝山 屯河", "cid": "aaaa000000000001_s0_c0"},
    )
    row = session.get(ChunkRow, "aaaa000000000001_s0_c0")
    session.refresh(row)
    assert row.text_tsv is not None
    assert "th801t" in row.text_tsv.lower()  # 触发器把 tokens 转成了 tsvector


# ────────────────────── 5. 检索侧要用的读接口 ──────────────────────


def test_get_chunks_returns_in_requested_order(repo, session):
    """Milvus 返回一串 chunk_id,我们拿它去 PG 取文本 —— 顺序必须是【相似度顺序】。

    坑:SQL 的 WHERE chunk_id IN (...) 【不保证】按你给的顺序返回,
    数据库爱怎么返回怎么返回。直接用 = 相似度排名被打乱,
    而且这个 bug 不报错,只是答案悄悄变差(最难查的那种)。
    """
    repo.save_document(_doc())
    repo.save_chunks(_chunks())
    session.flush()

    want = ["aaaa000000000001_s0_c1", "aaaa000000000001_s0_c0"]  # 故意倒着要
    got = repo.get_chunks(want)
    assert [c.chunk_id for c in got] == want


def test_get_parents_for_small_to_big(repo, session):
    """Small-to-Big:子块命中 → 取它爹的完整文本给 LLM。"""
    repo.save_document(_doc())
    repo.save_chunks(_chunks())
    session.flush()

    parents = repo.get_parents(["aaaa000000000001_s0_c0"])
    assert len(parents) == 1
    assert parents[0].chunk_id == "aaaa000000000001_s0"
    assert "密度 1.26" in parents[0].text  # 父块含子块没有的上下文


def test_get_chunks_missing_ids_are_skipped(repo, session):
    """要的 ID 不存在时:跳过,不是崩。

    真实场景:Milvus 和 PG 短暂不同步(PG 里删了文档,Milvus 索引还没重建)。
    索引指向了不存在的数据 —— 这是【正常的最终一致】,不是错误。
    """
    repo.save_document(_doc())
    repo.save_chunks(_chunks())
    session.flush()
    got = repo.get_chunks(["aaaa000000000001_s0", "根本不存在的id"])
    assert len(got) == 1
