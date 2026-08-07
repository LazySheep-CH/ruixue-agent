"""Repository:IR 对象(Document / Chunk)与数据库行之间的唯一翻译层。

持久化细节收口在此:上层(管道、检索)只调用 save/get,不感知底层是
PostgreSQL 还是别的存储。SQL 不散落在业务代码中,改列名只动这一个文件。

职责边界:仅负责 PG。向量归 MilvusVectorStore,两侧靠 chunk_id 关联。
"""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ruixue_agent.ingestion.schema import Chunk, Document
from ruixue_agent.persistence.models import ChunkRow, DocumentRow

# 单次 execute 的批大小。过小则网络往返多;过大则受 PG 单语句 65535 个
# 占位符上限约束(每行 9 列 × 5000 行 = 45000,留有余量)。
_BATCH = 5000


# 抽取失败时会落进 title 的"假标题":封面页眉、学校名、文档类型。
# 这些词出现在引用里既没用又占 token,还会让用户以为系统认不出文献。
_GENERIC_TITLE = re.compile(r"^(硕士|博士|学位论文|.*(大学|学院|研究院|学位论文|专业学位论文)$)")
# 文件名尾部的副本编号:"..._1.pdf" / "...(2).pdf"
_COPY_SUFFIX = re.compile(r"[_\-\s]*[(（]?\d+[)）]?$")


def display_title(title: str | None, filename: str | None) -> str:
    """给这篇文档挑一个【人看得懂的名字】。

    ## 为什么不能直接用 title

    实测 1578 篇里有 16% 的 title 是抽取失败的产物:"硕 士 学 位 论 文"、
    "西北农林科技大学"、空字符串 —— 抓到的是封面页眉,不是标题。
    而这些文档的【文件名】恰恰就是真标题:

        title = "硕 士 学 位 论 文"
        filename = "地膜残留对旱地玉米农田土壤理化性状及产量形成的影响.pdf"

    所以规则是:title 看起来像真标题就用它,否则退回文件名。
    宁可用文件名 —— 它至少是人起的名字,而假标题是纯噪声。
    """
    t = re.sub(r"\s+", "", title or "")
    if len(t) >= 12 and not _GENERIC_TITLE.match(t):
        return t
    stem = Path(filename).stem if filename else ""
    stem = _COPY_SUFFIX.sub("", stem).strip()
    return stem or t or ""


class PgRepository:
    """Document / Chunk 的存取。

    Session 由外部注入,事务边界归调用方:批量入库可将全部写入包成一个
    事务,测试可在事务内执行、结束后回滚。若在方法内部各自开事务,
    "存文档 + 存块"就无法构成原子操作。
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # ── 写 ──────────────────────────────────────────────

    def save_document(self, doc: Document) -> None:
        """写入文档元数据,幂等 upsert:重复写入不报错,内容以新值覆盖。

        用 PG 方言的 INSERT ... ON CONFLICT 而非"先查再插":后者是两条语句,
        并发下两个进程可能同时判定"不存在"而双插冲突;ON CONFLICT 是单条
        原子语句,无竞态窗口。幂等的锚是内容寻址的 document_id(sha256 前
        16 位)—— 同一文件永远得到同一主键。
        """
        m = doc.meta

        # 领域对象 → 行:meta 中的字段展开为列。元数据抽取不全是常态
        # (约 15% 文档无 year),用 .get() 取 None 落为 NULL,不拒绝入库。
        values = dict(
            document_id=doc.document_id,
            sha256=doc.sha256,
            filename=doc.filename,
            file_type=doc.file_type,
            source=doc.source,
            parser=doc.parser,
            title=m.get("title"),
            year=m.get("year"),
            doi=m.get("doi"),
            keywords=m.get("keywords"),
            abstract=m.get("abstract"),
        )
        # values 不含 created_at / updated_at:前者由 server_default 在首次
        # 插入时填充,后者由触发器维护,应用层不应写这两列。

        stmt = insert(DocumentRow).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["document_id"],
            # excluded 指"本次试图插入的那一行"(PG 语义,等价 MySQL 的 VALUES(col))
            set_={
                col: getattr(stmt.excluded, col)
                for col in values
                if col != "document_id"  # 主键是冲突判据,不参与更新
            },
        )

        # 不 commit:事务边界归 session 的持有方
        self.session.execute(stmt)

    def save_chunks(self, chunks: list[Chunk]) -> int:
        """批量写入块,幂等 upsert,返回写入条数。

        chunks 表有自引用外键 parent_id → chunks.chunk_id,子块插入时其
        父块必须已存在,因此父块与子块分两次 execute、父块在前。
        不能依赖"排好序放进同一批"—— 单次 execute 内 PG 不保证逐条按序
        处理,顺序保证只存在于两次 execute 之间。
        """
        if not chunks:
            return 0  # execute(stmt, []) 会报错,空列表直接返回

        def to_row(c: Chunk) -> dict:
            return dict(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                parent_id=c.parent_id,
                kind=c.meta.get("kind"),
                text=c.text,
                section_path=c.section_path,
                page_start=c.page_start,
                page_end=c.page_end,
                element_type=c.meta.get("element_type"),  # 父块无此键 → None
            )
            # 不含 text_tsv:由触发器从 text_tokens 生成,应用层不写

        # 批量形态的 upsert:不带 .values(),各行的值由 execute 的第二个参数
        # 按批传入,语句本身只是模板
        stmt = insert(ChunkRow)
        stmt = stmt.on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={
                col: getattr(stmt.excluded, col)
                for col in (
                    "document_id",
                    "parent_id",
                    "kind",
                    "text",  # text 更新后触发器自动重算 text_tsv
                    "section_path",
                    "page_start",
                    "page_end",
                    "element_type",
                )
            },
        )

        parents = [to_row(c) for c in chunks if c.parent_id is None]
        children = [to_row(c) for c in chunks if c.parent_id is not None]

        # 先父后子,各自分批
        for group in (parents, children):
            for i in range(0, len(group), _BATCH):
                self.session.execute(stmt, group[i : i + _BATCH])

        return len(chunks)

    # ── 读(检索侧)──────────────────────────────────

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRow]:
        """按给定顺序取块。

        SQL 的 IN 不保证返回顺序(按物理存储顺序扫描),直接使用会打乱
        检索侧传入的相似度排名,且不报错、只是结果变差。因此一次查回后
        在应用层按入参顺序重排;查不到的 ID 跳过 —— 向量索引与 PG 短暂
        不同步属正常的最终一致,不视为错误。
        """
        if not chunk_ids:
            return []

        # 单次查询取回全部,避免 N+1
        rows = self.session.scalars(select(ChunkRow).where(ChunkRow.chunk_id.in_(chunk_ids))).all()

        by_id = {r.chunk_id: r for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_documents_meta(
        self, document_ids: list[str]
    ) -> dict[str, tuple[str | None, int | None]]:
        """取文档的 (标题, 年份),给检索结果标出处用。

        ## 为什么值得多查这一次

        知识库里有 1990~2026 年的资料。片段被检索出来时如果不带年份,模型看到的
        就是一段无时间的事实 —— 于是文档里 2023 年的价格会被当成"现在的价格"
        答给用户,而用户会拿这个数去采购。实测评测里就出现过。

        靠提示词写"不要把历史价当现价"是**概率性**的;把年份摆到模型眼前是
        **结构性**的 —— 它没法看不见。能用结构解决的,不要留给模型自觉。

        一次 IN 查询,不是 N+1。
        """
        if not document_ids:
            return {}
        rows = self.session.execute(
            select(
                DocumentRow.document_id,
                DocumentRow.title,
                DocumentRow.year,
                DocumentRow.filename,
            ).where(DocumentRow.document_id.in_(set(document_ids)))
        ).all()
        return {r[0]: (display_title(r[1], r[3]), r[2]) for r in rows}

    def get_parents(self, child_ids: list[str]) -> list[ChunkRow]:
        """给定子块 ID,返回其父块,去重且保序。

        同一父块常被多个子块命中(同节多句相关),须去重,否则同段文本
        会重复送入生成层。去重用 dict.fromkeys 而非 set:保留首次出现的
        顺序,最相关子块的父块排在最前。
        """
        children = self.get_chunks(child_ids)
        parent_ids = list(dict.fromkeys(c.parent_id for c in children if c.parent_id is not None))
        return self.get_chunks(parent_ids)
