"""Repository:IR 对象(Document / Chunk)与数据库行之间的唯一翻译层。

持久化细节收口在此:上层(管道、检索)只调用 save/get,不感知底层是
PostgreSQL 还是别的存储。SQL 不散落在业务代码中,改列名只动这一个文件。

职责边界:仅负责 PG。向量归 MilvusVectorStore,两侧靠 chunk_id 关联。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ruixue_agent.ingestion.schema import Chunk, Document
from ruixue_agent.persistence.models import ChunkRow, DocumentRow

# 单次 execute 的批大小。过小则网络往返多;过大则受 PG 单语句 65535 个
# 占位符上限约束(每行 9 列 × 5000 行 = 45000,留有余量)。
_BATCH = 5000


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
        rows = self.session.scalars(
            select(ChunkRow).where(ChunkRow.chunk_id.in_(chunk_ids))
        ).all()

        by_id = {r.chunk_id: r for r in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_parents(self, child_ids: list[str]) -> list[ChunkRow]:
        """给定子块 ID,返回其父块,去重且保序。

        同一父块常被多个子块命中(同节多句相关),须去重,否则同段文本
        会重复送入生成层。去重用 dict.fromkeys 而非 set:保留首次出现的
        顺序,最相关子块的父块排在最前。
        """
        children = self.get_chunks(child_ids)
        parent_ids = list(
            dict.fromkeys(c.parent_id for c in children if c.parent_id is not None)
        )
        return self.get_chunks(parent_ids)
