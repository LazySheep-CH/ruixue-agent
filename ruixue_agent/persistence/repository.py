"""Repository:IR 对象(Document/Chunk)⇄ 数据库行。

【Repository 模式】—— 评审八股,也是这个文件存在的理由:
    把"怎么存"关在一个盒子里。上层(管道、检索)只说"帮我存这篇文档",
    不知道底下是 PostgreSQL 还是别的。
    好处:换存储只改这一个文件;上层代码 0 改动;测试能塞假的进去。
    对照反面:SQL 语句散落在管道各处 → 改个列名要全项目搜。

职责边界(想清楚再写):
    这里【只管 PG】。向量归 Milvus,由 MilvusVectorStore 管。
    两边靠 chunk_id 这个共同的 key 对上。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ruixue_agent.ingestion.schema import Chunk, Document
from ruixue_agent.persistence.models import ChunkRow, DocumentRow

# 一次 execute 塞多少条。
# 太小 → 网络往返次数多;太大 → 单条 SQL 撑爆内存 / 超过 PG 参数上限(65535 个占位符)。
# 我们一行 9 列 → 5000 行 × 9 = 45000 个占位符,安全线内。
_BATCH = 5000


class PgRepository:
    """存取 Document / Chunk。

    为什么 Session 是【传进来】的,不是自己 new 一个:
        叫【依赖注入】。谁调用谁决定事务边界 ——
        管道想"1578 篇一个事务",测试想"测完回滚",各自说了算。
        如果这里自己 new,每个方法各管各的事务,上层就没法把
        "存文档 + 存块" 变成一个原子操作(存了文档、存块时挂了 = 半吊子数据)。
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # ── 写 ──────────────────────────────────────────────

    def save_document(self, doc: Document) -> None:
        """存一篇文档的元数据。同一篇存两次 = 库里 1 行,且内容被更新(幂等 upsert)。

        ┌─ 要点 1:为什么不能用 session.merge() 或 "先 SELECT 再决定 INSERT/UPDATE" ─┐
        │  "先查再插" 有【竞态】:两个进程同时查(都说不存在)→ 都插 → 主键冲突炸。
        │  PG 的 INSERT ... ON CONFLICT 是【一条语句、一次原子操作】,没有中间窗口。
        │  (这就是常说的 upsert = UPDATE or INSERT)
        └────────────────────────────────────────────────────────────────┘

        ┌─ 要点 2:写法长这样 ─┐
        │   stmt = insert(DocumentRow).values(**列的字典)
        │   stmt = stmt.on_conflict_do_update(
        │       index_elements=["document_id"],    # 撞哪个键算冲突
        │       set_={要更新的列: stmt.excluded.那个列},   # excluded = "本来想插的那行"
        │   )
        │   self.session.execute(stmt)
        │
        │   注意 insert 是从 sqlalchemy.dialects.postgresql 导的(上面已经 import 好),
        │   不是通用的 sqlalchemy.insert —— ON CONFLICT 是 PG 方言,通用 insert 没有。
        └──────────────────────┘

        ┌─ 要点 3:哪些列该更新,哪些不该 ─┐
        │   想一想:document_id / sha256 / created_at 要不要放进 set_?
        │   (提示:document_id 是主键本身;created_at 是"第一次啥时候进来的")
        └────────────────────────────────┘

        映射关系(Document.meta 是个 dict,数据库要的是列):
            doc.document_id/filename/file_type/sha256/source/parser  → 同名列
            doc.meta["title"/"year"/"doi"/"keywords"/"abstract"]     → 同名列
            抽不到的字段:meta 里根本没这个 key → 用 .get() 拿到 None → 存 NULL
            (实测 238 篇没 year。这是常态,不是异常)
        """
        m = doc.meta

        # 第一步:翻译。领域对象(meta 是 dict)→ 数据库的形状(一列一个 key)。
        # 这就是 Repository 存在的意义 —— 这段"拧"的动作只在这里发生一次,
        # 上层永远不需要知道 meta["year"] 最后变成了哪一列。
        # 用 .get() 不用 []:抽不到的字段是常态(238 篇没 year),拿到 None → 存 NULL。
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
        # 注意 values 里【没有】created_at 和 updated_at:
        #   created_at → 第一次插入时 server_default=now() 自动填
        #   updated_at → 触发器负责,我们碰它就是抢它的活

        # 第二步:构造 upsert。等价于 MyBatis 的
        #     INSERT ... ON DUPLICATE KEY UPDATE title=VALUES(title), ...
        stmt = insert(DocumentRow).values(**values)
        stmt = stmt.on_conflict_do_update(
            # 撞哪个键算"冲突" —— 这里是主键 document_id
            index_elements=["document_id"],
            # 冲突时,哪些列用新值覆盖。
            # stmt.excluded = "本来想插进去的那一行" —— 就是 MySQL 的 VALUES(x)。
            # 名字很怪:它来自 PG 内部把这行"排除(excluded)"在插入之外,转而拿它来更新。
            set_={
                col: getattr(stmt.excluded, col)
                for col in values
                if col != "document_id"  # 主键是冲突判据本身,不更新
            },
        )

        # 第三步:执行。注意【没有 commit】—— 事务边界由调用方(session 的主人)决定。
        # 这就是为什么 Session 是传进来的:管道可以把 1578 篇包成一个事务,
        # 测试可以测完直接回滚。
        self.session.execute(stmt)

    def save_chunks(self, chunks: list[Chunk]) -> int:
        """批量存块,返回存了几个。同样要幂等。

        ┌─ 要点 1:顺序 —— 父块必须先插 ─┐
        │  chunks 表有【自引用外键】parent_id → chunks.chunk_id。
        │  先插子块时它爹还不存在 → PG 拒绝(违反外键约束)。
        │  所以进来的 list 得先排:父块在前,子块在后。
        │  (怎么判断是父块?parent_id is None)
        └────────────────────────────────┘

        ┌─ 要点 2:批量 —— 别一条一条插 ─┐
        │  26 万条,每条一次 execute = 26 万次网络往返。
        │  一次 execute 塞一批(几千条)是数量级的差别。
        │  SQLAlchemy 的写法:session.execute(stmt, [字典1, 字典2, ...])
        │  ——【传一个 list of dict 进去】,它会自动合并成批量 INSERT。
        │
        │  ⚠ 但父/子要分两批执行(要点 1),不能混在一个 list 里 —— 想想为什么。
        └────────────────────────────────┘

        ┌─ 要点 3:空列表 ─┐
        │  execute(stmt, []) 会炸。空的直接 return 0。
        └──────────────────┘

        ┌─ 要点 4:【不要】自己填 text_tsv ─┐
        │  那是数据库触发器的活(见 0001_baseline.py)。
        │  你只管 text,tsv 它自己算。
        └──────────────────────────────────┘

        映射:
            c.chunk_id / document_id / parent_id / text / section_path
            / page_start / page_end                        → 同名列
            c.meta["kind"]          → kind          ("parent" / "child")
            c.meta["element_type"]  → element_type  (父块没有这个 key → None)
        """
        if not chunks:
            return 0  # 空列表直接走人:execute(stmt, []) 会炸

        # ① 翻译:Chunk → 一行的字典。和 save_document 里那段是同一回事。
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
                element_type=c.meta.get("element_type"),  # 父块没这个 key → None
            )
            # 注意没有 text_tsv:那是触发器的活,我们碰它就是抢活干

        # ② 批量版的 upsert 语句。
        #    和 save_document 的区别:这里【没有 .values()】——
        #    单条时 .values(**d) 把值焊死在语句里;批量时值由 execute 的第二个参数
        #    一批批喂进来,语句本身只是个模板。
        stmt = insert(ChunkRow)
        stmt = stmt.on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={
                col: getattr(stmt.excluded, col)
                for col in (
                    "document_id",
                    "parent_id",
                    "kind",
                    "text",  # text 变了 → 触发器自动重算 text_tsv
                    "section_path",
                    "page_start",
                    "page_end",
                    "element_type",
                )
            },
        )

        # ③ 拆父子。这是这个方法的【核心】:
        #    chunks 表有自引用外键 parent_id → chunks.chunk_id。
        #    子块插入的那一刻,它爹必须已经在库里,否则 PG 拒绝。
        parents = [to_row(c) for c in chunks if c.parent_id is None]
        children = [to_row(c) for c in chunks if c.parent_id is not None]

        # ④ 先父后子,各自分批。
        #
        #    ⚠ 为什么不能"排好序塞进一个 list 一次 execute"?
        #    因为一次 execute 里,PG【不保证】按你给的顺序逐条处理 ——
        #    它可以重排、可以并行。你给的顺序只是个建议,不是承诺。
        #    真正的保证只有一个:【前一条 execute 完成后】,后一条才开始。
        #    所以"父在前子在后"必须体现为【两次 execute】,而不是一个 list 里的先后。
        #
        #    这类"看起来能跑,实际靠运气"的代码最坑 —— 小数据量时永远不出错,
        #    上了量或者换个 PG 版本就随机炸。
        for group in (parents, children):
            for i in range(0, len(group), _BATCH):
                self.session.execute(stmt, group[i : i + _BATCH])

        return len(chunks)

    # ── 读(检索侧用)──────────────────────────────────

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRow]:
        """按【给定顺序】取块。Milvus 给的是相似度排名,顺序不能丢。

        ┌─ 坑:SQL 的 IN 不保证顺序 ─┐
        │   SELECT * FROM chunks WHERE chunk_id IN ('b','a')
        │   可能返回 a, b —— 数据库爱怎么返回怎么返回(它按物理存储顺序扫)。
        │   直接用 = 相似度排名被打乱,而且【不报错】,只是答案悄悄变差。
        │
        │   解法(最简单的那个):一次查回来,在 Python 里按 chunk_ids 重排。
        │   查不到的 ID 跳过(Milvus 和 PG 可能短暂不同步,这是正常的)。
        └────────────────────────────┘

        """
        if not chunk_ids:
            return []

        # 一次查回来 —— 不是循环里查 N 次(那就是 N+1 查询,ORM 的经典性能坑)
        rows = self.session.scalars(
            select(ChunkRow).where(ChunkRow.chunk_id.in_(chunk_ids))
        ).all()

        # 用 dict 建索引:O(1) 查找。和你在 vector_store.py 里
        # self.parents.get(child.parent_id) 是同一个手法。
        by_id = {r.chunk_id: r for r in rows}

        # 按【要的顺序】取,查不到的跳过(Milvus 和 PG 可能短暂不同步 —— 正常,不是错)
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_parents(self, child_ids: list[str]) -> list[ChunkRow]:
        """Small-to-Big:给一串子块 ID,返回它们的父块(去重)。

        子块用来【找得准】,父块用来【答得全】—— 给 LLM 的是父块。

        为什么要【去重】:检索命中的 3 个子块很可能来自同一个父块
        (同一节里的三句话都跟问题相关 —— 这恰恰说明那一节是对的)。
        不去重就会把同一段父块文本重复三遍塞给 LLM:白烧 token,还可能
        让模型觉得"这事说了三遍,一定很重要"。

        为什么去重要【保序】:第 1 个子块的父块最相关,得排在最前面。
        用 set() 去重会打乱顺序 —— Python 的 set 不保证顺序。
        dict.fromkeys() 才对:它去重【且保留首次出现的顺序】
        (Python 3.7+ dict 有序,这是语言规范保证的,不是实现细节)。
        """
        children = self.get_chunks(child_ids)
        parent_ids = list(
            dict.fromkeys(c.parent_id for c in children if c.parent_id is not None)
        )
        return self.get_chunks(parent_ids)


# ── 评审八股:你要能答上来的四个 ──────────────────────────
# ① Repository 模式解决什么?
#    把持久化细节收口到一处,上层不依赖具体数据库 → 可替换、可测试。
# ② 为什么用 upsert 而不是"先查再插"?
#    "先查再插" 是两条语句,中间有窗口 → 并发下都判断"不存在"→ 双插冲突。
#    ON CONFLICT 是单条原子语句,PG 内部加锁保证。
# ③ 幂等是什么,为什么这里必须有?
#    同一操作执行 N 次和 1 次结果相同。灌 26 万条中途会挂,
#    幂等 = 直接重跑就行,不用"先清空再来过"。
#    我们的幂等锚是内容寻址的 document_id(sha256 前16位)。
# ④ 为什么 Session 要外部传入?
#    依赖注入 —— 事务边界由调用方决定。否则无法把多个写操作
#    合成一个原子事务,测试也没法用"跑完回滚"。
