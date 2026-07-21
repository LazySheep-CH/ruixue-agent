"""数据库模型(SQLAlchemy 2.0 声明式)—— PG 是数据的 source of truth。

为什么 persistence/ 是【独立模块】而不是塞进 ingestion:
    表是 ingestion(写)和 rag(读)【共用】的。塞进 ingestion 等于说
    "这是解析管道的私有东西",语义就错了。参考架构 也是把 persistence/ 和 agents/ 平级放。

为什么用 SQLAlchemy 模型而不是手写 SQL:
    ① 模型是【代码】:有类型、能被 IDE 检查、能被测试
    ② Alembic 能 diff 模型和数据库,【自动生成】migration
    ③ 改字段 = 改模型 + 生成一个 migration;手写 CREATE TABLE IF NOT EXISTS 改不了已有表

职责边界:PG 存【数据】(元数据/文本/父子关系),Milvus 只存【索引】(向量)。
        数据是资产要可靠可查有事务;索引是派生物可随时重建。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有模型的基类。Alembic 靠 Base.metadata 知道该建哪些表。"""


class DocumentRow(Base):
    """一篇文档的元数据。

    comment= 会变成数据库里的 COMMENT ON —— 新人 `\\d+ documents` 就能看懂每个字段,
    不用翻代码问人。注释必须写在【模型】上而不是 migration 里手写 COMMENT ON:
    模型是唯一事实来源,写别处 `alembic check` 会判定漂移(踩过)。
    """

    __tablename__ = "documents"

    # document_id = sha256 前16位 = 内容寻址(git/Docker 同思想):
    # 同样的文件永远同样的 ID → 幂等、天然去重、无需中心协调
    document_id: Mapped[str] = mapped_column(
        String(16),
        primary_key=True,
        comment="内容寻址 ID = sha256 前16位。同内容永远同 ID → 幂等、天然去重",
    )
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="来源类别:期刊论文 / 标准规范"
    )
    parser: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="数据血缘:这篇是谁解析的(mineru/text)。换解析器后能定位受影响的数据",
    )

    # 内容元数据(metadata 阶段从正文挖出来的,不是文件属性)
    title: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    doi: Mapped[str | None] = mapped_column(Text)
    # PG 原生数组:能 WHERE keywords @> ARRAY['PBAT'] 精确匹配
    # (拼成 "PBAT,PLA" 就只能 LIKE '%PBAT%',会误匹配 "PBAT-2")
    keywords: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text), comment="PG 原生数组,支持 keywords @> ARRAY['PBAT'] 精确匹配"
    )
    abstract: Mapped[str | None] = mapped_column(Text)

    # TIMESTAMPTZ(带时区)不是 TIMESTAMP —— PG 里 TIMESTAMP 存的是"墙上时间",
    # 不知道自己是哪个时区的。跨时区部署 / 夏令时切换时会算错,而且是【静默】算错。
    # 规矩:时间字段一律 timezone=True。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # 自动更新靠 migration 里的【触发器】,不是靠这里。
    # (踩过:光加字段没触发器 → updated_at 永远等于插入时间,是个摆设)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_documents_year", "year"),
        Index("idx_documents_source", "source"),
        # GIN 索引:数组包含查询专用
        Index("idx_documents_keywords", "keywords", postgresql_using="gin"),
        # 末尾的 dict = 表级选项(注意:必须是 tuple 的最后一个元素)
        {"comment": "文档元数据。一行 = 一篇论文/标准"},
    )


class TermDfRow(Base):
    """词 → 文档频率(出现在多少个父块里)。BM25 挑词用。

    为什么要单独一张表:
        词法检索的成败取决于【只用罕见词查】。实测:
          全部词 OR      R@10 0.350   8497 ms
          只留 DF<1% 的  R@10 0.525    140 ms   ← 召回 +50%,快 60 倍
        因为一个罕见词和七个常见词做 OR,常见词会把候选灌到几万,
        罕见词的定位能力被淹没。

        要挑罕见词就得知道每个词的文档频率。PG 的 ts_stat 能算,
        但它是【全表扫】—— 不能每次查询都跑。所以物化成一张表。

    这是【派生数据】(可以从 chunks 重新算出来),和索引一个性质:
    丢了不要紧,重跑 scripts/build_term_df.py 就有。
    """

    __tablename__ = "term_df"

    term: Mapped[str] = mapped_column(Text, primary_key=True)
    ndoc: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="这个词出现在多少个父块里"
    )

    __table_args__ = (
        Index("idx_term_df_ndoc", "ndoc"),
        {"comment": "词频表(派生数据,可重建)。BM25 靠它挑出罕见词,只用罕见词查"},
    )


class ChunkRow(Base):
    """一个检索单元。父块和子块【同一张表】—— 它们是同一种东西,只差 parent_id 空不空。"""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # 外键 + 级联删除:删一篇文档,它的块自动全删,不留孤儿数据
    document_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    # 自引用外键:指向【本表】另一行 = 父子关系。父块删了,子块的 parent_id 置空而非悬空
    parent_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("chunks.chunk_id", ondelete="SET NULL")
    )

    kind: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="parent=完整章节(给 LLM 看) / child=细粒度片段(拿去 embedding)",
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        comment="从文档根到本块的标题路径,如 {结果与讨论, 降解率}。栈算法重建章节树得到",
    )
    page_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    element_type: Mapped[str | None] = mapped_column(String(16))

    # jieba 分词后的词串(空格分隔)—— BM25 的输入。
    #
    # 为什么要单独存一列,不在触发器里现算:
    #   触发器跑在 PostgreSQL 进程里,调不到 Python 的 jieba。
    #   所以反过来:Python 侧分好词写进这一列,PG 侧的触发器只负责
    #   把它转成 tsvector。分词逻辑留在 Python 里的好处是能测、能改、能加词典。
    text_tokens: Mapped[str | None] = mapped_column(
        Text,
        comment="jieba 分词后的词串(空格分隔),由 Python 侧写入;text_tsv 由触发器从它生成",
    )

    # 全文检索向量 —— BM25 的索引。
    # 实测过纯向量检索的缺陷:问 "PBAT用什么牌号" 会命中 PLA 那条(0.502 vs 0.477),
    # 因为 embedding 不保证关键词精确匹配。
    #
    # 0002 之前它是从 text 直接生成的 —— 那是【废的】:
    #   simple 配置靠空格切词,中文没空格 → 整句一个 token。
    #   实测搜「地膜厚度」「厚度」「PBAT」「0.010」全部 false,
    #   连夹在中文里的英文数字都搜不到(和中文粘在一起了)。
    # 0002 起改成从 text_tokens 生成。
    text_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        comment="全文检索向量,触发器从 text_tokens 生成(不是从 text —— simple 配置不分中文词)",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("kind IN ('parent', 'child')", name="chunks_kind_check"),
        # 业务规则写进数据库:父块不能有父,子块必须有父。
        # 和 schema.py 的 @field_validator 同一个思想:Parse, don't validate —— 让非法状态无法存在。
        # 区别:那个挡的是【进程内】的脏数据,这个挡的是【任何写库路径】(包括别人手工 INSERT)
        CheckConstraint(
            "(kind = 'parent' AND parent_id IS NULL) OR (kind = 'child' AND parent_id IS NOT NULL)",
            name="chunks_parent_rule",
        ),
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_parent", "parent_id"),
        Index("idx_chunks_kind", "kind"),
        Index("idx_chunks_text_tsv", "text_tsv", postgresql_using="gin"),  # 全文检索
        {
            "comment": "检索单元。父块子块同表,靠 kind 区分:"
            "子块进 Milvus 做向量检索,命中后返回父块给 LLM(Small-to-Big)"
        },
    )
