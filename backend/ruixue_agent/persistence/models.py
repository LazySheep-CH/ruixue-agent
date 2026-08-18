"""数据库模型(SQLAlchemy 2.0 声明式)—— PG 是数据的 source of truth。

为什么 persistence/ 是独立模块而不是塞进 ingestion:
    表是 ingestion(写)和 rag(读)共用的。塞进 ingestion 等于说
    "这是解析管道的私有东西",语义就错了。persistence/ 与 agents/ 平级。

为什么用 SQLAlchemy 模型而不是手写 SQL:
    1) 模型是代码:有类型、能被 IDE 检查、能被测试
    2) Alembic 能 diff 模型和数据库,自动生成migration
    3) 改字段 = 改模型 + 生成一个 migration;手写 CREATE TABLE IF NOT EXISTS 改不了已有表

职责边界:PG 存数据(元数据/文本/父子关系),Milvus 只存索引(向量)。
        数据是资产要可靠可查有事务;索引是派生物可随时重建。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有模型的基类。Alembic 靠 Base.metadata 知道该建哪些表。"""


class DocumentRow(Base):
    """一篇文档的元数据。

    comment= 会变成数据库里的 COMMENT ON —— 新人 `\\d+ documents` 就能看懂每个字段,
    不用翻代码问人。注释必须写在模型上而不是 migration 里手写 COMMENT ON:
    模型是唯一事实来源,写在别处会被 alembic check 判定为漂移。
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
    # 不知道自己是哪个时区的。跨时区部署 / 夏令时切换时会算错,而且是静默算错。
    # 规矩:时间字段一律 timezone=True。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # 自动更新由 migration 中的触发器完成。仅有字段而无触发器时,
    # updated_at 恒等于插入时间,形同虚设。
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
        词法检索的成败取决于只用罕见词查。实测:
          全部词 OR      R@10 0.350   8497 ms
          只留 DF<1% 的  R@10 0.525    140 ms   ← 召回 +50%,快 60 倍
        因为一个罕见词和七个常见词做 OR,常见词会把候选灌到几万,
        罕见词的定位能力被淹没。

        要挑罕见词就得知道每个词的文档频率。PG 的 ts_stat 能算,
        但它是全表扫—— 不能每次查询都跑。所以物化成一张表。

    这是派生数据(可以从 chunks 重新算出来),和索引一个性质:
    丢了不要紧,重跑 scripts/build_term_df.py 就有。
    """

    __tablename__ = "term_df"

    term: Mapped[str] = mapped_column(Text, primary_key=True)
    ndoc: Mapped[int] = mapped_column(Integer, nullable=False, comment="这个词出现在多少个父块里")

    __table_args__ = (
        Index("idx_term_df_ndoc", "ndoc"),
        {"comment": "词频表(派生数据,可重建)。BM25 靠它挑出罕见词,只用罕见词查"},
    )


class ChunkRow(Base):
    """一个检索单元。父块和子块同一张表—— 它们是同一种东西,只差 parent_id 空不空。"""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # 外键 + 级联删除:删一篇文档,它的块自动全删,不留孤儿数据
    document_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    # 自引用外键:指向本表另一行 = 父子关系。父块删了,子块的 parent_id 置空而非悬空
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
    # 0002 之前它是从 text 直接生成的 —— 那是废的:
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
        # 区别:那个挡的是进程内的脏数据,这个挡的是任何写库路径(包括别人手工 INSERT)
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


class UserRow(Base):
    """用户账号。

    为什么用户表放 persistence/ 而不是 ruixue_app/:
        它是数据,和文档/chunk 一样属于持久化层;app 层只做 HTTP,不管存储。
        这样 CLI/脚本/将来的 IM 渠道也能复用同一套用户体系。

    字段说明:
        username        登录名,唯一,建索引(登录时按它查)
        password_hash   bcrypt 哈希串(含盐),绝不存明文
        is_active       软禁用:封号不删数据(删了会破坏历史会话的外键语义)
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RunRow(Base):
    """一次 agent 运行(Run)的记录。

    为什么要这张表 —— 解决"断线就白跑"的问题:
        原先 agent 直接在 SSE 请求里跑,客户端一断(刷新页面、切网络),
        生成器被取消 → agent 半路停下 → 钱花了、结果没有、用户还得重问。
        改成:创建 Run → 后台跑(不绑请求生命周期)→ 事件写 Redis Stream。
        客户端断了照跑完;重连时凭 run_id 把已产生的事件补发出来。

    这张表是 Run 的权威状态(Redis 里的事件流是可丢的缓存,有 TTL);
    进程重启后靠它知道哪些 Run 是残留的(见 runs.reap_stale)。

    字段说明:
        run_id      对外暴露的运行编号(uuid),客户端凭它查询/重连
        user_id     归属用户 —— 查询时必须校验,否则猜到 run_id 就能看别人的对话
        thread_id   会话(已含 user 前缀),用于 checkpointer
        status      running / succeeded / failed
        answer      成功时的最终答案(供刷新后直接取回,不必重放事件)
        error       失败时的脱敏原因(不放堆栈,那只进日志)
    """

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(160), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="running")
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MemoryRow(Base):
    """长期记忆:跨会话记住这个用户的事实。

    为什么需要长期记忆(先讲问题):
    老王上周说过"我在尉犁有 50 亩地种棉花"。这周他问"帮我算下用量" ——
    没有长期记忆的话,系统只能反问他面积,因为上周的对话早不在上下文里了。
    短期记忆(滑动窗口 + 摘要)只在一次会话内有效,会话一结束就没了。

    决定一:存抽取出的事实,不存原始对话:
    两种做法的取舍:
        存原始对话  —— 实现简单,但检索噪声大(半句闲聊也会被召回)、
                       浪费 token、而且用户想删某条信息时无从下手。
        存抽取的事实 —— 多一次 LLM 调用的成本,换来:精准、可审计、
                       可按条删除(隐私合规的前提)、token 省。

    我们选后者。地膜场景的记忆高度结构化(地点/面积/作物/配方/生育期),
    抽成事实几乎没有信息损失,反而把闲聊噪声滤掉了。

    决定二:为什么不只用关系库(面经必追的一问):
    "既然是存用户偏好标签,为什么不用 MySQL?" —— 因为记忆有两种查法:
        精确查:"这个用户的地块在哪" → 关系库,一条 SQL
        语义查:"当前这个问题,和用户以前说过的什么有关" → 只能靠向量

    第二种是关系库做不到的:用户说过"我这边风大",现在问"选哪种配方",
    关键词一个都不重合,但语义上高度相关(风大 → 要看拉伸强度)。

    所以我们两边都存:事实进这张表(可查、可编辑、可删除、可审计),
    同时把事实文本向量化进 Milvus 做语义召回。PG 是权威,向量是索引
    —— 和文档那套完全一样的分工,向量丢了重建即可。

    决定三:什么时候写:
    一次运行结束之后异步抽取,不阻塞用户。抽取失败不影响主流程 ——
    记忆是锦上添花,不能因为它拖垮回答。
    """

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 内容寻址:同一个用户重复说同一件事,不会存两条(sha256 前 16 位)
    memory_id: Mapped[str] = mapped_column(String(16), primary_key=False, unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    # 事实分类。分类不是装饰:检索时可以按类过滤(算用量只需要 plot 类),
    # 用户想删"我的地块信息"时也能按类批量删。
    kind: Mapped[str] = mapped_column(
        String(24), nullable=False, comment="plot=地块 / crop=作物 / preference=偏好 / other"
    )
    text: Mapped[str] = mapped_column(Text, nullable=False, comment="抽取出的事实,一句话")
    # 血缘:这条记忆是哪次运行抽出来的。用户质疑"你怎么知道我有50亩地"时能回溯。
    source_run_id: Mapped[str | None] = mapped_column(String(36), index=True)
    # 置信度低的先存但不注入,留人工/后续确认。目前抽取器只产出 high/low 两档。
    confidence: Mapped[str] = mapped_column(String(8), nullable=False, server_default="high")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # 软删除:用户删记忆时置位,不物理删 —— 保留审计痕迹,也避免下次又被抽出来
    deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    __table_args__ = (
        # 检索主路径:按用户 + 未删除 取记忆
        Index("idx_memories_user_alive", "user_id", "deleted"),
        {"comment": "长期记忆。一行 = 关于某用户的一条事实"},
    )


class DatasetRow(Base):
    """用户上传的实测数据表(CSV/Excel)。

    为什么进数据库而不是文件系统:
    最直觉的做法是把上传的文件存到磁盘,工具再去读。那会引入三个新问题:
    路径逃逸(文件名来自用户输入)、清理与配额(谁删、何时删、满了怎么办)、
    以及"agent 需要一个能读文件的工具"——而我们所有工具目前都是只读且
    不碰文件系统的。

    存进 PG 之后:
      · 归属校验和 runs 走同一套(user_id 必须匹配),猜到 id 也拿不到别人的;
      · 工具只接收一个 uuid 形式的 dataset_id,模型编不出别人的 id;
      · 删除就是一行 DELETE,没有孤儿文件;
      · 备份跟着 PG 走,不用单独备一份文件目录。

    和 ChunkRow 的取舍不同:那边正文大、行数多(26 万),值得单独设计;
    这边上限 5000 行(见 analysis/schema.MAX_ROWS),整表塞 JSON 完全够用,
    不值得为它再建一张"数据行"表。

    columns 存的是映射结果不是原始表头:
    上传时就把用户的列名归一到模型特征名(见 analysis/schema.map_columns),
    并把"认出了什么、没认出什么"一起存下来。这样分析工具拿到的是确定的结构,
    不必每次重新猜列名 —— 归一只做一次,做在入口。
    """

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    # 原始文件名,只用于展示。不用它拼路径(我们根本不落盘),也不进 HTTP 头。
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_cols: Mapped[int] = mapped_column(Integer, nullable=False)
    # {"features": {标准名: 原列名}, "targets": {DR/TS/WVTR: 原列名}, "unknown": [...]}
    columns: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # 数据行:[{标准名或原列名: 值}, ...]。上限 5000 行,JSONB 足够。
    rows: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    __table_args__ = (
        # 列表页主路径:按用户列最近上传的
        Index("idx_datasets_user_time", "user_id", "created_at"),
        {"comment": "用户上传的实测数据表。一行 = 一次上传"},
    )


class UserDocRow(Base):
    """用户上传的知识文档(PDF/TXT/MD)。一行 = 一份文档。

    正文不存这张表:切块后的文本在 user_doc_chunks,向量在 Milvus。
    这行只承担列表页和配额统计,所以字段克制 —— 文件名、块数、时间。
    归属与不落盘的理由同 DatasetRow,不再重复。
    """

    __tablename__ = "user_docs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    n_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    __table_args__ = (
        Index("idx_user_docs_user_time", "user_id", "created_at"),
        {"comment": "用户知识库文档。一行 = 一次上传"},
    )


class UserDocChunkRow(Base):
    """用户文档的切块正文。检索命中 chunk_id 后从这里取原文。

    不复用平台的 chunks 表:那张表的形状(父子块、tsv、章节路径)是给
    MinerU 结构化产物设计的,用户裸文本用不上;更重要的是平台语料没有
    user_id,混在一张表里,每条平台检索都要背一个"别把用户块查出来"的
    过滤条件 —— 忘一处就是泄露。分表让隔离成为默认而不是纪律。
    """

    __tablename__ = "user_doc_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    doc_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = ({"comment": "用户文档切块。检索走 Milvus,这里存正文"},)
