"""统一中间结构(Intermediate Representation)——整条文档 ETL 管道的"普通话"。

上游:各种 parser(MinerU / 文本 / 以后的 Word/OCR)把原始文件解析成 Document。
下游:clean / chunk / metadata / embed 一律只认 Document,不认原始 PDF。

用 Pydantic 而不是 dataclass,因为这个 IR:
  ① 要落盘再读回(data/parsed/*.json)→ model_dump_json / model_validate_json 自动搞定嵌套
  ② 吃 parser 的外部脏数据 → 构造时自动校验类型,坏数据在入口就被拦(Parse, don't validate)
  ③ 和 参考架构 一致(它全用 Pydantic)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# 允许的元素类型——把"领域里到底有哪几种结构单元"显式枚举出来。
# 下面 Element 的 validator 会拿它做校验:出现不认识的 type 直接报错(fail-fast)。
ELEMENT_TYPES: frozenset[str] = frozenset(
    {
        "heading",  # 标题(带 level:1=章 2=节 3=小节…),对应 MinerU 的 text + text_level
        "paragraph",  # 正文段落,对应 MinerU 的 text(无 text_level)
        "table",  # 表格,meta 里放 table_body(HTML)/caption/img_path
        "formula",  # 公式,meta 里放 latex
        "figure",  # 图 / 图表,meta 里放 img_path/caption(MinerU 的 image + chart)
        "list",  # 列表,对应 MinerU 的 list
        "footnote",  # 脚注,对应 MinerU 的 page_footnote(可能含来源/补充,故不当家具丢弃)
    }
)


class Element(BaseModel):
    """一篇文档里的一个"最小结构单元"。直接对应 MinerU content_list.json 里的一条。"""

    type: str  # 取值见 ELEMENT_TYPES;clean 靠它剔家具、chunk 靠它决定怎么切
    text: str  # 文字;table 存 HTML,formula 存 LaTeX
    page: int  # 页码(从 0 开始),用于溯源"这句话在原文第几页"
    level: int | None = None  # 标题层级;只有 type=heading 才有值,其余为 None
    bbox: list[float] | None = (
        None  # 版面坐标 [x0,y0,x1,y1];现在不用,留给以后高亮原文/多模态
    )
    meta: dict = Field(
        default_factory=dict
    )  # 各类型的专属数据袋(latex/table_body/img_path…)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        """质量门禁:构造 Element 时就校验 type 合法,坏数据进不了管道。"""
        if v not in ELEMENT_TYPES:
            raise ValueError(f"未知元素类型 {v!r};合法类型:{sorted(ELEMENT_TYPES)}")
        return v


class Chunk(BaseModel):
    """检索单元:chunk 层把 Document 切成一个个 Chunk,embed/入库/检索都认它。"""

    chunk_id: str  # 唯一 ID(如 {document_id}_{序号})
    document_id: str  # 外键 → 属于哪篇 Document,溯源用
    text: str  # 这一块的文本(送去 embedding 的内容)
    section_path: list[str] = Field(
        default_factory=list
    )  # 章节路径 ["1 实验部分","1.1 主要原料"]
    page_start: int = 0  # 起始页(溯源:答案在第几页)
    page_end: int = 0  # 结束页
    parent_id: str | None = None  # 子块 → 父块的 ID;父块为 None(Small-to-Big)
    meta: dict = Field(
        default_factory=dict
    )  # 携带 Document 的元数据(标题/年份/来源…)便于过滤


class Document(BaseModel):
    """一篇文档解析后的完整结果。下游只吃 elements 这个列表。"""

    document_id: str  # 全局唯一 ID;所有下游(chunk/向量/manifest)的外键 = 幂等的锚
    filename: str  # 原始文件名
    file_type: str  # pdf / txt / docx…;parser 路由靠它(支撑"N 种类型")
    sha256: str  # 文件内容哈希;文件级去重(字节完全相同 = 同一文件)
    source: str  # 来源:期刊论文 / 标准规范 / 爬虫;元数据过滤 + 权威性排序
    parser: str  # 谁解析的:mineru / text;数据血缘,换解析器后可追溯
    elements: list[Element] = Field(default_factory=list)  # ★ 结构单元列表,下游唯一入口
    meta: dict = Field(
        default_factory=dict
    )  # 文档级元数据:标题/作者/年份/DOI(后面 metadata 层填)


# ── 你要能讲的四个八股 ──────────────────────────────────────────────
# ① Parse, don't validate:让非法数据在入口(构造 Element/Document 时)就无法存在,
#    而不是一路带到下游才炸。@field_validator 就是把"质量门禁"焊在类型边界上。
# ② 落盘/读回怎么做?
#    写:doc.model_dump_json() → 存 data/parsed/doc_0001.json
#    读:Document.model_validate_json(文本) → 嵌套的 Element 自动重建 + 再次校验
#    dataclass 做不到这点(asdict 是单向的,读回要手写 Element(**e))——这是选 Pydantic 的关键。
# ③ 为什么可变默认值用 Field(default_factory=dict)?
#    dataclass 里 = {} 会让所有实例共享同一个 dict(可变默认值陷阱)。
#    Pydantic 其实会自动深拷贝默认值、不踩这个坑,但用 default_factory 更显式、更地道。
# ④ 为什么只存 level、不存 section_path(章节路径)?
#    section_path 是"派生数据",可由 level 序列在 chunk 阶段算出。
#    IR 里只放"原始信号",派生结构下游算——避免重复存储、避免不一致。
