"""切块阶段:Document → 父子两级 Chunk(Small-to-Big)。

父块 = 一个章节的完整文本(不做 embedding,检索命中后作为上下文返回);
子块 = 章节内的段落级片段(做 embedding,负责被检索命中)。
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ruixue_agent.ingestion.schema import Chunk, Document
from ruixue_agent.ingestion.utils.table import table_to_sentences

# 子块垃圾过滤:长度粗筛 + 模式精筛两层。
# 43508 个子块的长度分布实测:
#   [0,10)  占 3.12%,全部为噪声("其中"/"文献标志码:A"/"(下转第64页)"等)
#   [10,20) 占 5.52%,约七成噪声,但图注落在此区间(如 18 字的图题,有检索价值)
#   [20,∞)  大多有价值
# 因此长度阈值只砍 <10(零误伤),其余噪声按特征正则精筛,避免误杀图注。
_MIN_CHILD_CHARS = 10

# 卷首/排版噪声特征,均来自真实语料采样
_NOISE = re.compile(
    r"中图分类号|文献标[志识]码|文章编号|收稿日期|修回日期|责任编辑"
    r"|^\s*分类号|^\s*doi\s*[:：]|^\s*DOI\s*[:：]"
    r"|^\s*\(?[下上]转第?\s*\d+\s*页"  # (下转第64页)
    r"|^\s*(参照|同)\s*[\d.]+\s*。?\s*$"  # 参照 2.2.2.4 / 同2.1.6
    r"|^\s*(硕士|博士)研究生\s*[:：]|^\s*学院\s*[(（]"
)

# 子块限长:embedding 模型(bge-small-zh)最长 512 token,超长部分会被
# 静默截断、永远无法被检索。父块不限长(不做 embedding)。
_MAX_CHILD_CHARS = 400
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_MAX_CHILD_CHARS,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "!", "?", ";", ",", " ", ""],  # 中文优先按句切
)

# 可安全切分的类型。table(HTML)/formula(LaTeX)不切 —— 切断结构比截断更糟
_SPLITTABLE = {"paragraph", "list", "footnote", "figure"}


def _is_junk(text: str) -> bool:
    """判定文本是否为不值得入库的噪声:过短,或命中排版噪声特征。"""
    t = text.strip()
    return len(t) < _MIN_CHILD_CHARS or bool(_NOISE.search(t))


def _child_texts(el) -> list[str]:
    """一个元素产出的子块文本列表;噪声返回空列表。"""
    # 表格的 text 是 HTML,直接 embedding 等于编码标签,且实测 84% 超长。
    # 转为行级句子(每行自带表名+列名,独立完整),超长率降至 2.2%;
    # 原始 HTML 保留在 el.meta 中,结构不丢失。
    if el.type == "table":
        return table_to_sentences(el.text, el.meta.get("caption", ""))

    if _is_junk(el.text):
        return []
    if el.type in _SPLITTABLE and len(el.text) > _MAX_CHILD_CHARS:
        # 切分产生的碎片同样过一遍噪声过滤
        return [p for p in _splitter.split_text(el.text) if not _is_junk(p)]
    return [el.text]


def _section_paths(elements: list) -> list[list[str]]:
    """为每个元素计算其章节路径(从文档根到所在小节的标题序列)。

    用栈维护"当前由外到内的标题链"。遇到 heading 时,先弹出栈顶所有
    level >= 当前标题的项(离开同级或更深的小节),再压入自身;
    任意元素的路径即当前栈中标题的文本列表。
    """
    paths: list[list[str]] = []
    stack: list[tuple[int, str]] = []  # [(level, heading_text), ...]
    for el in elements:
        if el.type == "heading":
            while stack and stack[-1][0] >= el.level:
                stack.pop()
            stack.append((el.level, el.text))
        paths.append([text for (level, text) in stack])
    return paths


def _group_into_sections(elements: list, paths: list[list[str]]) -> list[tuple]:
    """把连续且章节路径相同的有效元素归为一个小节。

    标题本身与空元素不进块(标题已体现在 path 中)。
    返回 [(章节路径, [该节元素]), ...]。
    """
    sections: list[tuple] = []
    current_path = None
    current_els: list = []
    for el, path in zip(elements, paths, strict=False):
        if el.type == "heading" or not el.text.strip():
            continue
        if path != current_path:  # 换节:上一节收尾
            if current_els:
                sections.append((current_path, current_els))
            current_path = path
            current_els = []
        current_els.append(el)
    if current_els:
        sections.append((current_path, current_els))
    return sections


def _section_to_chunks(doc_id: str, si: int, path: list[str], els: list) -> list[Chunk]:
    """一个小节 → 1 个父块(整节)+ N 个子块,子块经 parent_id 指向父块。"""
    parent_id = f"{doc_id}_s{si}"
    parent = Chunk(
        chunk_id=parent_id,
        document_id=doc_id,
        text="\n".join(e.text for e in els),
        section_path=path,
        page_start=els[0].page,
        page_end=els[-1].page,
        parent_id=None,
        meta={"kind": "parent"},
    )
    children: list[Chunk] = []
    ci = 0
    for e in els:
        for piece in _child_texts(e):  # 超长元素会切出多段
            children.append(
                Chunk(
                    chunk_id=f"{parent_id}_c{ci}",
                    document_id=doc_id,
                    text=piece,
                    section_path=path,
                    page_start=e.page,
                    page_end=e.page,
                    parent_id=parent_id,
                    meta={"kind": "child", "element_type": e.type},
                )
            )
            ci += 1
    return [parent] + children


def chunk_document(doc: Document) -> list[Chunk]:
    """Document → 父子 Chunk 列表。

    流程:计算章节路径 → 按路径聚合小节 → 每节产出 1 父块 + N 子块。
    检索时命中子块(精确),经 parent_id 取回父块(完整上下文)交给生成层。
    """
    paths = _section_paths(doc.elements)
    sections = _group_into_sections(doc.elements, paths)
    chunks: list[Chunk] = []
    for si, (path, els) in enumerate(sections):
        chunks += _section_to_chunks(doc.document_id, si, path, els)
    return chunks
