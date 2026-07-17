"""第4层 chunk:把 Document 切成检索单元 Chunk(独立阶段,在 clean 之后)。

两步:
  ① _section_paths:用 heading 的 level 重建"每个元素属于哪条章节路径"(栈算法)—— 你写
  ② 父子分块:按章节聚成父块、段落切成子块 —— 下一步
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ruixue_agent.ingestion.schema import Chunk, Document
from ruixue_agent.ingestion.utils.table import table_to_sentences

# ── 子块垃圾过滤(两层:长度粗筛 + 模式精打)──
# 实测 43508 个子块的长度分布:
#   [0-10) 3.12% —— 100% 垃圾('其中'/'文献标志码:A'/'(下转第64页)'/'同2.1.6')
#   [10-20) 5.52% —— 约七成垃圾,但【图注在这个区间】('(a) 不同处理番茄叶片 SPAD 值' 18字,有价值)
#   [20+)        —— 大多有价值('全市平均回收利用率达到了79.5%(见表3)。')
# 所以:长度只砍 <10(零误伤),其余垃圾按【特征】精准打 —— 不误伤图注。
_MIN_CHILD_CHARS = 10

# 卷首/排版噪声的特征(全部来自真实语料采样)
_NOISE = re.compile(
    r"中图分类号|文献标[志识]码|文章编号|收稿日期|修回日期|责任编辑"
    r"|^\s*分类号|^\s*doi\s*[:：]|^\s*DOI\s*[:：]"
    r"|^\s*\(?[下上]转第?\s*\d+\s*页"  # (下转第64页)
    r"|^\s*(参照|同)\s*[\d.]+\s*。?\s*$"  # 参照 2.2.2.4。/ 同2.1.6
    r"|^\s*(硕士|博士)研究生\s*[:：]|^\s*学院\s*[(（]"
)

# ── 子块限长 ──
# bge-small-zh 最多吃 512 token,超了会被【悄悄截断】→ 超出部分永远进不了向量、检索不到。
# 所以子块必须限长。父块不限(它不做 embedding,只当上下文返回)。
_MAX_CHILD_CHARS = 400
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_MAX_CHILD_CHARS,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "!", "?", ";", ",", " ", ""],  # 中文优先按句切
)

# 可以安全切分的类型;table(HTML)/formula(LaTeX) 不能切——切断结构比截断更糟
_SPLITTABLE = {"paragraph", "list", "footnote", "figure"}


def _is_junk(text: str) -> bool:
    """这段文本该不该丢弃(不值得进向量库)?

    轮到你写(一个 if,两个条件 —— 命中任一就是垃圾):
      1. 太短:去掉首尾空白后,长度 < _MIN_CHILD_CHARS
      2. 命中卷首/排版噪声特征:_NOISE.search(text)

    提示:
        t = text.strip()
        return len(t) < _MIN_CHILD_CHARS or bool(_NOISE.search(t))
    """
    # 、你写
    t = text.strip()
    return len(t) < _MIN_CHILD_CHARS or bool(_NOISE.search(t))


def _child_texts(el) -> list[str]:
    """一个元素 → 该产出几段子块文本(垃圾直接返回空列表 = 不产出任何子块)。"""
    # 表格特殊:它的 text 是 HTML,直接 embedding 等于喂标签,而且实测 84% 超长会被截断。
    # → 转成【行级人话】,每行自带表名+列名,独立完整、天然不超长(实测超长率 84%→2.2%)。
    #   原始 HTML 仍在 el.meta 里,结构不丢。
    if el.type == "table":
        return table_to_sentences(el.text, el.meta.get("caption", ""))

    if _is_junk(el.text):
        return []
    if el.type in _SPLITTABLE and len(el.text) > _MAX_CHILD_CHARS:
        # 切开后每一段也要再过一遍垃圾过滤(切出来的碎尾巴可能是垃圾)
        return [p for p in _splitter.split_text(el.text) if not _is_junk(p)]
    return [el.text]


def _section_paths(elements: list) -> list[list[str]]:
    """给每个 element 算出它所属的章节路径。

    用一个栈跟踪"当前从外到内的标题",例:处理到 1.1 小节的段落时,
    栈 = [(1,'论文标题'), (2,'1 实验部分'), (2,'1.1 主要原料')]。

    规则:
      - 遇到 heading:先弹掉栈顶所有 level >= 当前的(说明离开了那些同级/更深的节),再压入自己
      - 每个元素(标题或正文)的路径 = 栈里所有标题的文字列表
    """
    paths: list[list[str]] = []
    stack: list[tuple[int, str]] = []  # [(level, heading_text), ...]
    for el in elements:
        # 你写(两小步):
        # 1) 若 el.type == "heading":
        #      while stack 非空 and stack[-1][0] >= el.level:  stack.pop()
        #      stack.append((el.level, el.text))
        if el.type == "heading":
            while stack and stack[-1][0] >= el.level:
                stack.pop()
            stack.append((el.level, el.text))

        # 2) paths.append( 栈里每个标题的文字组成的列表 )
        paths.append([text for (level, text) in stack])
    return paths


def _group_into_sections(elements: list, paths: list[list[str]]) -> list[tuple]:
    """把"连续、同章节路径"的有效元素归成一段段小节。

    跳过标题(已在 path 里)和空元素。返回 [(章节路径, [该节的元素]), ...]。
    用"累积当前节、遇到换节就收尾"的写法,读起来像大白话。
    """
    sections: list[tuple] = []
    current_path = None  # 当前正在收集的这一节的路径
    current_els: list = []  # 当前这一节收集到的元素
    for el, path in zip(elements, paths):
        if el.type == "heading" or not el.text.strip():
            continue  # 标题、空元素:不进块
        if path != current_path:  # 章节变了 → 上一节收尾,开新的一节
            if current_els:
                sections.append((current_path, current_els))
            current_path = path
            current_els = []
        current_els.append(el)  # 把当前元素加进当前节
    if current_els:  # 循环结束,别漏了最后一节
        sections.append((current_path, current_els))
    return sections


def _section_to_chunks(doc_id: str, si: int, path: list[str], els: list) -> list[Chunk]:
    """一个小节 → 1 个父块(整节)+ N 个子块(节内每个元素),子块都指向父块。"""
    parent_id = f"{doc_id}_s{si}"
    parent = Chunk(
        chunk_id=parent_id,
        document_id=doc_id,
        text="\n".join(e.text for e in els),  # 父块 = 整节文本拼起来
        section_path=path,
        page_start=els[0].page,
        page_end=els[-1].page,
        parent_id=None,
        meta={"kind": "parent"},  # 父块没有父
    )
    children: list[Chunk] = []
    ci = 0
    for e in els:
        for piece in _child_texts(e):  # 一个元素可能产出多段(超长被切开)
            children.append(
                Chunk(
                    chunk_id=f"{parent_id}_c{ci}",
                    document_id=doc_id,
                    text=piece,  # 子块 = 一段(≤400字,保证不被 embedding 截断)
                    section_path=path,
                    page_start=e.page,
                    page_end=e.page,
                    parent_id=parent_id,  # 子块指向父块
                    meta={"kind": "child", "element_type": e.type},
                )
            )
            ci += 1
    return [parent] + children


def chunk_document(doc: Document) -> list[Chunk]:
    """把 Document 切成父子 Chunk(Small-to-Big)。读起来就三步:

      1. 算每个元素的章节地址        _section_paths
      2. 按章节把元素分成一段段小节  _group_into_sections
      3. 每节做成 1 父块 + N 子块     _section_to_chunks

    检索时:命中【子块】(精确)→ 顺 parent_id 取回【父块】(整节上下文)给大模型。
    """
    paths = _section_paths(doc.elements)
    sections = _group_into_sections(doc.elements, paths)
    chunks: list[Chunk] = []
    for si, (path, els) in enumerate(sections):
        chunks += _section_to_chunks(doc.document_id, si, path, els)
    return chunks
