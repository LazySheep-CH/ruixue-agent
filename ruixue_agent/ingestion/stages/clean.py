"""第3层 clean:对 Document 里的文本做规范化。作用在 IR 上 → 所有 parser 共享。

分工:
- clean_text(str)  : 纯字符串规范化(NFKC/去乱码/去 sub-sup 标签/统一空白)
- clean_document(doc): 把 clean_text 施加到 Document,但【只清 prose,跳过 table(HTML)/formula(LaTeX)】
                       —— 因为对结构化文本做文本规范化会破坏它的结构。
"""

from __future__ import annotations

import re

import ftfy

from ruixue_agent.ingestion.utils.latex import unwrap_inline_latex
from ruixue_agent.ingestion.schema import Document

# <sub>/<sup> 标签去掉、保留里面的字(如 压<sub>力</sub> → 压力,把被上下标拆开的词接回来)
_SUBSUP = re.compile(r"</?su[bp]>")
# 控制/不可见字符(保留 \t \n)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")  # 多个空格/制表符 → 一个
_BLANKS = re.compile(r"\n{3,}")  # 3+ 连续空行 → 2


def clean_text(text: str) -> str:
    """字符串级规范化。每步作用见注释。"""
    text = unwrap_inline_latex(text)  # ⓪ 行内 $...$ 公式 → 可读文本(NH3/60°C)
    text = _SUBSUP.sub("", text)  # ① 去 sub/sup 标签,保留内容
    text = ftfy.fix_text(
        text, normalization="NFKC"
    )  # ② 修乱码(mojibake)+ 全角→半角(ＰＢＡＴ→PBAT)
    text = _CONTROL.sub("", text)  # ③ 删控制/不可见字符
    text = _SPACES.sub(" ", text)  # ④ 多空格 → 一个
    text = _BLANKS.sub("\n\n", text)  # ⑤ 过多空行 → 两个
    return text.strip()  # ⑥ 去首尾空白


# 需要做文本规范化的"散文类"类型;table(HTML)/formula(LaTeX) 不在其中,避免破坏结构
_PROSE = {"heading", "paragraph", "list", "footnote", "figure"}


def clean_document(doc: Document) -> Document:
    """把 clean_text 施加到每个 prose 类型的 Element.text(就地修改后返回同一个 doc)。

    轮到你:遍历 doc.elements,只对 type 在 _PROSE 里的元素,
    把它的 text 换成 clean_text(它的 text)。table/formula 跳过。
    """
    #  你写:一个 for 循环
    for el in doc.elements:
        if el.type in _PROSE:
            el.text = clean_text(el.text)
    return doc
