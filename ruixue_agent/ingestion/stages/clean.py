"""清洗阶段:对 Document 的文本做规范化。作用在 IR 上,所有 parser 共享。

分工:
- clean_text(str):纯字符串规范化(行内公式展开/去标签/修乱码/统一空白)
- clean_document(doc):把 clean_text 应用到 Document,但只处理散文类元素,
  跳过 table(HTML)与 formula(LaTeX)—— 文本规范化会破坏结构化内容。
"""

from __future__ import annotations

import re

import ftfy

from ruixue_agent.ingestion.schema import Document
from ruixue_agent.ingestion.utils.latex import unwrap_inline_latex

# <sub>/<sup> 标签去掉、保留内容(压<sub>力</sub> → 压力,拼回被拆开的词)
_SUBSUP = re.compile(r"</?su[bp]>")
# 控制/不可见字符(保留 \t \n)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")  # 连续空格/制表符 → 一个
_BLANKS = re.compile(r"\n{3,}")  # 3+ 连续空行 → 2


def clean_text(text: str) -> str:
    """字符串级规范化,步骤见行内注释。"""
    text = unwrap_inline_latex(text)  # 行内 $...$ 公式 → 可读文本(NH3/60°C)
    text = _SUBSUP.sub("", text)  # 去 sub/sup 标签,保留内容
    text = ftfy.fix_text(
        text, normalization="NFKC"
    )  # 修乱码 + 全角转半角(ＰＢＡＴ→PBAT)
    text = _CONTROL.sub("", text)  # 删控制字符
    text = _SPACES.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


# 散文类元素类型;table/formula 不做规范化,避免破坏 HTML/LaTeX 结构
_PROSE = {"heading", "paragraph", "list", "footnote", "figure"}


def clean_document(doc: Document) -> Document:
    """对 doc 中散文类元素的 text 就地规范化,返回同一对象。"""
    for el in doc.elements:
        if el.type in _PROSE:
            el.text = clean_text(el.text)
    return doc
