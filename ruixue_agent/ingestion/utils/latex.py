"""行内 LaTeX → 可读文本。

背景:MinerU 把化学式/单位/变量名当"公式"识别,渲染成 $\\mathrm{NH}_{3}$ 这种混进正文,
      对 embedding 是噪声(模型看到的是 "mathrm N H 3" 而不是 "NH3")。
      实测:14.8% 的正文段含行内公式,150篇里 6028 个。

策略【成熟库做重活 + 薄适配层收尾】:
  · pylatexenc 负责真正解析 LaTeX 语法树、剥样式壳(\\mathrm/\\mathbf/\\it)、转符号(μ × · ∼)
  · 我们只收它留下的尾巴:_ ^ 下标上标、∘→°、\\xa0、多余空格
  不手搓正则(库比我们写得好),也不指望库完美(它不知道我们要纯文本)。
"""

from __future__ import annotations

import re

from pylatexenc.latex2text import LatexNodes2Text

_conv = LatexNodes2Text()

# MinerU 用 $...$ 包行内公式。上限 600 是量出来的:实测 150 篇 3728 个公式,
# 中位数 37 字符、99分位 159、最长 575 → 600 覆盖 100%。
# (留上限当护栏:万一某段有落单的 $,不至于让正则抓一大片。)
_INLINE = re.compile(r"\$([^$]{1,600})\$")

# pylatexenc 把 \circ 转成 ∘(U+2218 环运算符),但在 60^∘C 这种语境里它其实是度数符 °
_RING_TO_DEGREE = re.compile(r"\^?\s*∘")

# 上标转 Unicode 用的对照表(superscript="unicode" 时才用)
_SUP = str.maketrans("0123456789+-()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ⁿ")
_SUB = str.maketrans("0123456789+-()", "₀₁₂₃₄₅₆₇₈₉₊₋₍₎")


def _strip_marks(s: str, mode: str) -> str:
    """处理 pylatexenc 留下的 _ 和 ^ 标记。"""
    if mode == "unicode":
        # 注意:pylatexenc 逐字符输出,标记后面是 "^ - 1" 这样带空格的
        #      → 捕获组必须允许空格,否则只匹配到第一个字符
        def sup(m):
            return m.group(1).replace(" ", "").translate(_SUP)

        def sub(m):
            return m.group(1).replace(" ", "").translate(_SUB)

        s = re.sub(r"\^\s*([0-9+\-()n\s]+?)(?=[^0-9+\-()n\s]|$)", sup, s)
        s = re.sub(r"_\s*([0-9+\-()\s]+?)(?=[^0-9+\-()\s]|$)", sub, s)
    # 剩下的(含 plain 模式)直接去掉标记,保留字符:NH_3 → NH3
    return s.replace("^", "").replace("_", "")


def latex_to_text(latex: str, superscript: str = "plain") -> str:
    """一段 LaTeX → 可读文本。

    superscript: "plain"   → g·kg-1 / NH3 / 60°C   (纯文本,字符常见)
                 "unicode" → g·kg⁻¹ / NH₃ / 60°C  (好看,但字符生僻)

    默认 plain,依据实测:
      · embedding 层对两种形式几乎无差别(相似度 0.91~0.98,跨形式也能搜到)
      · BM25 层两种形式分词结果不同(kg-1→"kg 1" vs kg⁻¹→"kg ¹"),必须统一
      · 用户键盘只打得出 plain,所以统一成 plain
    检索侧的兜底在 rag/text_segment.py:分词时会把残留的 unicode 上标归一化成 plain。
    """
    try:
        s = _conv.latex_to_text(latex)  # ← 重活:库来干
    except Exception:
        return latex  # 转不了就原样返回,不能让一个怪公式炸掉整篇
    s = _RING_TO_DEGREE.sub("°", s)  # ∘(环运算符) → °(度数)
    s = s.replace("∼", "~")  # ∼ → ~(中文语料里表示范围,如 0~80)
    s = _strip_marks(s, superscript)
    # 关键:识别器【逐字符输出】,普通空格全是垃圾("N H"其实是"NH")→ 全删;
    #       而 \xa0 来自原文的 ~,那是【真空格】→ 删完再变回来。顺序不能反。
    s = re.sub(r"[ \t]+", "", s)
    s = s.replace("\xa0", " ")
    return s.strip()


def unwrap_inline_latex(text: str, superscript: str = "plain") -> str:
    """把一段正文里所有 $...$ 行内公式换成可读文本。"""
    return _INLINE.sub(lambda m: latex_to_text(m.group(1), superscript), text)
