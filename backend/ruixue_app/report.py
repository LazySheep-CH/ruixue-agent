"""把一次运行渲染成 PDF 报告。

## 为什么是【接口】而不是【给 agent 的工具】

最直觉的做法是加一个 `generate_pdf_report` 工具让 agent 自己调。那会踩掉
`agents/builder.py` 里立的规矩 ——「工具**全为只读**,故不是出于防破坏的考虑」。
这句话是整个工具安全模型的地基,一旦有工具能写文件,四个新问题同时出现:

    ① 只读不变式没了  →  所有"工具很安全"的推理都要重做
    ② 路径逃逸        →  文件名来自模型,而模型的输入里可能藏注入
    ③ 磁盘配额与清理  →  谁删、何时删、满了怎么办
    ④ 新攻击面        →  提示注入可以诱导 agent 写文件

改成"用户主动点导出、接口负责渲染",这四条同时消失:agent 只负责【想】
(结论已经在 runs 表里),渲染只负责【排版】。**提示注入触发不了它**,
因为它根本不在 agent 的工具集里。

和"不给 agent 写 MCP 工具"、"检索为空时直接短路不调模型"是同一条原则:
**能用受限接口做到的,不要开放通用能力。**

## 两个技术点

**中文字体**:PyMuPDF 内置 `china-s`(简体中文),不必往镜像里塞字体文件。

**字体子集化**:不做的话,两行字的 PDF 是 **1664 KB** —— 整套中文字形被
原样嵌进去。`subset_fonts()` 之后是 **13 KB(0.8%)**,文字仍可选可搜。
这一步不能省:一份报告 1.6MB,用户下载会以为出了什么问题。

**不落盘**:整份 PDF 在内存里生成后直接流式返回。没有临时文件,
自然也没有清理、配额、并发写冲突这些问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

# 版面(单位:pt,A4 = 595 × 842)
_PAGE = (595.0, 842.0)
_MARGIN_X = 56.0
_TOP = 64.0
_BOTTOM = 72.0
_LEADING = 16.0  # 行距

_SIZE_TITLE = 17.0
_SIZE_H = 12.0
_SIZE_BODY = 9.5
_SIZE_META = 8.0

_BRAND = "瑞雪地膜智能助手"
_FOOTER = "本报告由瑞雪地膜智能助手自动生成,结论依据文末出处,请结合实际情况判断。"


@dataclass
class ReportData:
    """渲染一份报告需要的全部输入 —— 刻意和 ORM 解耦,便于测试。"""

    run_id: str
    question: str
    answer: str
    created_at: datetime | None = None
    status: str = "succeeded"


# ── 极简 Markdown 解析 ────────────────────────────────────────
#
# agent 的答案是 Markdown。这里【不】引入 markdown → HTML → PDF 那条链路:
# 那需要 weasyprint 一类的重依赖,在 Windows 上还要装系统库。
# 报告只需要认三种结构(标题 / 列表 / 加粗),够用即可 —— 剩下的按正文走,
# 最坏情况是几个 * 号出现在纸上,不影响可读性。

_H_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LI_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+(.*)$")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_INLINE_JUNK_RE = re.compile(r"`+|~~")
# 表格分隔行 |---|:--:|---| —— 它是 Markdown 的语法记号,印在纸上纯粹是噪声
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")


def _clean(text: str) -> str:
    """去掉 PDF 里没有意义的行内标记。加粗保留文字、丢掉星号。"""
    return _INLINE_JUNK_RE.sub("", _BOLD_RE.sub(r"\1", text)).rstrip()


@dataclass
class _Line:
    text: str
    size: float
    indent: float = 0.0
    gap_before: float = 0.0


@dataclass
class _Table:
    """一张表:第一行当表头,分隔行已剔除。"""

    rows: list[list[str]]


def _split_row(s: str) -> list[str]:
    """把 `| a | b |` 切成 ['a','b']。首尾的空串是竖线两端造成的,要去掉。"""
    cells = [c.strip() for c in s.strip().split("|")]
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return [_clean(c) for c in cells]


def _parse_markdown(md: str) -> list[_Line | _Table]:
    """把 Markdown 拍平成"一块一块怎么画"。

    表格单独成块:`screen_film_recipes` 的输出本来就是配方对比表,
    是报告的主要内容之一。按原文逐行画的话,`|---|---|---|` 这种语法记号
    会原样印在纸上 —— 用户看到的是乱码般的竖线,不是表。
    """
    out: list[_Line | _Table] = []
    pending: list[list[str]] = []  # 正在累积的表格行

    def _flush_table() -> None:
        if pending:
            out.append(_Table(rows=list(pending)))
            pending.clear()

    for raw in (md or "").splitlines():
        s = raw.rstrip()
        if s.lstrip().startswith("|"):
            if not _TABLE_SEP_RE.match(s):  # 分隔行直接扔掉
                pending.append(_split_row(s))
            continue
        _flush_table()
        if not s.strip():
            # 连续空行只留一个,否则模型爱空行会把报告撑得很散
            if out and not (isinstance(out[-1], _Line) and not out[-1].text):
                out.append(_Line("", _SIZE_BODY))
            continue
        if m := _H_RE.match(s):
            out.append(_Line(_clean(m.group(2)), _SIZE_H, gap_before=6.0))
            continue
        if m := _LI_RE.match(s):
            out.append(_Line("• " + _clean(m.group(2)), _SIZE_BODY, indent=12.0))
            continue
        out.append(_Line(_clean(s), _SIZE_BODY))
    _flush_table()
    return out


def _wrap(font, text: str, size: float, width: float) -> list[str]:
    """按可用宽度折行。

    **必须按【字符】折,不能按空格折** —— 中文一整段里没有空格,
    按空格折等于不折,文字会直接画出纸外(而且不会报错,只是看不见)。
    """
    if not text:
        return [""]
    lines, cur = [], ""
    for ch in text:
        if font.text_length(cur + ch, fontsize=size) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def render_pdf(data: ReportData) -> bytes:
    """渲染并返回 PDF 字节。全程在内存里,不落盘。"""
    import fitz

    doc = fitz.open()
    font = fitz.Font("china-s")  # PyMuPDF 内置简体中文,无需外部字体文件
    usable = _PAGE[0] - 2 * _MARGIN_X

    page = doc.new_page(width=_PAGE[0], height=_PAGE[1])
    writer = fitz.TextWriter(page.rect)
    y = _TOP

    def _flush_page():
        nonlocal page, writer, y
        writer.write_text(page)
        page = doc.new_page(width=_PAGE[0], height=_PAGE[1])
        writer = fitz.TextWriter(page.rect)
        y = _TOP

    def put(text: str, size: float, indent: float = 0.0) -> None:
        nonlocal y
        for ln in _wrap(font, text, size, usable - indent):
            if y > _PAGE[1] - _BOTTOM:
                _flush_page()
            if ln:
                writer.append((_MARGIN_X + indent, y), ln, font=font, fontsize=size)
            y += _LEADING if size <= _SIZE_BODY else _LEADING + 4

    # ── 抬头 ──
    put(f"{_BRAND} · 咨询报告", _SIZE_TITLE)
    when = (data.created_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    put(f"生成时间 {when}    运行编号 {data.run_id}", _SIZE_META)
    y += 8

    # ── 提问 ──
    put("咨询内容", _SIZE_H)
    put(data.question or "(无)", _SIZE_BODY, indent=8.0)
    y += 8

    def put_table(tbl: _Table) -> None:
        """画一张表:按内容算列宽,表头下加一条横线。

        列宽按【各列最长单元格】的自然宽度分配,总宽超出可用宽度时按比例压缩,
        放不下的单元格截断加省略号。为什么不折行:折行要算行高、要处理跨页,
        复杂度陡增;而报告里的表格单元格基本是"29.3%""19.7 MPa"这类短值,
        截断几乎不会发生。**先把 95% 的情况画对,不为 5% 引入一倍复杂度。**
        """
        nonlocal y
        if not tbl.rows:
            return
        ncol = max(len(r) for r in tbl.rows)
        rows = [r + [""] * (ncol - len(r)) for r in tbl.rows]
        pad = 8.0
        # 各列自然宽度 = 该列最宽单元格 + 间距
        nat = [
            max(font.text_length(r[c], fontsize=_SIZE_BODY) for r in rows) + pad
            for c in range(ncol)
        ]
        avail = usable - 8.0
        total = sum(nat)
        widths = [w * avail / total for w in nat] if total > avail else nat

        for i, row in enumerate(rows):
            if y > _PAGE[1] - _BOTTOM:
                _flush_page()
            x = _MARGIN_X + 8.0
            for c, cell in enumerate(row):
                txt = cell
                # 截断:留出省略号的位置,避免画出格外
                while txt and font.text_length(txt, fontsize=_SIZE_BODY) > widths[c] - pad:
                    txt = txt[:-1]
                if txt != cell and txt:
                    txt = txt[:-1] + "…"
                if txt:
                    writer.append((x, y), txt, font=font, fontsize=_SIZE_BODY)
                x += widths[c]
            y += _LEADING
            if i == 0:
                # 表头下画一条细线 —— 只画这一条,满格线框在纯文字报告里更吵
                page.draw_line(
                    fitz.Point(_MARGIN_X + 8.0, y - _LEADING + 4),
                    fitz.Point(_MARGIN_X + 8.0 + sum(widths), y - _LEADING + 4),
                    color=(0.6, 0.6, 0.6),
                    width=0.5,
                )
        y += 4

    # ── 结论 ──
    put("分析与建议", _SIZE_H)
    if data.status != "succeeded" or not (data.answer or "").strip():
        # 失败的运行也允许导出:用户手里有 run_id,给一份说明比 404 更有用
        put("本次运行未能完成,没有可导出的结论。", _SIZE_BODY, indent=8.0)
    else:
        for block in _parse_markdown(data.answer):
            if isinstance(block, _Table):
                put_table(block)
            else:
                y += block.gap_before
                put(block.text, block.size, indent=block.indent)

    y += 12
    put(_FOOTER, _SIZE_META)
    writer.write_text(page)

    # 子集化:1664 KB → 13 KB。不做的话整套中文字形都会被嵌进去。
    doc.subset_fonts()
    buf = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return buf


def filename_for(data: ReportData) -> str:
    """下载文件名。只用日期和 run_id —— **不拼用户的提问**。

    提问是用户输入,拼进 Content-Disposition 就是把不可信字符串放进 HTTP 头:
    换行会造成响应头注入,中文和引号还要额外编码。文件名不值得冒这个险。
    """
    day = (data.created_at or datetime.now()).strftime("%Y%m%d")
    return f"ruixue-report-{day}-{data.run_id[:8]}.pdf"
