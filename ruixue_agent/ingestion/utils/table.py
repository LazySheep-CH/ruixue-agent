"""表格 HTML → 行级自然语言(可检索的人话)。

为什么要转:
  ① embedding 看到 <table><tr><td> 是噪声,看不懂表格(和 LaTeX 那个问题一样)
  ② 实测 84% 的表格 HTML 超过 400 字符 → 整块 embedding 会被【悄悄截断】,大半内容检索不到
  ③ 单元格里还混着行内 LaTeX($\\mathrm{P}_{2}0_{5}$ 其实是 P2O5)——
     因为 clean 阶段故意跳过表格(保 HTML 结构),这些一直没被清过

怎么转 —— 行级自然语言,一招解决三个问题:
    表 1 三种试样各组成成分的比较(%)
    <tr><td></td>  <td>纤维素</td><td>半纤维素</td>...     ← 第一行当表头
    <tr><td>全杆</td><td>31.16</td><td>16.4</td>...       ← 数据行
        ↓
    表:三种试样各组成成分的比较(%)。全杆:纤维素 31.16,半纤维素 16.4,木质素 13.15。
        ↑ 每行【自带表名+列名】= 完整独立的一句话
          → 是人话(能 embedding)、能按行切(不怕超长)、每块都有上下文

原始 HTML 仍存在 Element.meta 里(结构不丢)—— 这就是企业级的"表格多形态":
结构留着备用,embedding 用人话。
"""

from __future__ import annotations

from html.parser import HTMLParser

from ruixue_agent.ingestion.utils.latex import unwrap_inline_latex

# 实测:标签只有 table/tr/td(没有 th、没有 colspan/rowspan)→ 标准库 html.parser 足够,
# 不用为这点复杂度装 bs4/lxml。


class _TableParser(HTMLParser):
    """把 <table><tr><td> 拆成 [[行1单元格...], [行2单元格...], ...]。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_table(html: str) -> list[list[str]]:
    """表格 HTML → 二维列表。解析不了就返回空(不能让一个怪表格炸掉整篇)。"""
    try:
        p = _TableParser()
        p.feed(html)
        return [r for r in p.rows if any(c for c in r)]  # 丢掉整行全空的
    except Exception:
        return []


def _clean_cell(text: str) -> str:
    """单元格清洗:行内 LaTeX → 人话($\\mathrm{P}_{2}0_{5}$ → P2O5)。"""
    return unwrap_inline_latex(text).strip()


def table_to_sentences(html: str, caption: str = "") -> list[str]:
    """表格 → 一行一句人话。每句自带表名+列名,完整独立可检索。

    返回空列表 = 这个表格没有可用内容(调用方应丢弃它)。
    """
    rows = parse_table(html)
    if len(rows) < 2:  # 没有表头+至少一行数据 → 没有"行级"可言
        return []

    header = [_clean_cell(c) for c in rows[0]]
    prefix = f"表:{_clean_cell(caption)}。" if caption else "表:"

    out: list[str] = []
    for row in rows[1:]:
        cells = [_clean_cell(c) for c in row]
        label = cells[0] if cells else ""
        # 把"列名 值"配对成人话;跳过空值和第一列(它是行标签)
        pairs = [
            f"{header[i]} {cells[i]}"
            for i in range(1, min(len(header), len(cells)))
            if cells[i] and header[i]
        ]
        if not pairs:
            continue
        sentence = f"{prefix}{label}:" if label else prefix
        out.append(sentence + ",".join(pairs) + "。")
    return out
