"""中文分词 —— 给 PostgreSQL 全文检索(BM25)喂词。

═══ 为什么需要这一层 ═══

PG 的 'simple' 配置靠【空格】切词。中文没空格 → 整句被当成一个 token:
    to_tsvector('simple', '地膜厚度应不小于0.010mm')
      → '地膜厚度应不小于0.010mm':1        ← 一整坨
    搜「地膜厚度」 false / 搜「厚度」 false / 搜「PBAT」 false / 搜「0.010」 false
    —— 实测【全线失败】,连夹在中文里的英文和数字都搜不到(它们跟中文粘在一起了)。

所以要在【存进去之前】先把词切开、用空格连起来:
    '地膜 厚度 应 不 小于 0.010 mm'
      → '地膜':1 '厚度':2 '小于':5 '0.010':6 'mm':7
    四项全中。

═══ 为什么用 jieba 而不是装 zhparser 扩展 ═══
    zhparser 要自己 build Docker 镜像,以后每次升级 PG 都得重来,而且分词逻辑
    埋在 PG 扩展里是个黑盒 —— 改不了、测不了。
    jieba 在 Python 侧:纯 Python 无依赖、能加自定义词典、能写单元测试。
"""

from __future__ import annotations

import re

import jieba

# ── 领域词典 ──────────────────────────────────────────────
# 实测 jieba 默认会把这些切碎:
#     "氧化-生物双降解地膜" → 氧化 / - / 生物 / 双 / 降解 / 地膜
# 切碎了搜整词就搜不到。加进词典后它们是【一个词】。
#
# 这份词典该由懂地膜的人持续补 —— 它直接决定 BM25 在专业术语上的成败。
_TERMS = [
    # 材料
    "PBAT",
    "PLA",
    "PBS",
    "PHA",
    "PCL",
    "PPC",
    "PBSA",
    "聚乳酸",
    "聚己二酸对苯二甲酸丁二醇酯",
    "聚丁二酸丁二醇酯",
    "生物降解地膜",
    "全生物降解地膜",
    "氧化生物双降解地膜",
    "可降解地膜",
    "液态地膜",
    "淀粉基地膜",
    "纸质地膜",
    "光降解地膜",
    # 性能指标
    "断裂标称应变",
    "断裂伸长率",
    "拉伸负荷",
    "拉伸强度",
    "直角撕裂负荷",
    "熔融指数",
    "水蒸气透过率",
    "诱导期",
    "功能期",
    "崩解率",
    "降解率",
    "生物分解率",
    "质量损失率",
    # 农艺
    "覆膜",
    "揭膜",
    "残膜",
    "地膜残留",
    "白色污染",
    "微塑料",
    "土壤含水率",
    "地温",
    "出苗率",
    "保墒",
    "垄作",
    "膜下滴灌",
]
for _t in _TERMS:
    jieba.add_word(_t)

# 标准号:GB/T 35795-2017、DB37/T 2446-2013、NY/T 1227 …
# jieba 会把 "DB37/T2446-2" 切成 DB37 / / / T2446 / - / 2 —— 彻底废掉。
# 而"这膜符合 GB/T 35795 吗"是质检采购最高频的查询之一,必须整体保留。
#
# ⚠ 不能用 \b(单词边界)—— 中文字符不算 word char,所以 "符合GB/T35795" 里
#   "符合" 和 "GB" 之间【没有边界】,\b 匹配失败,标准号被漏掉。
#   这是抄英文语料的正则模式在中文里静默失效的典型(踩过,实测才发现)。
#   改用"前面不能紧跟字母数字"的否定后顾断言,中英文都成立。
_STD_CODE = re.compile(
    r"(?<![A-Za-z0-9])(?:GB|DB\d{2}|NY|QB|HG|JB|SN|ISO|ASTM|EN|JIS)"
    r"\s*/?\s*[TZ]?\s*\d[\d.]*(?:\s*-\s*\d{4})?",
    re.IGNORECASE,
)

# LaTeX 公式:切出来全是 $ \ Delta m 这种噪音,对检索毫无价值,直接扔
_LATEX = re.compile(r"\$\$?.*?\$\$?", re.DOTALL)

# 纯标点/空白 token —— 索引里留着只占地方
_JUNK = re.compile(r"^[\s\W_]+$")

# Unicode 上标/下标 → 普通数字。归一化,不是美化。
#
# 实测决定的(不是拍脑袋):
#   embedding 层:g·kg-1 和 g·kg⁻¹ 相似度 0.91~0.98,跨形式也能搜到(0.779)—— 无所谓
#   BM25   层:g·kg-1 切成 "kg 1",g·kg⁻¹ 切成 "kg ¹" —— 完全不同,互相搜不到
# 而语料里两种混着存(plain 4893 块 + unicode 5528 块),用户键盘只打得出 plain,
# 所以统一成 plain:库里的 unicode 上标要在分词时抹平,否则用户永远搜不到它们。
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻₀₁₂₃₄₅₆₇₈₉", "0123456789+-0123456789")


def _norm_std_code(s: str) -> str:
    """GB/T 35795-2017 → gb/t35795-2017,统一形态,搜的时候才对得上。"""
    return re.sub(r"\s+", "", s).lower()


def tokenize(text: str) -> str:
    """把中文文本切成【空格分隔】的词串,喂给 to_tsvector('simple', ...)。

    三步预处理都是实测踩出来的:
      ① 扔掉 LaTeX  —— 否则索引里全是 $ \\ Delta m 这种噪音
      ② 标准号整体保护 —— 否则 "DB37/T2446-2" 被切成 4 段,搜标准号直接废
      ③ jieba 切 + 领域词典 —— 否则 "氧化生物双降解地膜" 被拆成 6 段
    """
    text = _LATEX.sub(" ", text)
    text = text.translate(_SUPERSCRIPT)  # unicode 上标 → plain,和用户键盘输入对齐

    # 标准号先抠出来占位,免得被 jieba 拆碎。
    #
    # ⚠ 占位符不能用 \x00 这类控制字符 —— jieba 会把它们剥掉,
    #   "\x000\x00" 只剩下 "0",标准号就丢了(踩过,测试红了才发现)。
    #   用纯字母数字的形式,jieba 会当成一个普通英文词整体保留。
    codes: list[str] = []

    def _stash(m: re.Match) -> str:
        codes.append(_norm_std_code(m.group()))
        return f" stdcode{len(codes) - 1}x "

    text = _STD_CODE.sub(_stash, text)

    words = [w for w in jieba.lcut(text) if w.strip() and not _JUNK.match(w)]

    # 占位符还原
    out = []
    for w in words:
        m = re.fullmatch(r"stdcode(\d+)x", w, re.IGNORECASE)
        out.append(codes[int(m.group(1))] if m else w.lower())
    return " ".join(out)
