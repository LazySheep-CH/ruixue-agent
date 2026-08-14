"""中文分词,供 PostgreSQL 全文检索使用。

PG 的 simple 配置按空格切词,中文文本没有空格,整句会被当成单个 token,
导致任何子串查询都无法命中(连夹在中文里的英文、数字也一并失效)。
因此在写入前用 jieba 分词、以空格连接,存入 chunks.text_tokens,
再由触发器转成 tsvector。

选 jieba 而非 zhparser 扩展:zhparser 需要自行构建 PG 镜像,升级即重做,
且分词逻辑无法单测;jieba 在应用侧,可加领域词典、可测试、可调整。

注意:查询串必须经过同一个 tokenize(),索引和查询的切词方式不一致就对不上。
"""

from __future__ import annotations

import re

import jieba

# 领域词典。jieba 默认会把专业术语切碎(如"氧化生物双降解地膜"切成 6 段),
# 切碎后整词查询无法命中。词典质量直接决定词法检索在专业术语上的效果,
# 需随语料持续维护。
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

# 标准号(GB/T 35795-2017、DB37/T 2446 等)需整体保留:jieba 默认会把
# "DB37/T2446-2013" 切成 4 段,标准号查询直接失效,而这是质检场景的高频查询。
#
# 不能用 \b 定边界:中文字符不属于 word char,"符合GB/T35795" 中"符合"与"GB"
# 之间不存在 \b 边界,匹配会静默失败。改用否定后顾断言,中英文语境均成立。
_STD_CODE = re.compile(
    r"(?<![A-Za-z0-9])(?:GB|DB\d{2}|NY|QB|HG|JB|SN|ISO|ASTM|EN|JIS)"
    r"\s*/?\s*[TZ]?\s*\d[\d.]*(?:\s*-\s*\d{4})?",
    re.IGNORECASE,
)

# LaTeX 公式切词后只剩 "$ \ Delta m" 一类噪音,无检索价值,直接剔除
_LATEX = re.compile(r"\$\$?.*?\$\$?", re.DOTALL)

# 纯标点/空白 token,不入索引
_JUNK = re.compile(r"^[\s\W_]+$")

# Unicode 上标/下标归一化为普通数字。
# 实测:embedding 对两种写法几乎无差别(相似度 0.91~0.98),但词法层
# g·kg-1 与 g·kg⁻¹ 的分词结果不同("kg 1" vs "kg ¹"),互相无法命中。
# 语料中两种形式混存(plain 4893 块 / unicode 5528 块),而用户键盘输入
# 只会产生 plain 形式,故统一归一化为 plain。索引与查询两侧走同一函数,
# 任一写法的查询都能命中任一写法的文档。
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻₀₁₂₃₄₅₆₇₈₉", "0123456789+-0123456789")


def _norm_std_code(s: str) -> str:
    """标准号归一化:去空格、转小写(GB/T 35795-2017 → gb/t35795-2017)。"""
    return re.sub(r"\s+", "", s).lower()


def tokenize(text: str) -> str:
    """切分中文文本,返回空格分隔的词串,供 to_tsvector('simple', ...) 使用。

    预处理顺序:剔除 LaTeX → 上标归一化 → 标准号占位保护 → jieba 分词 → 还原。
    """
    text = _LATEX.sub(" ", text)
    text = text.translate(_SUPERSCRIPT)

    # 标准号先替换为占位符,避免被 jieba 拆碎。
    # 占位符必须是纯字母数字:控制字符(如 \x00)会被 jieba 剥掉,
    # "\x000\x00" 只剩 "0",标准号即丢失。
    codes: list[str] = []

    def _stash(m: re.Match) -> str:
        codes.append(_norm_std_code(m.group()))
        return f" stdcode{len(codes) - 1}x "

    text = _STD_CODE.sub(_stash, text)

    words = [w for w in jieba.lcut(text) if w.strip() and not _JUNK.match(w)]

    out = []
    for w in words:
        m = re.fullmatch(r"stdcode(\d+)x", w, re.IGNORECASE)
        out.append(codes[int(m.group(1))] if m else w.lower())
    return " ".join(out)
