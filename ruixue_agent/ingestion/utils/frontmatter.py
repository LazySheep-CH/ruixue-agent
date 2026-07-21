"""卷首信息提取:从论文开头把 摘要 / 关键词 挖出来当元数据。

为什么单独一层:卷首这些东西(作者/单位/摘要/关键词/文章编号)不是"知识正文",
但摘要和关键词是【作者亲手写的主题声明】—— 比我们用词典去猜主题准得多。
(实测:84.5% 的论文有"摘要"、81.2% 有"关键词"。)

真实写法很野,四种情况都要接住:
  A  "摘 要 为明确不同种类地膜…"        标记和内容在同一元素,没冒号
  B  "摘要"                            标记独占一个元素,内容在下一个
  C  "(1甘肃农大…)摘 要 为探明玉…"     标记挤在单位后面,不在开头
  D  "［摘 要］:" / "关键词<sub>:</sub>" 各种全角括号/冒号/sub标签
"""

from __future__ import annotations

import re

# 标记:允许"摘要"中间有任意空白(含全角空格  ),前后可有各种括号
ABSTRACT_MARK = re.compile(r"[\[【［]?\s*摘\s*要\s*[\]】］]?")
KEYWORD_MARK = re.compile(r"[\[【［]?\s*关\s*键\s*词\s*[\]】］]?")
# 摘要的"下界":遇到这些说明摘要结束了
END_MARKS = re.compile(
    r"关\s*键\s*词|中图分类号|文献标[识志]码|文章编号|^\s*Abstract|^\s*Key\s*words",
    re.I,
)

# 标记后面可能跟的垃圾:冒号、方括号、sub/sup标签、空白
_LEAD_JUNK = re.compile(r"^(?:\s|[:：\]】］]|</?su[bp]>)+")

_KW_SPLIT = re.compile(r"[;；,，、]+|\s{2,}")  # 关键词的分隔符(实测五花八门)

FRONT_SCAN = 25  # 只在前 25 个元素里找(卷首不会更靠后)


def _after_mark(text: str, mark: re.Pattern) -> str | None:
    """在一个元素里找标记,返回标记【后面】的内容。

    返回值三态,调用方据此分派:
      None  → 这个元素里【没有】标记        → 调用方应继续看下一个元素
      ""    → 有标记但后面是空的(情况B)    → 调用方应取【下一个】元素当内容
      "内容" → 有标记且后面就是内容(情况A/C)
    """
    m = mark.search(text)  # search 不是 match:标记可能在中间(情况C)
    if not m:
        return None
    rest = text[m.end() :]  # 标记结束的位置之后
    return _LEAD_JUNK.sub("", rest).strip()  # 剥掉开头的 :：]】<sub> 等垃圾


def extract_abstract(texts: list[str]) -> str:
    """提取摘要:找到标记后,一直取到下一个标记(关键词/中图分类号/Abstract)为止。"""
    for i, t in enumerate(texts[:FRONT_SCAN]):
        got = _after_mark(t, ABSTRACT_MARK)
        if got is None:
            continue  # 没标记,看下一个元素
        parts = [got] if got else []  # 情况A/C:标记后就有内容
        # 继续往后收,直到撞上结束标记(摘要可能分好几个元素)
        for nxt in texts[i + 1 : i + 6]:
            if END_MARKS.search(nxt):
                break
            parts.append(nxt.strip())
            if sum(len(p) for p in parts) > 1200:  # 摘要不会更长了,防跑飞
                break
        text = " ".join(p for p in parts if p)
        return re.sub(r"</?su[bp]>", "", text).strip()
    return ""


def extract_keywords(texts: list[str]) -> list[str]:
    """提取关键词并切成列表(和 extract_abstract 同构,最后多一步切分)。"""
    for i, t in enumerate(texts[:FRONT_SCAN]):
        got = _after_mark(t, KEYWORD_MARK)
        if got is None:
            continue
        if not got and i + 1 < len(texts):  # 情况B:内容在下一个元素
            got = texts[i + 1]
        got = re.sub(r"</?su[bp]>", "", got)  # 剥掉 sub/sup 标签
        # 关键词行后面常紧跟"中图分类号:U495 文献标志码:A" → 在那儿截断,别吃进来
        cut = END_MARKS.search(got)
        if cut:
            got = got[: cut.start()]
        parts = [p.strip() for p in _KW_SPLIT.split(got) if p.strip()]
        return [p for p in parts if 1 < len(p) < 30][:12]  # 过滤掉太短/太长的怪东西
    return []
