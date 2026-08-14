"""metadata:从 Document 内容里抽【内容元数据】,填进 doc.meta。

抽什么(v2 —— 砍掉了 v1 的硬编码材料/作物词典):
    title      第一个 level-1 heading
    year       卷首四位年份取最大(通常是出版年)
    doi        正则
    keywords   作者声明的"关键词"(实测覆盖 74.4%)
    abstract   文档自述的摘要(实测覆盖 79.7%)

为什么砍掉 materials/crops 词典:
  1) 硬编码 brittle:新材料/别名抓不到,词典永远补不完
  2) 词典分不清"提到"和"是主题"——一篇玉米论文顺口提了棉花,就被打上棉花标签,
     拿去过滤"棉花的论文"就会误伤
  3) 关键词是【作者亲手声明的主题】:免费、权威、不用猜 —— 直接用它

用途:检索按年份/来源硬过滤 + 引用溯源 + 权威排序。
     abstract 还有个用处:检索命中的是碎片,把 title+abstract 当"文档名片"一起给大模型,
     它才知道"这段来自一篇讲什么的论文"。

区别于 parser 记的来源血缘(document_id/sha256/source/parser = 信封):
这里是【分析文档内容】挖出来的,不是文件属性。
"""

from __future__ import annotations

import re

from ruixue_agent.ingestion.schema import Document
from ruixue_agent.ingestion.utils.frontmatter import extract_abstract, extract_keywords

_YEAR = re.compile(r"(?:19|20)\d{2}")  # 1900~2099 的四位年
_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")  # DOI 标准格式


def _title(doc: Document) -> str:
    """标题 = 第一个 level-1 的 heading。"""
    for e in doc.elements:
        if e.type == "heading" and e.level == 1:
            return e.text
    return ""


def extract_metadata(doc: Document) -> Document:
    """分析 doc 内容,把元数据填进 doc.meta(就地),返回同一个 doc。"""
    texts = [e.text for e in doc.elements if e.type in ("heading", "paragraph")]
    front = "\n".join(texts[:15])  # 卷首:标题/作者/摘要/关键词
    full = "\n".join(e.text for e in doc.elements)

    years = [int(y) for y in _YEAR.findall(front) if 1990 <= int(y) <= 2026]
    doi = _DOI.search(full)

    doc.meta.update(
        {
            "title": _title(doc),
            "year": max(years) if years else None,  # 卷首多个年份取最大(通常是出版年)
            "doi": doi.group(0) if doi else None,
            "keywords": extract_keywords(texts),
            "abstract": extract_abstract(texts),
        }
    )
    return doc
