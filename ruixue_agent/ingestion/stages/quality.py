"""质量门禁:入库前给每篇 Document 打分,不合格的不许入库。

与 schema 的 @field_validator 同一思路,把关粒度从字段放大到整篇文档。
不合格文档进入 data/failed/ 供人工复核,而不是静默混入知识库。
"""

from __future__ import annotations

from ruixue_agent.ingestion.schema import Document

_MIN_BODY = 200  # 正文少于这么多字,基本是解析失败
_MAX_GARBLE = 0.02  # 乱码率上限


def _body_len(doc: Document) -> int:
    """正文总字数(标题/段落/列表)。"""
    return sum(
        len(e.text) for e in doc.elements if e.type in ("heading", "paragraph", "list")
    )


def _garble_ratio(doc: Document) -> float:
    """乱码率 = 替换符 '�' 占全文比例。"""
    text = "".join(e.text for e in doc.elements)
    if not text:
        return 1.0
    return sum(1 for c in text if c == "�") / len(text)


def assess_quality(doc: Document) -> dict:
    """体检 doc,返回 {score, checks, passed, body_len}。"""
    body = _body_len(doc)
    checks = {
        "has_body": body >= _MIN_BODY,  # 有正文(硬门槛)
        "has_title": bool(doc.meta.get("title")),  # 标题抽到了
        "has_year": doc.meta.get("year") is not None,  # 年份抽到了
        "low_garble": _garble_ratio(doc) < _MAX_GARBLE,  # 乱码少
    }
    score = round(sum(checks.values()) / len(checks), 2)
    passed = checks["has_body"] and score >= 0.5  # 没正文一票否决 + 总分达标
    return {"score": score, "checks": checks, "passed": passed, "body_len": body}
