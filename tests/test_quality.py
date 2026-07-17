"""quality 门禁测试:好文档过、空文档拦、正文硬门槛。"""

from ruixue_agent.ingestion.stages.quality import assess_quality
from ruixue_agent.ingestion.schema import Document, Element


def _doc(elements, meta=None):
    d = Document(
        document_id="d",
        filename="f.pdf",
        file_type="pdf",
        sha256="h",
        source="期刊论文",
        parser="mineru",
        elements=elements,
    )
    d.meta.update(meta or {})
    return d


def test_good_doc_passes():
    doc = _doc(
        [Element(type="paragraph", text="正文" * 200, page=0)],
        {"title": "某标题", "year": 2024},
    )
    r = assess_quality(doc)
    assert r["passed"] and r["score"] == 1.0


def test_empty_doc_rejected():
    doc = _doc([Element(type="paragraph", text="", page=0)])
    r = assess_quality(doc)
    assert not r["passed"] and r["checks"]["has_body"] is False


def test_body_is_hard_gate():
    # 有标题有年份,但正文太短 → 仍不过(has_body 一票否决)
    doc = _doc(
        [Element(type="paragraph", text="短", page=0)], {"title": "T", "year": 2024}
    )
    r = assess_quality(doc)
    assert not r["passed"]
