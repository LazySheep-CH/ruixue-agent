"""metadata v2:抽 标题/年份/DOI/关键词/摘要(不再用硬编码材料·作物词典)。"""

from ruixue_agent.ingestion.schema import Document, Element
from ruixue_agent.ingestion.stages.metadata import extract_metadata


def _doc(elements):
    return Document(
        document_id="d",
        filename="f.pdf",
        file_type="pdf",
        sha256="h",
        source="期刊论文",
        parser="mineru",
        elements=elements,
    )


def test_extract_metadata_v2():
    doc = _doc(
        [
            Element(type="heading", text="PBAT地膜在棉花上的应用研究", page=0, level=1),
            Element(
                type="paragraph",
                text="摘 要 研究PBAT可降解地膜在新疆棉区的降解特性",
                page=0,
            ),
            Element(
                type="paragraph", text="关键词 PBAT;可降解地膜;棉花;降解特性", page=0
            ),
            Element(
                type="paragraph", text="文章编号(2024)01 DOI:10.1234/abc.2024", page=0
            ),
        ]
    )
    extract_metadata(doc)
    assert doc.meta["title"] == "PBAT地膜在棉花上的应用研究"
    assert doc.meta["year"] == 2024
    assert doc.meta["doi"] == "10.1234/abc.2024"
    # ★ 关键词是作者亲手声明的主题 —— 比拿词典猜的准
    assert doc.meta["keywords"] == ["PBAT", "可降解地膜", "棉花", "降解特性"]
    assert "新疆棉区" in doc.meta["abstract"]


def test_no_frontmatter_is_safe():
    """新闻/短文没有摘要关键词(实测 15~25% 是这样)→ 给空值,不报错。"""
    doc = _doc([Element(type="paragraph", text="春耕备耕早字当头", page=0)])
    extract_metadata(doc)
    assert doc.meta["title"] == "" and doc.meta["year"] is None
    assert doc.meta["keywords"] == [] and doc.meta["abstract"] == ""
