"""clean 层测试:prose 被规范化,table/formula 结构不被破坏。"""

from ruixue_agent.ingestion.stages.clean import clean_document, clean_text
from ruixue_agent.ingestion.schema import Document, Element


def test_clean_text_normalizes():
    assert clean_text("ＰＢＡＴ ２０２３") == "PBAT 2023"  # 全角 → 半角
    assert clean_text("压<sub>力</sub>") == "压力"  # 去 sub 标签,词接回


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


def test_prose_cleaned_but_table_formula_untouched():
    table_html = "<table><tr><td>ＰＢＡＴ</td></tr></table>"
    latex = r"$$X_{\mathrm{c}} = 100\%$$"
    doc = _doc(
        [
            Element(type="paragraph", text="复合薄膜ＰＢＡＴ压<sub>力</sub>", page=0),
            Element(type="table", text=table_html, page=1),
            Element(type="formula", text=latex, page=2),
        ]
    )
    clean_document(doc)
    assert doc.elements[0].text == "复合薄膜PBAT压力"  # prose:全角转半角 + 去标签
    assert doc.elements[1].text == table_html  # table(HTML)原封不动
    assert doc.elements[2].text == latex  # formula(LaTeX)原封不动
