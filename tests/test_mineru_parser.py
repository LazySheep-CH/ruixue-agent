"""mineru_parser 映射测试:每种 MinerU 元素 → 正确的 Element(或丢弃)。"""

from ruixue_agent.ingestion.parsers.mineru_parser import parse_content_list

# 一份覆盖所有类型的合成 content_list(含 3 种页面家具)
RAW = [
    {
        "type": "text",
        "text": "1 实验部分",
        "text_level": 2,
        "page_idx": 0,
        "bbox": [0, 0, 1, 1],
    },
    {"type": "text", "text": "采用挤出复合……", "page_idx": 0, "bbox": [0, 0, 1, 1]},
    {"type": "header", "text": "李帅 等", "page_idx": 0, "bbox": [0, 0, 1, 1]},
    {"type": "footer", "text": "中国知网", "page_idx": 0, "bbox": [0, 0, 1, 1]},
    {"type": "page_number", "text": "133", "page_idx": 0, "bbox": [0, 0, 1, 1]},
    {
        "type": "table",
        "table_body": "<table><tr><td>x</td></tr></table>",
        "table_caption": ["表1 配方"],
        "table_footnote": [],
        "img_path": "images/t.jpg",
        "page_idx": 1,
        "bbox": [0, 0, 1, 1],
    },
    {
        "type": "equation",
        "text": r"$$E=mc^2$$",
        "text_format": "latex",
        "img_path": "images/e.jpg",
        "page_idx": 2,
        "bbox": [0, 0, 1, 1],
    },
    {
        "type": "image",
        "img_path": "images/i.jpg",
        "image_caption": ["图1"],
        "image_footnote": [],
        "page_idx": 3,
        "bbox": [0, 0, 1, 1],
    },
    {
        "type": "chart",
        "img_path": "images/c.jpg",
        "chart_caption": ["(a)"],
        "chart_footnote": [],
        "content": "",
        "page_idx": 3,
        "bbox": [0, 0, 1, 1],
    },
    {
        "type": "list",
        "sub_type": "ref_text",
        "list_items": ["[1] A", "[2] B"],
        "page_idx": 6,
        "bbox": [0, 0, 1, 1],
    },
    {
        "type": "page_footnote",
        "text": "收稿日期:2023",
        "page_idx": 0,
        "bbox": [0, 0, 1, 1],
    },
]
META = {
    "document_id": "d1",
    "original_filename": "f.pdf",
    "sha256": "h",
    "source": "期刊论文",
}


def test_furniture_dropped():
    doc = parse_content_list(RAW, META)
    assert len(doc.elements) == 8  # 11 原始 - 3 家具(header/footer/page_number)
    types = {e.type for e in doc.elements}
    assert "header" not in types and "footer" not in types


def test_type_mapping():
    doc = parse_content_list(RAW, META)
    got = [e.type for e in doc.elements]
    assert got == [
        "heading",
        "paragraph",
        "table",
        "formula",
        "figure",
        "figure",
        "list",
        "footnote",
    ]


def test_heading_keeps_level():
    doc = parse_content_list(RAW, META)
    assert doc.elements[0].type == "heading" and doc.elements[0].level == 2


def test_table_and_formula_payload():
    doc = parse_content_list(RAW, META)
    table = next(e for e in doc.elements if e.type == "table")
    assert "<table>" in table.text and table.meta["caption"] == "表1 配方"
    formula = next(e for e in doc.elements if e.type == "formula")
    assert formula.text == r"$$E=mc^2$$" and formula.meta["text_format"] == "latex"


def test_document_metadata():
    doc = parse_content_list(RAW, META)
    assert (
        doc.document_id == "d1" and doc.source == "期刊论文" and doc.parser == "mineru"
    )


def test_empty_shell_table_is_dropped():
    """回归:全量实测 7794 个表格里 251 个(3.2%)缺 table_body —— MinerU 认出表格
    但内容一个字没解析出来(img_path/caption 全空)= 空壳。以前用 raw["table_body"]
    直接取会 KeyError 炸掉整个批处理。"""
    shell = [
        {
            "type": "table",
            "img_path": "",
            "table_caption": [],
            "table_footnote": [],
            "bbox": [0, 0, 1, 1],
            "page_idx": 59,
        }
    ]
    doc = parse_content_list(shell, META)
    assert doc.elements == []  # 空壳丢弃,而且不炸


def test_equation_without_text_is_dropped():
    """回归:实测有 1 个 equation 缺 text。"""
    doc = parse_content_list(
        [
            {
                "type": "equation",
                "img_path": "x.jpg",
                "page_idx": 1,
                "bbox": [0, 0, 1, 1],
            }
        ],
        META,
    )
    assert doc.elements == []


def test_unknown_type_is_counted_not_silently_dropped():
    """未知类型仍然丢弃,但要计数 —— 静默丢弃会掩盖问题
    (全量才发现语料里有 aside_text/code,3篇样本里根本没有)。"""
    from ruixue_agent.ingestion.parsers.mineru_parser import UNKNOWN_TYPES

    UNKNOWN_TYPES.clear()
    doc = parse_content_list(
        [{"type": "aside_text", "text": "边栏", "page_idx": 0}], META
    )
    assert doc.elements == []
    assert UNKNOWN_TYPES["aside_text"] == 1
