"""schema 冒烟测试:构造、校验、序列化往返。"""

import pytest
from pydantic import ValidationError

from ruixue_agent.ingestion.schema import Document, Element, ELEMENT_TYPES


def test_build_a_document():
    title = Element(type="heading", text="1 实验部分", page=0, level=1)
    para = Element(type="paragraph", text="采用挤出复合和吹膜的方法……", page=1)
    doc = Document(
        document_id="doc_0001",
        filename="李帅_PBAT复合薄膜.pdf",
        file_type="pdf",
        sha256="abc123",
        source="期刊论文",
        parser="mineru",
        elements=[title, para],
    )
    # ① "有 2 个元素" → 就是列表长度等于 2
    assert len(doc.elements) == 2
    # ② "第一个是标题" → 下标 [0] 取第一个,看它的 type
    assert doc.elements[0].type == "heading"
    # ③ "标题 level 是 1,正文 level 是 None" → 分别取两个元素的 level
    assert doc.elements[0].level == 1
    assert doc.elements[1].level is None
    # ④ "所有元素 type 都合法" → 遍历每个元素,都得在 ELEMENT_TYPES 里
    assert all(e.type in ELEMENT_TYPES for e in doc.elements)


def test_bad_type_is_rejected():
    """质量门禁:非法 type 在构造时就被拦(Parse, don't validate)。"""
    with pytest.raises(ValidationError):
        Element(type="banana", text="x", page=0)


def test_json_round_trip():
    """落盘读回:写成 JSON 再读回来,嵌套 Element 自动重建、内容不变。"""
    doc = Document(
        document_id="doc_0002",
        filename="f.pdf",
        file_type="pdf",
        sha256="h",
        source="期刊论文",
        parser="mineru",
        elements=[
            Element(type="formula", text=r"E=mc^2", page=3, meta={"latex": r"E=mc^2"})
        ],
    )
    blob = doc.model_dump_json()  # → 存盘时就写这个字符串
    back = Document.model_validate_json(blob)  # ← 读盘时这样还原
    assert back == doc  # 完全相等 = 往返无损
    assert back.elements[0].meta["latex"] == r"E=mc^2"
