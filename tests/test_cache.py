"""落盘测试:存 → 读 → 内容一模一样(往返无损),不存在时返回 None。"""

from ruixue_agent.ingestion.cache import (
    load_chunks,
    load_document,
    save_chunks,
    save_document,
)
from ruixue_agent.ingestion.schema import Chunk, Document, Element


def _doc():
    return Document(
        document_id="b35a6c0e48246b4b",
        filename="李帅_PBAT.pdf",
        file_type="pdf",
        sha256="b35a6c0e48246b4bb581be0259e9b34b",
        source="期刊论文",
        parser="mineru",
        elements=[
            Element(type="heading", text="1 实验部分", page=0, level=2),
            Element(type="formula", text=r"$$E=mc^2$$", page=2, meta={"latex": r"E=mc^2"}),
            Element(
                type="table",
                text="<table><tr><td>PBAT</td></tr></table>",
                page=1,
                meta={"caption": "表1 配方"},
            ),
        ],
        meta={"title": "PBAT复合薄膜", "year": 2025},
    )


def test_document_round_trip(tmp_path):
    doc = _doc()
    path = save_document(doc, folder=tmp_path)
    assert path.name == "b35a6c0e48246b4b.json"  # 文件名 = document_id
    back = load_document(doc.document_id, folder=tmp_path)
    assert back == doc  # 往返无损:嵌套 Element/meta 全还原
    assert back.elements[1].text == r"$$E=mc^2$$"  # 公式的 LaTeX 没被破坏
    assert back.meta["year"] == 2025


def test_load_missing_returns_none(tmp_path):
    assert load_document("不存在的id", folder=tmp_path) is None


def test_chunks_round_trip(tmp_path):
    chunks = [
        Chunk(
            chunk_id="d_s0",
            document_id="d",
            text="整节内容",
            section_path=["论文", "1.1 原料"],
        ),
        Chunk(
            chunk_id="d_s0_c0",
            document_id="d",
            text="PBAT牌号TH801T",
            section_path=["论文", "1.1 原料"],
            parent_id="d_s0",
            meta={"kind": "child"},
        ),
    ]
    save_chunks("d", chunks, folder=tmp_path)
    back = load_chunks("d", folder=tmp_path)
    assert back == chunks
    assert back[1].parent_id == "d_s0"  # 父子关系还在
    assert load_chunks("没有的", folder=tmp_path) is None


def test_creates_folder_if_missing(tmp_path):
    """目录不存在也要能存(mkdir parents=True)。"""
    save_document(_doc(), folder=tmp_path / "深" / "几层" / "都没有")
    assert (tmp_path / "深" / "几层" / "都没有" / "b35a6c0e48246b4b.json").exists()
