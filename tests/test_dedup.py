"""dedup 层测试:Jaccard 近似去重(近似的丢、不同的留),阈值尺度无关。"""

from ruixue_agent.ingestion.stages.dedup import dedup_documents, jaccard
from ruixue_agent.ingestion.schema import Document, Element

_A = (
    "全生物降解地膜以PBAT和PLA为主要原料,通过吹膜工艺制备。在新疆棉花种植区开展田间试验,"
    "研究其降解特性对土壤温湿度、棉花出苗率和产量的影响。结果表明覆膜显著提高地温约二到三摄氏度,"
    "生育前期保墒效果明显,棉花产量较露地提高百分之十五,且地膜在收获后逐步降解,减少残膜污染。"
)
_C = "聚乳酸材料在不同土壤环境中的生物降解菌群结构差异显著,采用高通量测序分析细菌群落组成与降解速率。"


def _doc(id_, text):
    return Document(
        document_id=id_,
        filename=f"{id_}.pdf",
        file_type="pdf",
        sha256=id_,
        source="期刊论文",
        parser="mineru",
        elements=[Element(type="paragraph", text=text, page=0)],
    )


def test_near_duplicate_dropped():
    a = _doc("a", _A)
    b = _doc("b", _A.replace("百分之十五", "百分之十六"))  # 改2字:近似重复
    c = _doc("c", _C)  # 完全不同
    kept, dropped = dedup_documents([a, b, c])
    assert {d.document_id for d in kept} == {"a", "c"}
    assert [d.document_id for d in dropped] == ["b"]


def test_jaccard_bounds():
    assert jaccard({"x", "y"}, {"x", "y"}) == 1.0  # 全同
    assert jaccard({"x"}, {"y"}) == 0.0  # 无交集
