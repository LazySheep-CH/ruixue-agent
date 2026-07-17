"""store 测试:父/子分开索引、检索命中子块并顺 parent_id 返回父块。"""

from ruixue_agent.ingestion.schema import Chunk
from ruixue_agent.rag.vector_store import VectorStore


def _chunk(cid, text, parent=None):
    return Chunk(chunk_id=cid, document_id="d", text=text, parent_id=parent)


CHUNKS = [
    _chunk("p0", "整节:PBAT牌号TH801T。PLA牌号4032D。"),  # 父块
    _chunk("p0_c0", "聚己二酸对苯二甲酸丁二酯PBAT牌号TH801T,新疆蓝山屯河", parent="p0"),
    _chunk("p0_c1", "聚乳酸PLA牌号4032D,美国NatureWorks公司", parent="p0"),
    _chunk("p1", "整节:覆膜后地温提高二到三摄氏度,保墒效果明显。"),  # 另一个父块
    _chunk("p1_c0", "覆膜使地温提高二到三摄氏度,生育前期保墒效果明显", parent="p1"),
]


def test_index_separates_parent_and_child():
    s = VectorStore()
    s.index(CHUNKS)
    assert len(s.children) == 3  # 3 个子块进索引
    assert set(s.parents) == {"p0", "p1"}  # 只有父块进字典(不含子块!)
    assert s.vectors.shape[0] == 3  # 向量数 = 子块数


def test_search_hits_right_section_and_parent_carries_the_answer():
    """纯语义检索【不保证】命中"PBAT那条"(PLA那条语义同样像,甚至更短更占优),
    但它能命中【正确的小节】;而 Small-to-Big 返回整节父块,把答案兜回来。
    (要精确匹配关键词,以后加 BM25 混合检索。)
    """
    s = VectorStore()
    s.index(CHUNKS)
    child, parent, score = s.search("PBAT用的什么牌号", k=1)[0]
    assert child.parent_id == "p0"  # 命中了正确的小节
    assert parent is not None and parent.chunk_id == "p0"  # 顺 parent_id 取回父块
    assert "TH801T" in parent.text  # ★ 父块把 PBAT 的答案带回来了
    assert 0 < score <= 1


def test_unrelated_query_ranks_low():
    """不相关的问题,相似度应该明显低。"""
    s = VectorStore()
    s.index(CHUNKS)
    _, _, score = s.search("今天天气怎么样", k=1)[0]
    assert score < 0.4
