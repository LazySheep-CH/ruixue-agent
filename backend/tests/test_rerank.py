"""rerank 的测试。

模型部分不测(那是 sentence-transformers 的事,且慢)。
测的是编排逻辑:Reranker 类怎么处理打分结果,Retriever 怎么用它重排。
用假模型注入,精确控制"模型给了什么分",检查我们的代码怎么反应。
"""

from __future__ import annotations

import pytest

from ruixue_agent.rag.rerank import Reranker
from ruixue_agent.rag.retriever import Retriever


class FakeModel:
    """假 cross-encoder:按一张 query→{文本片段:分} 的表打分。"""

    def __init__(self, table: dict[str, float]):
        self.table = table
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        # 文本里含表中的关键词就给对应分,否则 0
        out = []
        for _q, text in pairs:
            score = max((v for kw, v in self.table.items() if kw in text), default=0.0)
            out.append(score)
        return out


# ────────────── Reranker 本身 ──────────────


def test_rerank_reorders_by_score():
    r = Reranker(model=FakeModel({"降解": 0.9, "施肥": 0.1}))
    out = r.rerank("降解性能", [("a", "尿素施肥方法"), ("b", "PBAT降解特性")])
    assert [cid for cid, _ in out] == ["b", "a"]  # 降解那条被顶到前面
    assert out[0][1] == 0.9


def test_rerank_empty():
    r = Reranker(model=FakeModel({}))
    assert r.rerank("q", []) == []


# ────────────── Retriever 集成 ──────────────


class FakeStore:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=3, year_min=None, source=None):
        return self.hits[:k]


class FakeRow:
    def __init__(self, cid, text, parent_id=None):
        self.chunk_id = cid
        self.text = text
        self.parent_id = parent_id
        self.document_id = "d1"
        self.section_path = ["节"]
        self.page_start = 1
        self.page_end = 1


class FakeRepo:
    def __init__(self, rows):
        self.rows = rows

    def get_chunks(self, ids):
        return [self.rows[i] for i in ids if i in self.rows]

    def get_documents_meta(self, document_ids):
        """检索层要给每条结果标上标题和年份(见 PgRepository.get_documents_meta)。
        假 repo 里返回固定值即可 —— 这里测的是编排,不是元数据本身。"""
        return {d: (f"标题-{d}", 2023) for d in document_ids}


@pytest.fixture
def rows():
    return {
        "P1": FakeRow("P1", "尿素施肥,每亩15公斤"),
        "P1_c": FakeRow("P1_c", "尿素施肥", parent_id="P1"),
        "P2": FakeRow("P2", "PBAT地膜降解特性研究"),
        "P2_c": FakeRow("P2_c", "PBAT降解", parent_id="P2"),
    }


def test_rerank_promotes_relevant_parent(rows):
    """核心:检索把无关的排前面,rerank 应该把相关的顶上来。

    向量把 P1(施肥,无关)排在 P2(降解,相关)前面 —— 这正是 R@1 低的场景。
    rerank 看 query"降解"应该纠正过来。
    """
    # 向量侧:P1_c 分高(0.9),P2_c 分低(0.7) → P1 排前面(错的)
    store = FakeStore([("P1_c", 0.9), ("P2_c", 0.7)])
    model = FakeModel({"降解": 0.95, "施肥": 0.05})
    r = Retriever(store, FakeRepo(rows), reranker=Reranker(model=model))

    got = r.search("PBAT降解性能", k=2)
    assert got[0].chunk_id == "P2"  # rerank 把降解那条顶到第 1
    assert model.calls == 1  # rerank 真的被调了


def test_no_reranker_keeps_retrieval_order(rows):
    """不给 reranker,顺序就是检索的顺序 —— 默认行为不变。"""
    store = FakeStore([("P1_c", 0.9), ("P2_c", 0.7)])
    r = Retriever(store, FakeRepo(rows))  # 没有 reranker
    got = r.search("降解", k=2)
    assert got[0].chunk_id == "P1"  # 保持向量的(错误)顺序,不纠正


def test_rerank_over_fetches_before_truncate(rows):
    """rerank 必须在截断前拿到超取的候选,否则无米下锅。

    要 k=1,但检索超取了 2 个候选。如果先截到 1 再 rerank,P2 就没了。
    正确行为:rerank 看到 2 个,把 P2 顶上来,再截到 1。
    """
    store = FakeStore([("P1_c", 0.9), ("P2_c", 0.7)])
    model = FakeModel({"降解": 0.95, "施肥": 0.05})
    r = Retriever(store, FakeRepo(rows), reranker=Reranker(model=model))
    got = r.search("降解", k=1)
    assert len(got) == 1
    assert got[0].chunk_id == "P2"  # 被 rerank 救回来的
