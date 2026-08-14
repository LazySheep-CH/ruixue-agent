"""Retriever 的规格说明书 —— 组合 Milvus(给 ID)+ PG(给文本)。

这些测试用【假的】Milvus 和 PG,不连真库。为什么:
    Retriever 自己不查任何东西,它只【编排】。要测的是编排逻辑对不对,
    不是 Milvus 能不能搜。塞假的进去,就能精确控制"Milvus 返回了什么",
    然后检查 Retriever 怎么处理 —— 包括真库里很难造出来的情况
    (比如"3个子块全来自同一个父块")。

    这正是 PgRepository 把 session 从外部传入的回报:能塞假的。
    如果 Retriever 自己 new 一个 MilvusVectorStore,这些测试根本没法写。
"""

from __future__ import annotations

import pytest

from ruixue_agent.rag.retriever import Retriever


class FakeStore:
    """假 Milvus:你让它返回什么,它就返回什么。"""

    def __init__(self, hits: list[tuple[str, float]]):
        self.hits = hits
        self.last_call: dict = {}

    def search(self, query, k=3, year_min=None, source=None):
        self.last_call = {
            "query": query,
            "k": k,
            "year_min": year_min,
            "source": source,
        }
        return self.hits[:k]


class FakeRow:
    """假的 ChunkRow —— 只带 Retriever 会用到的字段。"""

    def __init__(self, chunk_id, text, parent_id=None, section_path=None, document_id="d1"):
        self.chunk_id = chunk_id
        self.text = text
        self.parent_id = parent_id
        self.section_path = section_path or ["实验部分", "1.1 主要原料"]
        self.document_id = document_id
        self.page_start = 1
        self.page_end = 1


class FakeRepo:
    """假 PG:按 id 还行,顺序按要求来(真 PgRepository 的契约就是这样)。"""

    def __init__(self, rows: dict[str, FakeRow]):
        self.rows = rows

    def get_chunks(self, chunk_ids):
        return [self.rows[i] for i in chunk_ids if i in self.rows]

    def get_documents_meta(self, document_ids):
        """检索层要给每条结果标上标题和年份(见 PgRepository.get_documents_meta)。
        假 repo 里返回固定值即可 —— 这里测的是编排,不是元数据本身。"""
        return {d: (f"标题-{d}", 2023) for d in document_ids}

    def get_parents(self, child_ids):
        children = self.get_chunks(child_ids)
        pids = list(dict.fromkeys(c.parent_id for c in children if c.parent_id))
        return self.get_chunks(pids)


@pytest.fixture
def rows():
    return {
        # 父块 P1 有两个子块;父块 P2 有一个
        "P1": FakeRow("P1", "1.1 主要原料。PBAT 牌号 TH801T,蓝山屯河。密度 1.26。"),
        "P1_c0": FakeRow("P1_c0", "PBAT 牌号 TH801T,蓝山屯河。", parent_id="P1"),
        "P1_c1": FakeRow("P1_c1", "密度 1.26。", parent_id="P1"),
        "P2": FakeRow("P2", "2.1 测试方法。按 GB/T 1040 测拉伸强度。", document_id="d2"),
        "P2_c0": FakeRow("P2_c0", "按 GB/T 1040 测拉伸强度。", parent_id="P2", document_id="d2"),
    }


# ────────────── 1. 基本:子块命中 → 还父块 ──────────────


def test_returns_parent_text_not_child_text(rows):
    """Small-to-Big 的全部意义:子块负责【被搜到】,父块负责【被返回】。

    给 LLM 的必须是父块 —— 子块太碎,LLM 看不出上下文。
    """
    store = FakeStore([("P1_c0", 0.82)])
    r = Retriever(store, FakeRepo(rows))
    got = r.search("PBAT 牌号", k=1)

    assert len(got) == 1
    assert got[0].chunk_id == "P1"  # 父块
    assert "密度 1.26" in got[0].text  # 父块含子块没有的上下文 ← 这就是 Small-to-Big 的收益


def test_dedups_parents(rows):
    """决策1):3 个子块可能都来自【同一个父块】。

    同一节里三句话都跟问题相关 —— 这恰恰说明那一节是对的。
    但不能把同一段父块文本重复三遍塞给 LLM:白烧 token,
    还可能让模型觉得"这事说了三遍,一定很重要"。
    """
    store = FakeStore([("P1_c0", 0.82), ("P1_c1", 0.71), ("P2_c0", 0.55)])
    r = Retriever(store, FakeRepo(rows))
    got = r.search("PBAT", k=3)

    assert [g.chunk_id for g in got] == ["P1", "P2"]  # 2 个,不是 3 个


def test_dedup_preserves_similarity_order(rows):
    """去重要【保序】—— 最相关的父块必须排最前。

    坑:用 set() 去重会打乱顺序(Python 的 set 不保证顺序)。
    dict.fromkeys() 才对:去重且保留首次出现的顺序。
    """
    store = FakeStore([("P2_c0", 0.90), ("P1_c0", 0.80), ("P1_c1", 0.70)])
    r = Retriever(store, FakeRepo(rows))
    got = r.search("随便", k=3)

    assert [g.chunk_id for g in got] == ["P2", "P1"]  # P2 分高,排前面


# ────────────── 2. 决策1):k 不够怎么办 ──────────────


def test_overfetches_children_to_fill_k_parents(rows):
    """决策1)的答案:用户要 k 个【父块】,就得多搜几个子块。

    只搜 3 个子块 → 可能全来自 1 个父块 → 只还 1 个,用户要 3 个。
    所以要【超取】:搜 k * fanout 个子块,去重后取前 k 个父块。

    这个测试盯的是:Retriever 真的向 store 要了【更多】子块。
    """
    store = FakeStore([("P1_c0", 0.82), ("P1_c1", 0.71), ("P2_c0", 0.55)])
    r = Retriever(store, FakeRepo(rows))
    r.search("PBAT", k=2)

    assert store.last_call["k"] > 2  # 要 2 个父块,向 Milvus 要了多于 2 个子块


def test_returns_at_most_k_parents(rows):
    """超取归超取,还给用户的不能超过 k 个。"""
    store = FakeStore([("P2_c0", 0.90), ("P1_c0", 0.80)])
    r = Retriever(store, FakeRepo(rows))
    got = r.search("随便", k=1)

    assert len(got) == 1


# ────────────── 3. 决策2):父块的分数 ──────────────


def test_parent_score_is_best_child_score(rows):
    """决策2):父块自己没有分数 —— Milvus 给的是子块的。

    P1 命中两个子块(0.82 和 0.71),父块 P1 算几分?
    取【最高】那个:一节里最相关的那句话,代表这一节的相关度。
    (取平均会惩罚长节 —— 长节里必然有很多不相关的句子,不公平)
    """
    store = FakeStore([("P1_c0", 0.82), ("P1_c1", 0.71)])
    r = Retriever(store, FakeRepo(rows))
    got = r.search("PBAT", k=1)

    assert got[0].score == pytest.approx(0.82)


# ────────────── 4. 决策3):返回结构 ──────────────


def test_result_carries_provenance(rows):
    """决策3):LLM 光有文本不够,还得知道【这话哪儿来的】。

    没有出处 = 用户没法核实 = 这个 RAG 不能用在生产上。
    地膜标准这种场景尤其:答错了是要赔钱的。
    """
    store = FakeStore([("P1_c0", 0.82)])
    r = Retriever(store, FakeRepo(rows))
    got = r.search("PBAT", k=1)

    assert got[0].text
    assert got[0].document_id == "d1"  # 哪篇文档
    assert got[0].section_path == ["实验部分", "1.1 主要原料"]  # 哪一节
    assert got[0].score == pytest.approx(0.82)  # 多相关


# ────────────── 5. 过滤参数要透传 ──────────────


def test_filters_pass_through_to_store(rows):
    """year_min / source 要原样传给 Milvus —— 前过滤必须发生在向量检索【内部】。

    如果 Retriever 自己在结果里筛(后过滤),就可能筛完剩 0 条。
    """
    store = FakeStore([("P1_c0", 0.82)])
    r = Retriever(store, FakeRepo(rows))
    r.search("PBAT", k=1, year_min=2020, source="期刊论文")

    assert store.last_call["year_min"] == 2020
    assert store.last_call["source"] == "期刊论文"


# ────────────── 6. 边界 ──────────────


def test_empty_hits(rows):
    """Milvus 什么也没搜到:还空列表,不是崩。"""
    r = Retriever(FakeStore([]), FakeRepo(rows))
    assert r.search("外星人", k=3) == []


def test_missing_parent_is_skipped(rows):
    """索引指向了 PG 里不存在的块:跳过,不是崩。

    真实场景:PG 里删了文档,Milvus 索引还没重建 —— 这是【正常的最终一致】。
    """
    store = FakeStore([("P1_c0", 0.82), ("幽灵子块", 0.7)])
    r = Retriever(store, FakeRepo(rows))
    got = r.search("PBAT", k=2)

    assert [g.chunk_id for g in got] == ["P1"]
