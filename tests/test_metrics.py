"""指标本身也要测 —— 指标错了,后面所有结论都是错的。

(例:仅统计 type=='text' 字数的质量指标,会把内容全在表格里的
 实测报告误判为空文档 —— 指标本身的缺陷会污染所有下游结论。)
"""

from ruixue_agent.rag.metrics import evaluate, recall_at_k, reciprocal_rank


def test_recall_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"b"}, k=1) == 0.0  # 第2名,不在前1
    assert recall_at_k([], {"b"}, k=3) == 0.0


def test_recall_any_gold_counts(rows=None):
    """多标准答案:命中【任意一个】就算数。

    这正是 pooling 要修的那个 bug:早期每题只标一个 gold,
    检索器返回了库里另一段同样能回答的块,却被判 0 分。
    """
    assert recall_at_k(["x", "b"], {"a", "b"}, k=2) == 1.0  # 命中 b
    assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 1.0  # 命中 a,一样算
    assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0  # 一个没中


def test_reciprocal_rank_is_position_sensitive():
    """这正是 MRR 补 Recall 的地方:排第1和排第3不一样。"""
    assert reciprocal_rank(["b", "x", "y"], {"b"}) == 1.0
    assert reciprocal_rank(["x", "b", "y"], {"b"}) == 0.5
    assert reciprocal_rank(["x", "y", "b"], {"b"}) == 1 / 3
    assert reciprocal_rank(["x", "y"], {"b"}) == 0.0


def test_reciprocal_rank_takes_first_gold():
    """多个标准答案时,按【最靠前】那个算 —— LLM 只要够到一段就能答。"""
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5  # b 在第2,不是 a 的第3


def test_evaluate_averages():
    results = [
        (["a", "b"], {"a"}),  # 第1名 → rr=1.0
        (["x", "c"], {"c"}),  # 第2名 → rr=0.5
        (["p", "q"], {"z"}),  # 没找到 → rr=0
    ]
    m = evaluate(results, ks=(1, 2))
    assert m["recall@1"] == 1 / 3
    assert m["recall@2"] == 2 / 3
    assert m["mrr"] == (1.0 + 0.5 + 0) / 3
    assert m["n"] == 3


def test_evaluate_empty():
    assert evaluate([]) == {}
