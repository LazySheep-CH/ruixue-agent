"""RRF 的测试 —— 这几条锁住它的核心性质。"""

import pytest

from ruixue_agent.rag.fusion import rrf


def test_single_ranking_preserves_order():
    out = rrf([[("a", 0.9), ("b", 0.5), ("c", 0.1)]])
    assert [cid for cid, _ in out] == ["a", "b", "c"]


def test_agreement_wins():
    """两路都投的排在前面 —— 这就是融合的意义。

    b 在两路里都排第 2,a 只在第一路排第 1、第二路根本没出现。
    两票的 b 应该赢过一票的 a。
    """
    out = rrf(
        [
            [("a", 0.9), ("b", 0.8)],
            [("c", 0.9), ("b", 0.8)],
        ]
    )
    assert out[0][0] == "b"


def test_score_scale_does_not_matter():
    """RRF 的核心性质:只看排名,不看分数。

    两路的分数量纲差 1000 倍(向量的余弦 vs BM25 的 ts_rank 就是这种情况),
    融合结果必须完全一样 —— 否则量纲大的那一方会说了算,
    而量纲跟"谁更准"毫无关系。
    """
    a = rrf([[("x", 0.9), ("y", 0.8)], [("y", 0.02), ("x", 0.01)]])
    b = rrf([[("x", 900), ("y", 800)], [("y", 0.00002), ("x", 0.00001)]])
    assert [c for c, _ in a] == [c for c, _ in b]


def test_weights_shift_the_result():
    """加权:让某一路更有话语权。"""
    rankings = [[("a", 1.0)], [("b", 1.0)]]
    assert rrf(rankings, weights=[3.0, 1.0])[0][0] == "a"
    assert rrf(rankings, weights=[1.0, 3.0])[0][0] == "b"


def test_k_controls_head_flatness():
    """k 越大,头部名次的差距被压得越平。

    k 小 → 看重"谁排第一";k 大 → 看重"两边都投票了"。
    """
    r = [[("a", 1.0), ("b", 1.0)]]
    small = dict(rrf(r, k=1))
    large = dict(rrf(r, k=1000))
    assert small["a"] / small["b"] > large["a"] / large["b"]


def test_empty_and_mismatched():
    assert rrf([]) == []
    assert rrf([[], []]) == []
    with pytest.raises(ValueError):
        rrf([[("a", 1.0)]], weights=[1.0, 1.0])
