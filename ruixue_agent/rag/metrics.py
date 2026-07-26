"""检索评测指标:Recall@k、MRR、nDCG@k。

标准答案按集合处理,一题可有多个正确块 —— 单标签会把"检索器返回了另一段
同样能回答的块"误判为失败,系统性低估 Recall。多标注(见 scripts/pool_evalset.py)
用 relevance 分级(0-3)修正这一点,并支撑 nDCG(同时看相关程度与排序)。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def recall_at_k(ranked_ids: list[str], gold_ids: Iterable[str], k: int) -> float:
    """前 k 个结果中含任一标准答案则为 1,否则 0。

    取"任一"而非"全部":下游只需一段足以回答的材料即可生成,
    无需捞全所有相关段落。Recall 是整条链路的上限。
    """
    gold = set(gold_ids)
    return 1.0 if any(cid in gold for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_ids: Iterable[str]) -> float:
    """首个标准答案排名的倒数:第 1 名 1.0,第 2 名 0.5,未命中 0。"""
    gold = set(gold_ids)
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[str], rel_map: Mapping[str, int], k: int) -> float:
    """nDCG@k:用 relevance 分级(0-3)同时衡量"找到多相关的块"和"排得多靠前"。

    DCG 用 (2^rel − 1) / log2(i+1);IDCG 为理想排序(按 rel 降序)的 DCG。
    rel_map 为空(该题无分级)返回 0,由调用方决定是否计入平均。
    """
    if not rel_map:
        return 0.0
    dcg = sum(
        (2 ** rel_map.get(cid, 0) - 1) / math.log2(i + 1)
        for i, cid in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    results: list[tuple],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """批量计算指标。

    results 元素为 (检索结果 id 列表, 标准答案 id 集合) 或
    (检索结果 id 列表, 标准答案 id 集合, relevance 分级 dict)。
    含第三项时额外计算 nDCG@k(仅在提供了分级的题上平均)。
    """
    if not results:
        return {}
    out = {
        f"recall@{k}": sum(recall_at_k(r[0], r[1], k) for r in results) / len(results) for k in ks
    }
    out["mrr"] = sum(reciprocal_rank(r[0], r[1]) for r in results) / len(results)
    out["n"] = len(results)

    graded = [r for r in results if len(r) >= 3 and r[2]]
    if graded:
        for k in ks:
            out[f"ndcg@{k}"] = sum(ndcg_at_k(r[0], r[2], k) for r in graded) / len(graded)
    return out
