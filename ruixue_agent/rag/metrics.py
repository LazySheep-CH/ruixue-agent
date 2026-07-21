"""检索评测指标:Recall@k 与 MRR。

标准答案按集合处理,一题可有多个正确块。单标签评测会把"检索器返回了
另一段同样能回答的块"误判为失败 —— 对照实验显示单标签把 Recall@3
低估约 28%,标注方法见 scripts/pool_evalset.py。
"""

from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(ranked_ids: list[str], gold_ids: Iterable[str], k: int) -> float:
    """前 k 个结果中含任一标准答案则为 1,否则 0。

    取"任一"而非"全部":下游只需一段足以回答的材料即可生成,
    无需捞全所有相关段落。

    Recall 是整条链路的上限 —— 未召回的内容,重排和生成都无法补救。
    """
    gold = set(gold_ids)
    return 1.0 if any(cid in gold for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_ids: Iterable[str]) -> float:
    """首个标准答案排名的倒数:第 1 名 1.0,第 2 名 0.5,未命中 0。

    补充 Recall 感知不到的排序质量:Recall@5 对排第 1 和排第 5 同分,
    但对生成层差别很大 —— 靠后意味着前面塞进了无关材料。
    """
    gold = set(gold_ids)
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def evaluate(
    results: list[tuple[list[str], Iterable[str]]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """批量计算指标。results: [(检索结果 id 列表, 标准答案 id 集合), ...]。"""
    if not results:
        return {}
    out = {
        f"recall@{k}": sum(recall_at_k(r, g, k) for r, g in results) / len(results)
        for k in ks
    }
    out["mrr"] = sum(reciprocal_rank(r, g) for r, g in results) / len(results)
    out["n"] = len(results)
    return out
