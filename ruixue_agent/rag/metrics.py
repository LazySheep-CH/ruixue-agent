"""检索评测指标。

只有这两个数才回答"我的 RAG 好不好"。之前量的 59x 去重、210x 查询、150ms 延迟
—— 全都在量【快不快】,没有一个在量【对不对】。而对不对才是这个系统的唯一价值。

★ 标准答案是【一组】,不是一个。
  早期版本每题只标一个 gold,结果:库里另一段同样能答,检索器返回了它 → 判 0 分。
  量到的不是"检索差",是"尺子冤枉了它"。
  专业做法(TREC 几十年的标准)是 pooling:把多个检索器的候选捞出来逐条判,
  能答的全都算标准答案。所以这里一律按【集合】处理。
"""

from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(ranked_ids: list[str], gold_ids: Iterable[str], k: int) -> float:
    """前 k 个里有【任意一个】标准答案吗?有=1,没有=0。

    为什么是"任意一个"而不是"全部":
        RAG 要的是"LLM 手里有能答这题的材料"。有一段够答就行,
        不需要把所有相关段落都捞全。(那是另一个指标,叫 Recall 的严格版,
        适用于"找出所有相关文献"这种场景,不是我们这个。)

    为什么它最重要:检索是 RAG 的【天花板】。
    没检索到 → LLM 手里根本没有那段材料 → 它只能瞎编或者说不知道。
    后面的 rerank / prompt / 换更强的模型,一个都救不回来。
    Recall 是 0,后面全是 0。
    """
    gold = set(gold_ids)
    return 1.0 if any(cid in gold for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_ids: Iterable[str]) -> float:
    """【第一个】标准答案排第几的倒数:第1名=1.0,第2名=0.5,没找到=0。

    为什么光有 Recall 不够:
        Recall@5 只问"在不在前5",排第1和排第5【一样得分】。
        但对 LLM 完全不同 —— 排第5意味着它前面有4段无关材料,
        很可能被带偏,或者干脆超出 context 被截掉。
    Recall 看"找没找到",MRR 看"找得好不好"。两个一起看才完整。
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
    """results: [(检索出的id列表, 标准答案id集合), ...] → {"recall@1": .., "mrr": ..}

    MRR = Mean Reciprocal Rank,所有查询的 reciprocal_rank 求平均。
    """
    if not results:
        return {}
    out = {
        f"recall@{k}": sum(recall_at_k(r, g, k) for r, g in results) / len(results)
        for k in ks
    }
    out["mrr"] = sum(reciprocal_rank(r, g) for r, g in results) / len(results)
    out["n"] = len(results)
    return out
