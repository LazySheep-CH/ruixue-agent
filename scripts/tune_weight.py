"""企业标准的超参选择:交叉验证 + 自助置信区间 + 显著性。

回答"权重比企业怎么选"—— 不是手调 4 个点挑最高的(那叫过拟合)。

═══ 三个关键方法 ═══

① 交叉验证(k-fold cross-validation)
   150 题太少,舍不得分出独立 test 集。做法:切成 5 折,每次用 4 折调权重、
   在剩下 1 折上验证,轮 5 次。这样每道题都当过一次"没见过的题"。
   如果 5 折各自选出的最优权重差很多 → 说明"最优权重"是在拟合噪声,不可信。

② 自助置信区间(bootstrap CI)
   从 150 题里【有放回】地重抽 150 题,重算指标,抽 1000 次。
   看这 1000 个结果的分布(2.5% ~ 97.5% 分位)= 95% 置信区间。
   如果混合和纯向量的区间【重叠】,差异就【不显著】—— 是运气不是本事。

③ 加速:两路排名固定,只有融合权重变。
   跑一次检索缓存两路排名,之后扫权重是纯 Python(微秒级)。
   否则每个权重都重跑一遍检索 = 几十分钟。

用法:
    uv run python scripts/tune_weight.py
"""

import json
import sys
import time

from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.repository import PgRepository
from ruixue_agent.rag.bm25 import Bm25Search
from ruixue_agent.rag.fusion import rrf
from ruixue_agent.rag.metrics import recall_at_k, reciprocal_rank
from ruixue_agent.rag.milvus_store import MilvusVectorStore

sys.stdout.reconfigure(encoding="utf-8")

_K = 10
_FANOUT = 3
# 用 numpy 之外的确定性随机:传入种子,可复现
_SEED = 42


def _lcg(seed: int):
    """自己实现的线性同余随机数(避免依赖 numpy,且完全可复现)。"""
    state = seed
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def vector_ranking(store, repo, query: str) -> list[tuple[str, float]]:
    """复刻 Retriever 的向量侧:搜子块 → 折算父块取最高分。"""
    hits = store.search(query, k=_K * _FANOUT)
    if not hits:
        return []
    score_by_child = dict(hits)
    children = repo.get_chunks(list(score_by_child))
    best: dict[str, float] = {}
    for c in children:
        if c.parent_id is None:
            continue
        s = score_by_child[c.chunk_id]
        if s > best.get(c.parent_id, float("-inf")):
            best[c.parent_id] = s
    return sorted(best.items(), key=lambda kv: -kv[1])


def score_at_weight(cache, weight: float, idx: list[int]) -> dict:
    """给定权重,在指定题目子集上算指标。cache 已缓存两路排名,这里纯计算。"""
    r1 = r3 = r10 = mrr = 0.0
    for i in idx:
        vec, lex, gold = cache[i]
        if weight == 0.0:
            fused = [c for c, _ in vec]  # 纯向量
        else:
            fused = [c for c, _ in rrf([vec, lex], weights=[1.0, weight])]
        r1 += recall_at_k(fused, gold, 1)
        r3 += recall_at_k(fused, gold, 3)
        r10 += recall_at_k(fused, gold, 10)
        mrr += reciprocal_rank(fused, gold)
    n = len(idx)
    return {"r1": r1 / n, "r3": r3 / n, "r10": r10 / n, "mrr": mrr / n}


def main() -> None:
    qs = [json.loads(x) for x in open("data/eval/evalset.jsonl", encoding="utf-8")]
    print(f"评测集 {len(qs)} 题\n")

    store = MilvusVectorStore()
    with Session(get_engine()) as sess:
        repo = PgRepository(sess)
        bm25 = Bm25Search(sess)

        # ── 一次性缓存两路排名 ──
        print("① 跑一次检索,缓存两路排名(之后扫权重是免费的)…")
        t0 = time.time()
        cache = []
        for q in qs:
            vec = vector_ranking(store, repo, q["question"])
            lex = bm25.search(q["question"], k=_K * _FANOUT)
            cache.append((vec, lex, set(q["gold_chunk_ids"])))
        print(f"   {time.time() - t0:.0f}s\n")

    all_idx = list(range(len(qs)))
    weights = [round(w * 0.05, 2) for w in range(0, 31)]  # 0.00 ~ 1.50,步长 0.05

    # ── ② 全量细扫(先看全局形状)──
    print("② 全量细扫(注意:这是在【全部】题上调,会过拟合 —— 只用来看形状)")
    grid = {w: score_at_weight(cache, w, all_idx) for w in weights}
    best_full = max(weights, key=lambda w: grid[w]["r3"])
    print(f"   {'权重':>6} {'R@1':>7} {'R@3':>7} {'R@10':>7} {'MRR':>7}")
    for w in weights[::2]:  # 隔一个打,省版面
        m = grid[w]
        star = "  ←最高R@3" if w == best_full else ""
        print(f"   {w:>6} {m['r1']:>7.3f} {m['r3']:>7.3f} {m['r10']:>7.3f} {m['mrr']:>7.3f}{star}")

    # ── ③ 5 折交叉验证(每折在没见过的题上验证)──
    print("\n③ 5 折交叉验证 —— 每折选出的最优权重,看它们稳不稳")
    rng = _lcg(_SEED)
    shuffled = sorted(all_idx, key=lambda _: next(rng))
    folds = [shuffled[i::5] for i in range(5)]
    picked = []
    for f in range(5):
        dev = [i for j in range(5) if j != f for i in folds[j]]  # 4 折调参
        val = folds[f]  # 1 折验证
        best_w = max(weights, key=lambda w: score_at_weight(cache, w, dev)["r3"])
        val_score = score_at_weight(cache, best_w, val)
        base_score = score_at_weight(cache, 0.0, val)
        picked.append(best_w)
        print(
            f"   折{f + 1}: dev选出权重={best_w:<4}  "
            f"验证集 R@3 {base_score['r3']:.3f}(纯向量) → {val_score['r3']:.3f}(混合)"
        )
    lo, hi = min(picked), max(picked)
    print(f"\n   5 折各自选出的权重: {picked}")
    print(f"   范围 {lo}~{hi} —— {'稳(都在小范围)' if hi - lo <= 0.3 else '不稳!说明最优权重在拟合噪声'}")

    # ── ④ 自助置信区间(混合 vs 纯向量,差异显著吗)──
    print("\n④ 自助置信区间(重抽 1000 次)—— 混合真的比纯向量好吗,还是运气?")
    W = 0.3  # 用一个固定的、合理的权重
    rng2 = _lcg(_SEED + 1)
    diffs_r3, diffs_mrr = [], []
    for _ in range(1000):
        sample = [int(next(rng2) * len(qs)) for _ in range(len(qs))]  # 有放回重抽
        hyb = score_at_weight(cache, W, sample)
        vec = score_at_weight(cache, 0.0, sample)
        diffs_r3.append(hyb["r3"] - vec["r3"])
        diffs_mrr.append(hyb["mrr"] - vec["mrr"])
    diffs_r3.sort()
    diffs_mrr.sort()

    def ci(d):
        return d[25], d[975]  # 2.5% ~ 97.5%

    lo3, hi3 = ci(diffs_r3)
    lom, him = ci(diffs_mrr)
    print(f"   权重={W} 相对纯向量的提升(95% 置信区间):")
    print(f"     R@3  提升 [{lo3:+.3f}, {hi3:+.3f}]   {'✓ 显著(区间不含0)' if lo3 > 0 else '✗ 不显著(区间含0)'}")
    print(f"     MRR  提升 [{lom:+.3f}, {him:+.3f}]   {'✓ 显著' if lom > 0 else '✗ 不显著'}")

    print(f"\n{'═' * 60}")
    print("结论要点(面试能讲的):")
    print(f"  · 全量最优权重 {best_full},但那是过拟合的数字")
    print(f"  · 交叉验证权重范围 {lo}~{hi},这才是可信区间")
    print("  · 是否显著看置信区间是否含 0 —— 不含才敢说'确实更好'")


if __name__ == "__main__":
    main()
