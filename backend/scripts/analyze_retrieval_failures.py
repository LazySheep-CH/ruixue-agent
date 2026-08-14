"""Phase 0:定位当前检索的失败题,判断知识图谱能不能救。

【为什么先做这一步,而不是直接把图接进来】
知识图谱在检索里能提供的东西是【结构化的多跳连接】——A 关联 B、B 关联 C,
于是问 A 能带出 C。它对以下两类失败【完全无效】:

    1. 排序失败(gold 在 top-10 里但不在第 1)—— 那是重排的活,图改不了顺序
    2. 词汇鸿沟(用户说大白话,文献用术语)—— 图里存的也是术语,一样对不上

所以真正值得投图的,只有一种失败:**gold 压根没进 top-10,且这题需要跨文档
把两条知识连起来**。这个脚本就是去数这一类到底有多少。

GraphRAG-Bench(arXiv 2506.05690)的结论与此一致:图在 L1 简单事实检索上
【输给】普通 RAG,只在多跳推理和全局摘要上有优势;token 开销还差两个数量级
(vanilla ~900 vs GraphRAG 3,000–330,000+)。所以"图一定更好"是错的,
必须先在【自己的数据】上量出那一类失败的占比。

【判据 —— 跑之前就先定好,免得看到结果再找理由】
    未命中@10 的题中,multihop 占比 < 30%  →  图救不了,不做 Phase 1-3
    30% ~ 50%                          →  边际,看绝对题数再定
    > 50%                              →  图有明确空间,继续做

先定判据再看数,是为了防止"结果出来后按结果编一个支持自己想做的事的理由"。

用法:
    uv run python scripts/analyze_retrieval_failures.py
    uv run python scripts/analyze_retrieval_failures.py --dump 25   # 多导几条失败样本
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.repository import PgRepository
from ruixue_agent.rag.bm25 import Bm25Search
from ruixue_agent.rag.milvus_store import MilvusVectorStore
from ruixue_agent.rag.rerank import Reranker
from ruixue_agent.rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")

EVAL = Path("data/eval/evalset.jsonl")
OUT = Path("data/eval/baselines/failure_analysis.json")

# 判据(跑之前定死,见模块 docstring)
THRESHOLD_GO = 0.50
THRESHOLD_MAYBE = 0.30


def _first_gold_rank(ranked: list[str], gold: set[str]) -> int | None:
    """gold 第一次出现在第几名(1-based);没出现返回 None。"""
    for i, cid in enumerate(ranked, 1):
        if cid in gold:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=15, help="导出多少条未命中样本供人工看")
    args = ap.parse_args()

    qs = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines() if x.strip()]
    answerable = [q for q in qs if q.get("has_answer")]
    print(f"评测集 {len(qs)} 题,其中有答案 {len(answerable)} 题\n")

    with Session(get_engine()) as s:
        repo = PgRepository(s)
        r = Retriever(MilvusVectorStore(), repo, bm25=Bm25Search(s), reranker=Reranker())
        r.search("预热", k=1)

        buckets: Counter[str] = Counter()
        by_strategy: dict[str, Counter[str]] = {}
        misses: list[dict] = []

        for q in answerable:
            gold = set(q["gold_chunk_ids"])
            ranked = [h.chunk_id for h in r.search(q["question"], k=10)]
            rank = _first_gold_rank(ranked, gold)

            if rank == 1:
                bucket = "命中@1"
            elif rank is not None:
                bucket = "命中@2-10(排序问题)"
            else:
                bucket = "未命中@10(召回失败)"
                misses.append(
                    {
                        "strategy": q["strategy"],
                        "question": q["question"],
                        "gold_title": q.get("title", ""),
                        "gold_text": (q.get("gold_text") or "")[:220],
                        "retrieved_top3": ranked[:3],
                    }
                )

            buckets[bucket] += 1
            by_strategy.setdefault(q["strategy"], Counter())[bucket] += 1

    total = sum(buckets.values())
    print("=" * 62)
    print("失败归因(只有第三类图才【可能】有帮助)")
    print("=" * 62)
    for k in ("命中@1", "命中@2-10(排序问题)", "未命中@10(召回失败)"):
        n = buckets[k]
        print(f"  {k:24s} {n:4d}  ({n / total * 100:5.1f}%)")

    print("\n分策略:")
    header = f"  {'策略':10s} {'命中@1':>8s} {'@2-10':>8s} {'未命中':>8s} {'合计':>6s}"
    print(header)
    for st, c in sorted(by_strategy.items()):
        tot = sum(c.values())
        print(
            f"  {st:10s} {c['命中@1']:8d} {c['命中@2-10(排序问题)']:8d} "
            f"{c['未命中@10(召回失败)']:8d} {tot:6d}"
        )

    # —— 判据 ——
    miss_strategies = Counter(m["strategy"] for m in misses)
    n_miss = len(misses)
    multihop_ratio = miss_strategies["multihop"] / n_miss if n_miss else 0.0

    print("\n" + "=" * 62)
    print("判据(跑之前已定,见脚本 docstring)")
    print("=" * 62)
    print(f"  未命中@10 共 {n_miss} 题,策略分布 {dict(miss_strategies)}")
    print(f"  其中 multihop 占比 = {multihop_ratio:.1%}")
    if multihop_ratio > THRESHOLD_GO:
        verdict = "继续 —— 图有明确空间,进 Phase 1(修图结构)"
    elif multihop_ratio > THRESHOLD_MAYBE:
        verdict = f"边际 —— 看绝对题数({n_miss} 题)决定投入是否值得"
    else:
        verdict = "不做 —— 失败主要不是多跳缺连接,图救不了,Phase 1-3 取消"
    print(f"  结论:{verdict}")

    if misses:
        print(f"\n未命中样本(前 {min(args.dump, len(misses))} 条,人工判断图能否救):")
        for m in misses[: args.dump]:
            print(f"\n  [{m['strategy']}] {m['question']}")
            print(f"    应命中:《{m['gold_title']}》 {m['gold_text'][:110]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "n_answerable": total,
                "buckets": dict(buckets),
                "by_strategy": {k: dict(v) for k, v in by_strategy.items()},
                "miss_strategies": dict(miss_strategies),
                "multihop_ratio_in_misses": round(multihop_ratio, 4),
                "verdict": verdict,
                "misses": misses,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n完整结果已写入 {OUT}")


if __name__ == "__main__":
    main()
