"""拿评测集量检索质量,打出基线。

所有检索侧改动(混合权重、embedding 模型、fanout、索引类型)都用本脚本
在同一评测集上对比,量化后再合入。

用法:
    uv run python scripts/run_eval.py                  # 量当前基线
    uv run python scripts/run_eval.py --fanout 1 3 5   # 顺便扫一下 _FANOUT 该取几
"""

import argparse
import json
import sys
import time
from pathlib import Path

from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.repository import PgRepository
from ruixue_agent.rag.bm25 import Bm25Search
from ruixue_agent.rag.metrics import evaluate
from ruixue_agent.rag.milvus_store import MilvusVectorStore
from ruixue_agent.rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")

EVALSET = Path("data/eval/evalset.jsonl")
_MAX_K = 10  # 要算 recall@10,就得取回 10 个父块


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fanout",
        type=int,
        nargs="*",
        default=None,
        help="扫描不同的超取倍数,看哪个最好",
    )
    ap.add_argument(
        "--ab",
        action="store_true",
        help="A/B 对比:纯向量 vs 混合检索(BM25 + RRF),用同一批题",
    )
    ap.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="扫描 RRF 权重,如 --weights 1 2 3 表示 BM25 权重取这几个值(向量固定 1.0)",
    )
    args = ap.parse_args()

    qs = [json.loads(line) for line in EVALSET.read_text(encoding="utf-8").splitlines()]
    print(f"评测集 {len(qs)} 题\n")

    store = MilvusVectorStore()

    with Session(get_engine()) as sess:
        repo = PgRepository(sess)
        bm25 = Bm25Search(sess)

        # ── 要跑哪几个配置 ──
        # 关键:所有配置用【同一批题、同一个 Retriever 类】,只差参数。
        # 这样差值才是那个参数带来的,不掺别的变量。
        configs: list[tuple[str, Retriever, int | None]] = []
        if args.ab:
            configs = [
                ("A · 纯向量(基线)", Retriever(store, repo), None),
                (
                    "B · 混合 向量+BM25 (RRF 1:1)",
                    Retriever(store, repo, bm25=bm25),
                    None,
                ),
            ]
        elif args.weights:
            configs = [("A · 纯向量(基线)", Retriever(store, repo), None)] + [
                (
                    f"BM25 权重 {w}",
                    Retriever(store, repo, bm25=bm25, weights=(1.0, w)),
                    None,
                )
                for w in args.weights
            ]
        elif args.fanout:
            configs = [(f"fanout={f}", Retriever(store, repo), f) for f in args.fanout]
        else:
            configs = [("基线(纯向量)", Retriever(store, repo), None)]

        summary: list[tuple[str, dict, float]] = []

        for tag, r, fanout in configs:
            if fanout is not None:
                import ruixue_agent.rag.retriever as rt

                rt._FANOUT = fanout

            r.search("预热", k=1)  # 预热,别把模型加载算进耗时

            t0 = time.time()
            results = []
            for q in qs:
                hits = r.search(q["question"], k=_MAX_K)
                results.append(([h.chunk_id for h in hits], q["gold_chunk_ids"]))
            el = time.time() - t0

            m = evaluate(results)
            summary.append((tag, m, el / len(qs) * 1000))
            print(f"── {tag} ──")
            print(f"  recall@1  {m['recall@1']:.3f}   ← 一发命中")
            print(f"  recall@3  {m['recall@3']:.3f}   ← 前3有(LLM 通常就吃 3 段)")
            print(f"  recall@5  {m['recall@5']:.3f}")
            print(f"  recall@10 {m['recall@10']:.3f}  ← 天花板:这都没有,后面全白搭")
            print(f"  MRR       {m['mrr']:.3f}   ← 排名质量")
            print(f"  延迟      {el / len(qs) * 1000:.0f} ms/题\n")

            # ── 分组看 ──
            # 一个总分什么也告诉不了你该修哪儿。按题型拆开才知道:
            #   "数值查询"差 → 关键词精确匹配的锅 → 该上 BM25
            #   "因果解释"差 → 语义的锅 → 该换 embedding 模型 / 调块大小
            # 两个病因,两种药。不拆开就只能瞎试。
            for field, label in (("kind", "题型"), ("persona", "角色")):
                if not qs[0].get(field) or tag != configs[-1][0]:
                    continue
                groups: dict[str, list] = {}
                for (ranked, gold), q in zip(results, qs):
                    groups.setdefault(q[field], []).append((ranked, gold))
                print(f"  按{label}拆(从差到好):")
                for name, g in sorted(
                    groups.items(), key=lambda kv: evaluate(kv[1])["recall@3"]
                ):
                    gm = evaluate(g)
                    print(
                        f"    {name:6} n={gm['n']:3}  recall@3 {gm['recall@3']:.3f}"
                        f"  recall@10 {gm['recall@10']:.3f}  MRR {gm['mrr']:.3f}"
                    )
                print()

            # 抽几个没命中的看看 —— 数字告诉你【多少】,badcase 告诉你【为什么】
            if tag == configs[-1][0]:
                misses = [
                    (q, r_)
                    for (r_, g), q in zip(results, qs)
                    if not set(q["gold_chunk_ids"]) & set(r_)
                ]
                if misses:
                    print(f"  没命中的 {len(misses)} 题,抽 3 个看:")
                    for q, _ in misses[:3]:
                        print(f"    ✗ [{q.get('kind', '?')}] {q['question'][:56]}")
                        print(
                            f"      该找到: {' > '.join((q['section_path'] or [])[:2])[:56]}"
                        )

        # ── 汇总对比 ──
        # 多个配置时,并排看差值。绝对值不重要(合成评测集偏高),
        # 【差值】才是可信的 —— 同一批题、同一个检索器,只差一个参数。
        if len(summary) > 1:
            base_tag, base_m, _ = summary[0]
            print("\n" + "═" * 72)
            print(f"配置对比(基准 = {base_tag})\n")
            print(
                f"{'配置':<30} {'R@1':>7} {'R@3':>7} {'R@10':>7} {'MRR':>7} {'ms':>6}"
            )
            print("─" * 72)
            for tag, m, ms in summary:
                print(
                    f"{tag:<30} {m['recall@1']:>7.3f} {m['recall@3']:>7.3f}"
                    f" {m['recall@10']:>7.3f} {m['mrr']:>7.3f} {ms:>6.0f}"
                )
            print("\n相对基准的变化:")
            for tag, m, _ in summary[1:]:
                parts = []
                for key in ("recall@1", "recall@3", "recall@10", "mrr"):
                    d = m[key] - base_m[key]
                    pct = d / base_m[key] * 100 if base_m[key] else 0
                    parts.append(f"{key} {d:+.3f} ({pct:+.1f}%)")
                print(f"  {tag}")
                print(f"    {'  '.join(parts)}")


if __name__ == "__main__":
    main()
