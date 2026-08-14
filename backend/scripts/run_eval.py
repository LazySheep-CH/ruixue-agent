"""检索评测入口:在冻结评测集上量 Recall@k / MRR,分策略诊断。

评测集 data/eval/evalset.jsonl(338 题,多标注 + relevance)。指标 Recall@k / MRR
(下游是大模型读 top-k,这两个直接决定"有没有对的料、最相关的靠不靠前";nDCG 不测,
理由见 docs/检索优化记录.md)。所有检索侧改动都用本脚本对比,量化后再合入。

用法:
    uv run python scripts/run_eval.py                 # 上线管线(向量+BM25+rerank)
    uv run python scripts/run_eval.py --rewrite       # 叠加查询改写(实时调 LLM)
    uv run python scripts/run_eval.py --ab            # 逐层对比:纯向量 / +BM25 / +rerank
    uv run python scripts/run_eval.py --fanout 2      # 覆盖 fanout(默认取 retriever 设置)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sqlalchemy.orm import Session

import ruixue_agent.rag.retriever as rt
from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.repository import PgRepository
from ruixue_agent.rag.bm25 import Bm25Search
from ruixue_agent.rag.metrics import evaluate
from ruixue_agent.rag.milvus_store import MilvusVectorStore
from ruixue_agent.rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")
EVAL = Path("data/eval/evalset.jsonl")


def run(retriever, qs):
    """跑一个配置:返回 (总体指标, 分策略指标, (拒答正确, 拒答总数))。"""
    res, groups, na_ok, na_tot = [], {}, 0, 0
    for q in qs:
        ranked = [h.chunk_id for h in retriever.search(q["question"], k=10)]
        if not q.get("has_answer"):
            na_tot += 1
            na_ok += 1 if not ranked else 0
            continue
        res.append((ranked, q["gold_chunk_ids"]))
        groups.setdefault(q["strategy"], []).append((ranked, q["gold_chunk_ids"]))
    return evaluate(res), {k: evaluate(v) for k, v in groups.items()}, (na_ok, na_tot)


def _print(tag, m, per, na, dt):
    print(f"── {tag} ──(有答案 {m['n']} 题)")
    print(
        f"  R@1 {m['recall@1']:.3f}  R@3 {m['recall@3']:.3f}  R@5 {m['recall@5']:.3f}"
        f"  R@10 {m['recall@10']:.3f}  MRR {m['mrr']:.3f}  ({dt:.0f}ms/题)"
    )
    print(
        "  分策略 R@3:  "
        + "  ".join(
            f"{s} {per.get(s, {}).get('recall@3', 0):.3f}" for s in ("fact", "user", "multihop")
        )
    )
    if na[1]:
        print("  拒答:检索永返回结果,拒答是生成层职责(见优化记录),此处不计入 Recall")


BASELINE = Path("data/eval/baselines/retrieval_latest.json")


def _save_baseline(rows: list[tuple[str, dict]], n_total: int, n_answerable: int) -> None:
    """把这次跑出的指标落成机器可读基线,供 README 一致性测试比对。

    为什么【不直接改写 README】—— 这是我否掉的第一版设计:
        脚本一旦有权改 README,任何一次 debug 跑(--fanout 1、跑一半 Ctrl-C、
        改了检索参数试水)都会把线上口径的数字覆盖成假数,而且是【静默】覆盖。
        文档写错至少还看得见;被脚本写错则连"曾经是什么"都丢了。

    改成:脚本只负责【如实记录自己跑出了什么】,README 由测试盯着(见
    tests/test_readme_metrics.py)。两者不一致时测试红,由人决定是"文档过期了"
    还是"这次跑法不标准",而不是让机器替人做这个判断。

    只在【标准口径】(--ab 且不叠加改写、不覆盖 fanout)时写,
    非标准跑法不污染基线 —— 这正是上面担心的那个坑。
    """
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(
        json.dumps(
            {
                "evalset": str(EVAL),
                "n_total": n_total,
                "n_answerable": n_answerable,
                "layers": {tag: {k: round(v, 4) for k, v in m.items()} for tag, m in rows},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"基线已写入 {BASELINE}(README 指标表由 tests/test_readme_metrics.py 盯着)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="逐层对比:纯向量 / +BM25 / +rerank")
    ap.add_argument("--rewrite", action="store_true", help="叠加查询改写(实时调 LLM)")
    ap.add_argument("--fanout", type=int, default=None, help="覆盖 fanout(默认取 retriever 设置)")
    args = ap.parse_args()

    if args.fanout:
        rt._FANOUT = args.fanout
    qs = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines()]
    print(f"评测集 {len(qs)} 题(fanout={rt._FANOUT})\n")

    store = MilvusVectorStore()
    with Session(get_engine()) as sess:
        repo = PgRepository(sess)
        bm25 = Bm25Search(sess)
        rewriter = None
        if args.rewrite:
            from ruixue_agent.rag.query_rewrite import QueryRewriter

            rewriter = QueryRewriter()

        if args.ab:
            from ruixue_agent.rag.rerank import Reranker

            rr = Reranker()
            configs = [
                ("纯向量", Retriever(store, repo, rewriter=rewriter)),
                ("+BM25(混合)", Retriever(store, repo, bm25=bm25, rewriter=rewriter)),
                ("+BM25+rerank", Retriever(store, repo, bm25=bm25, reranker=rr, rewriter=rewriter)),
            ]
        else:
            from ruixue_agent.rag.rerank import Reranker

            tag = "上线管线(向量+BM25+rerank" + ("+改写)" if rewriter else ")")
            configs = [
                (tag, Retriever(store, repo, bm25=bm25, reranker=Reranker(), rewriter=rewriter))
            ]

        configs[0][1].search("预热", k=1)
        rows = []
        for tag, r in configs:
            t0 = time.time()
            m, per, na = run(r, qs)
            _print(tag, m, per, na, (time.time() - t0) / max(m["n"], 1) * 1000)
            rows.append((tag, m))
            print()

        # 只有【标准口径】才更新基线:必须是 --ab、不叠改写、不覆盖 fanout。
        # 任何非标准跑法都不许动基线,否则 README 一致性测试会被假数带跑偏。
        if args.ab and not args.rewrite and args.fanout is None:
            _save_baseline(rows, len(qs), rows[0][1]["n"])


if __name__ == "__main__":
    main()
