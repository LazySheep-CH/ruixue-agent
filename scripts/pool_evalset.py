"""TREC 式 pooling:把单标注升级成多标注 + relevance 分级(修正尺子偏严)。

═══ 为什么 ═══
出题只绑 1 个 gold,但库里 26 万块,一个问题(尤其口语题)常有多篇能答。检索找到
另一篇有效的却被判未命中 → 系统性【低估】Recall。这不是检索差,是标注口径窄。
多标注是企业/微软/RAGAS 检索评测的前提(算 nDCG 也需要分级)。

═══ 怎么做(吸取上次 reranker 翻车的教训)═══
上次用本地 reranker 当 judge → 它把"同话题"给 0.9+,过度标注、把 recall 刷虚高。
这次:① 多路建池(向量+BM25,不同原理都进池)② 用 LLM judge,且 prompt 严格区分
"能答出那个具体点"(2-3 分)和"只是同话题"(1 分)③ 保守:宁可给低。
relevance:3 直接完整含核心事实 / 2 必要证据的一部分 / 1 同话题答不出 / 0 无关。
gold(算 Recall/MRR)= relevance≥2;nDCG 用完整分级。primary_gold(出题确认过)记 3。

用法:uv run python scripts/pool_evalset.py [--depth 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy.orm import Session

from ruixue_agent.models import create_model
from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.repository import PgRepository
from ruixue_agent.rag.bm25 import Bm25Search
from ruixue_agent.rag.milvus_store import MilvusVectorStore

sys.stdout.reconfigure(encoding="utf-8")
EVAL = Path("data/eval/evalset.jsonl")

_JUDGE = """判断下面这段【内容】对【问题】的相关程度,按四级打分:
3 = 直接且完整地包含回答这个问题所需的核心事实——单看这段就能答出问题问的那个具体点
2 = 包含回答所需的必要证据,但只是一部分,单独不足以完整回答
1 = 与问题同主题,但答不出问题问的那个具体点(比如只泛泛谈到这个话题)
0 = 与问题无关

关键:只有能给出"问题问的那个具体答案"才给 2 或 3;"只是聊到同一个话题、给不出那个数/那个结论"一律给 1。宁可给低,不要给高。

问题:{q}

内容:
\"\"\"
{t}
\"\"\"

只输出一个数字(0/1/2/3),不要解释。"""


def build_pool(store, repo, bm25, question, seed, depth):
    """多路建父块候选池:向量(子块折父)+ BM25(父块)+ seed 保底。"""
    hits = store.search(question, k=depth * 5)  # 子块
    sbc = {cid: s for cid, s in hits}
    best = {}
    for c in repo.get_chunks(list(sbc)):
        if c.parent_id and sbc[c.chunk_id] > best.get(c.parent_id, -1e9):
            best[c.parent_id] = sbc[c.chunk_id]
    vec = [p for p, _ in sorted(best.items(), key=lambda kv: -kv[1])[:depth]]
    lex = [p for p, _ in bm25.search(question, k=depth)]
    return list(dict.fromkeys([*seed, *vec, *lex]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--model", default="deepseek-v4-flash")
    args = ap.parse_args()

    qs = [json.loads(x) for x in EVAL.read_text(encoding="utf-8").splitlines()]
    ans = [q for q in qs if q.get("has_answer")]
    print(f"评测集 {len(qs)} 题(有答案 {len(ans)}),每路捞 {args.depth} 候选\n")

    store = MilvusVectorStore()
    llm = create_model(args.model)

    # ① 建池
    print("① 多路建候选池(向量+BM25)…")
    t0 = time.time()
    pools = {}
    with Session(get_engine()) as sess:
        repo = PgRepository(sess)
        bm25 = Bm25Search(sess)
        store.search("预热", k=1)
        for i, q in enumerate(ans):
            seed = [g for g in q.get("gold_chunk_ids", []) if g]
            pools[q["question"]] = build_pool(store, repo, bm25, q["question"], seed, args.depth)
            if (i + 1) % 60 == 0:
                print(f"   {i + 1}/{len(ans)}")
        # 取候选正文
        all_ids = list({c for p in pools.values() for c in p})
        texts = {r.chunk_id: r.text for r in repo.get_chunks(all_ids)}
    avg_pool = sum(len(p) for p in pools.values()) / len(pools)
    print(f"   池均 {avg_pool:.1f} 候选  ({time.time() - t0:.0f}s)\n")

    # ② LLM judge(seed 直接记 3,不判)
    print("② LLM judge 逐候选评 relevance(严格)…")
    t0 = time.time()
    jobs = []  # (question, cid)
    for q in ans:
        seeds = set(g for g in q.get("gold_chunk_ids", []) if g)
        for c in pools[q["question"]]:
            if c not in seeds and c in texts:
                jobs.append((q["question"], c))

    def judge(item):
        ques, cid = item
        try:
            raw = llm.invoke(_JUDGE.format(q=ques, t=texts[cid][:1400])).content.strip()
            d = next((ch for ch in raw if ch in "0123"), "0")
            return ques, cid, int(d)
        except Exception:
            return ques, cid, 0  # 判不了当无关(保守:宁少不多)

    rel = {q["question"]: {} for q in ans}
    for q in ans:  # seed 记 3
        for g in q.get("gold_chunk_ids", []):
            if g:
                rel[q["question"]][g] = 3
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ques, cid, grade in ex.map(judge, jobs):
            if grade >= 1:
                rel[ques][cid] = grade
            done += 1
            if done % 400 == 0:
                print(f"   {done}/{len(jobs)}  ({time.time() - t0:.0f}s)")
    print(f"   judge 完成 {len(jobs)} 次  ({time.time() - t0:.0f}s)\n")

    # ③ 写回:relevance 全分级;gold_chunk_ids = relevance≥2
    for q in qs:
        if not q.get("has_answer"):
            q["relevance"] = {}
            continue
        r = rel[q["question"]]
        q["relevance"] = r
        q["gold_chunk_ids"] = sorted([c for c, g in r.items() if g >= 2])
        if q.get("primary_gold") and q["primary_gold"] not in q["gold_chunk_ids"]:
            q["gold_chunk_ids"].append(q["primary_gold"])  # 保底
    EVAL.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n", encoding="utf-8")

    # ④ 质量检查
    ng = [len(q["gold_chunk_ids"]) for q in ans]
    print(f"{'═' * 60}\n质量检查\n{'═' * 60}")
    print(f"  每题 gold 数(rel≥2):平均 {sum(ng) / len(ng):.2f},最多 {max(ng)}")
    print(f"  分布:{dict(sorted(Counter(ng).items()))}")
    print(f"  只有 1 个 gold 的题:{sum(1 for x in ng if x == 1)}/{len(ans)}"
          f"(比例越低,说明单标注低估越严重)")
    by = {}
    for q in ans:
        by.setdefault(q["strategy"], []).append(len(q["gold_chunk_ids"]))
    print("  按策略平均 gold 数:")
    for s, v in by.items():
        print(f"    {s:9} {sum(v) / len(v):.2f}")
    # 抽查:primary_gold 都在 gold 里(保底)
    miss = [q["question"] for q in ans if q.get("primary_gold") and q["primary_gold"] not in q["gold_chunk_ids"]]
    print(f"  primary_gold 保底检查:{'✓ 全部在 gold 内' if not miss else f'✗ {len(miss)} 题缺失'}")
    print(f"\n→ 已写回 {EVAL}")


if __name__ == "__main__":
    main()
