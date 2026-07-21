"""Pooling:把"只标一个标准答案"补全成"所有能答的都算标准答案"。

═══ 为什么必须做这一步 ═══

出题时我们只知道一个 gold —— 就是出题的那个 chunk。但库里有 26 万块,
另一段同样能回答的段落几乎必然存在,检索器返回它却被判 0 分 ——
测得的不是检索质量,而是标注缺口。

TREC(信息检索领域几十年的评测标准)的解法叫 pooling:
    1. 用【多个不同的检索器】各捞 top-N 候选
    2. 合并成一个候选池(pool)
    3. 逐条让裁判判"这段到底能不能回答这个问题"
    4. 能答的【全部】算标准答案

为什么要【多个】检索器而不是一个:
    只用向量检索捞候选的话,池子里全是"向量觉得像"的东西 ——
    向量漏掉的(比如精确的数字、型号)永远进不了池,永远不会被标成 gold。
    那 BM25 找到它时反而被判错。所以必须让不同原理的检索器都往池里丢。
    这里用:向量检索(语义)+ PG 全文检索(词法)。

注意:Pooling 的已知局限(TREC 也有,这是公认的代价):
    池子只覆盖【参与 pooling 的检索器】能找到的东西。
    将来来个全新原理的检索器,它找到的正确答案可能不在池里 → 被低估。
    缓解办法就是让池子尽量多样。我们两种原理都放了。

用法:
    uv run python scripts/pool_evalset.py            # 补全 gold
    uv run python scripts/pool_evalset.py --depth 15 # 每个检索器捞多少
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from ruixue_agent.models import create_model
from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.repository import PgRepository
from ruixue_agent.rag.milvus_store import MilvusVectorStore
from ruixue_agent.rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")

EVALSET = Path("data/eval/evalset.jsonl")

_JUDGE = """判断下面这段内容能不能回答这个问题。

问题:{question}

内容:
\"\"\"
{text}
\"\"\"

标准:内容里要有【直接支撑答案的信息】。
  - 只是同一个话题、但答不出具体答案 → 不算
  - 能答出问题问的那个点 → 算

只输出 JSON:{{"can_answer": true/false}}"""


def _lexical_search(conn, query: str, k: int) -> list[str]:
    """PG 全文检索捞候选 —— 和向量【不同原理】,这正是它进池子的意义。

    用的是 chunks.text_tsv 那一列(触发器一直在维护)。
    ts_rank 不是严格的 BM25,但同属词法排序 —— 对 pooling 来说足够:
    我们要的是"向量漏掉的、靠关键词能找到的东西"也能进池。

    注意:'simple' 配置不做中文分词(已知技术债),所以它主要在英文术语、
      数字、型号上发力 —— 而那恰好是向量最弱的地方。互补正好。
    """
    rows = conn.execute(
        text("""
        SELECT c.chunk_id, ts_rank(p.text_tsv, plainto_tsquery('simple', :q)) AS rank
        FROM chunks c JOIN chunks p ON p.chunk_id = c.chunk_id
        WHERE c.kind = 'parent' AND p.text_tsv @@ plainto_tsquery('simple', :q)
        ORDER BY rank DESC LIMIT :k
    """),
        {"q": query, "k": k},
    ).all()
    return [r[0] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=12, help="每个检索器往池里丢多少")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="deepseek-v4-flash")
    args = ap.parse_args()

    qs = [json.loads(x) for x in EVALSET.read_text(encoding="utf-8").splitlines()]
    print(f"评测集 {len(qs)} 题,每个检索器捞 {args.depth} 个候选\n")

    llm = create_model(args.model)
    store = MilvusVectorStore()
    engine = get_engine()

    # ── 建池 ──
    print("① 建候选池(向量 + 全文,两种原理)…")
    t0 = time.time()
    pools: list[list[str]] = []
    with Session(engine) as sess, engine.connect() as conn:
        r = Retriever(store, PgRepository(sess))
        r.search("预热", k=1)
        for i, q in enumerate(qs):
            vec = [h.chunk_id for h in r.search(q["question"], k=args.depth)]
            lex = _lexical_search(conn, q["question"], args.depth)
            # 已知的 gold 也丢进去 —— 它必然能答(出题就是它出的),
            # 但万一两个检索器都没捞到它,也得在池里
            pool = list(dict.fromkeys([*q["gold_chunk_ids"], *vec, *lex]))
            pools.append(pool)
            if (i + 1) % 40 == 0:
                print(
                    f"   {i + 1}/{len(qs)}  平均池大小 {sum(len(p) for p in pools) / len(pools):.1f}"
                )

    sizes = [len(p) for p in pools]
    print(
        f"   池平均 {sum(sizes) / len(sizes):.1f} 个候选,共 {sum(sizes)} 条要判  ({time.time() - t0:.0f}s)\n"
    )

    # ── 取文本 ──
    all_ids = list({cid for p in pools for cid in p})
    with Session(engine) as sess:
        rows = PgRepository(sess).get_chunks(all_ids)
        texts = {row.chunk_id: row.text for row in rows}

    # ── 逐条裁判 ──
    print(f"② 裁判逐条判(共 {sum(sizes)} 条)…")
    t0 = time.time()

    def judge(item):
        qi, cid = item
        if cid == qs[qi]["primary_gold"]:
            return qi, cid, True  # 出题的那个,不用判,省一次调用
        txt = texts.get(cid)
        if not txt:
            return qi, cid, False
        try:
            raw = llm.invoke(
                _JUDGE.format(question=qs[qi]["question"], text=txt[:1500])
            ).content
            raw = raw[raw.find("{") : raw.rfind("}") + 1]
            return qi, cid, bool(json.loads(raw).get("can_answer"))
        except Exception:
            return qi, cid, False  # 判不了就当不能答 —— 保守:宁可少标,不可错标

    jobs = [(qi, cid) for qi, p in enumerate(pools) for cid in p]
    golds: list[set[str]] = [set() for _ in qs]
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        for qi, cid, ok in pool_exec.map(judge, jobs):
            if ok:
                golds[qi].add(cid)
            done += 1
            if done % 200 == 0:
                print(f"   {done}/{len(jobs)}  ({time.time() - t0:.0f}s)")

    # ── 写回 ──
    for q, g in zip(qs, golds):
        g.add(q["primary_gold"])  # 兜底:出题的那个永远算 gold
        q["gold_chunk_ids"] = sorted(g)

    with EVALSET.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    n_gold = [len(q["gold_chunk_ids"]) for q in qs]
    print(f"\n{'═' * 66}")
    print(f"补全完成,耗时 {time.time() - t0:.0f}s")
    print(f"  每题标准答案数:平均 {sum(n_gold) / len(n_gold):.2f},最多 {max(n_gold)}")
    print(f"  分布:{dict(sorted(Counter(n_gold).items()))}")
    print(f"  只有 1 个 gold 的题:{sum(1 for x in n_gold if x == 1)}/{len(qs)}")
    print("\n标准答案数大于 1 的题,是此前单标签评测会误判为未命中的部分;")
    print("比例越高,说明单标签评测的低估越严重。")

    print("\n按题型看平均 gold 数(高 = 这类题本来就有多个正确答案):")
    by_kind: dict[str, list[int]] = {}
    for q in qs:
        by_kind.setdefault(q["kind"], []).append(len(q["gold_chunk_ids"]))
    for k, v in sorted(by_kind.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {k:6}: {sum(v) / len(v):.2f}  (n={len(v)})")


if __name__ == "__main__":
    main()
