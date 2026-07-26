"""全量过管道:mineru-out → parse → clean → dedup → metadata → quality → chunk → 落盘。

不做 embedding(那步贵,留到 Milvus 一次做对)。几分钟就能跑完,拿到真实统计。

用法: uv run python scripts/run_pipeline.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

from ruixue_agent.ingestion.cache import FAILED, save_chunks, save_document
from ruixue_agent.ingestion.parsers.mineru_parser import (
    UNKNOWN_TYPES,
    parse_content_list,
)
from ruixue_agent.ingestion.stages.chunk import chunk_document
from ruixue_agent.ingestion.stages.clean import clean_document
from ruixue_agent.ingestion.stages.dedup import dedup_documents
from ruixue_agent.ingestion.stages.metadata import extract_metadata
from ruixue_agent.ingestion.stages.quality import assess_quality

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK,不改打不出中文

PROJ = Path(__file__).resolve().parent.parent
MINERU_OUT = Path("D:/LLM_RuiXue/mineru-out")


def load_manifest() -> dict[str, dict]:
    """document_id → manifest 行(排除已标记 excluded 的)。"""
    mf = PROJ / "data" / "raw" / "raw_manifest.jsonl"
    rows = [json.loads(line) for line in mf.read_text(encoding="utf-8").splitlines()]
    return {r["document_id"]: r for r in rows if not r.get("excluded")}


def main() -> None:
    t0 = time.time()
    manifest = load_manifest()

    # ── 阶段1:parse + clean(每篇独立)──
    print("① parse + clean …")
    docs = []
    miss = 0
    for cat in ("literature", "standards"):
        for doc_dir in sorted((MINERU_OUT / cat).iterdir()):
            row = manifest.get(doc_dir.name)
            if not row:  # 已排除的 or 不在账上
                continue
            p = next(
                (x for x in doc_dir.glob("*/*_content_list.json") if "v2" not in x.name),
                None,
            )
            if not p:
                miss += 1
                continue
            doc = parse_content_list(
                json.load(open(p, encoding="utf-8")),
                {
                    "document_id": row["document_id"],
                    "original_filename": row["original_filename"],
                    "sha256": row["sha256"],
                    "source": "标准规范" if cat == "standards" else "期刊论文",
                },
            )
            clean_document(doc)
            docs.append(doc)
    print(f"   解析 {len(docs)} 篇 ({time.time() - t0:.0f}s)" + (f",缺输出 {miss}" if miss else ""))

    # ── 阶段2:dedup(跨文档,要看全集)──
    t = time.time()
    print("② dedup(Jaccard 近似去重,两两比对)…")
    kept, dropped = dedup_documents(docs)
    print(f"   保留 {len(kept)},丢弃近似重复 {len(dropped)} ({time.time() - t:.0f}s)")
    for d in dropped[:8]:
        print(f"     重复: {d.meta.get('title') or d.filename[:44]}")

    # ── 阶段3:metadata + quality ──
    print("③ metadata + quality 门禁 …")
    passed, failed = [], []
    for doc in kept:
        extract_metadata(doc)
        if assess_quality(doc)["passed"]:
            passed.append(doc)
        else:
            failed.append(doc)
    FAILED.mkdir(parents=True, exist_ok=True)
    for doc in failed:
        save_document(doc, folder=FAILED)  # 不合格的也留账
    print(f"   通过 {len(passed)},不合格 {len(failed)} → data/failed/")

    # ── 阶段4:chunk + 落盘 ──
    t = time.time()
    print("④ chunk + 落盘 …")
    total_chunks = n_parent = n_child = 0
    for doc in passed:
        save_document(doc)
        chunks = chunk_document(doc)
        save_chunks(doc.document_id, chunks)
        total_chunks += len(chunks)
        n_parent += sum(1 for c in chunks if c.parent_id is None)
        n_child += sum(1 for c in chunks if c.parent_id)
    print(f"   {total_chunks} chunk(父{n_parent} + 子{n_child}) ({time.time() - t:.0f}s)")

    # ── 对账 + 统计 ──
    print(f"\n{'═' * 66}\n对账")
    print(f"  manifest 在册(未排除): {len(manifest)}")
    print(f"  解析出 Document      : {len(docs)}")
    print(f"  - 近似重复丢弃        : {len(dropped)}")
    print(f"  - 质量不合格          : {len(failed)}")
    print(f"  = 入库                : {len(passed)}")
    print(f"  产出 chunk            : {total_chunks}(子块 {n_child} 个要 embedding)")

    print(f"\n元数据覆盖率(入库的 {len(passed)} 篇)")
    for k in ("title", "year", "doi", "keywords", "abstract"):
        got = sum(1 for d in passed if d.meta.get(k))
        print(f"  {k:9}: {got:4}/{len(passed)} = {got / max(len(passed), 1) * 100:5.1f}%")

    if UNKNOWN_TYPES:
        print("\n注意:遇到未知元素类型(已丢弃,但记了账 —— 别让新类型被静默吞掉)")
        for k, v in UNKNOWN_TYPES.most_common():
            print(f"  {k:12}: {v}")

    print("\n质量门禁毙掉的原因")
    reasons = Counter()
    for doc in failed:
        for k, v in assess_quality(doc)["checks"].items():
            if not v:
                reasons[k] += 1
    for k, v in reasons.most_common():
        print(f"  {k:12}: {v}")

    print(f"\n总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
