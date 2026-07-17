"""端到端 demo:MinerU 输出 → 整条管道 → 向量索引 → 真检索。

用法: uv run python scripts/demo_search.py
"""

import glob
import json
import sys
from pathlib import Path


from ruixue_agent.ingestion.pipeline import build_chunks, parse_and_clean
from ruixue_agent.rag.vector_store import VectorStore

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK,不改打不出中文

MINERU_OUT = r"D:/LLM_RuiXue/mineru-tool/test_out"

# ── 1. 读 MinerU 的输出,过 parse+clean ──
docs = []
for f in sorted(glob.glob(f"{MINERU_OUT}/*/auto/*_content_list.json")):
    if "v2" in f:
        continue
    name = Path(f).parent.parent.name
    meta = {
        "document_id": name[:14],
        "original_filename": name + ".pdf",
        "sha256": name[:16],
        "source": "期刊论文",
    }
    docs.append(parse_and_clean(json.load(open(f, encoding="utf-8")), meta))

# ── 2. 过 dedup → metadata → quality → chunk ──
chunks, stats = build_chunks(docs)
print("═══ 管道对账 ═══")
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n═══ 抽到的元数据 ═══")
for d in docs:
    print(f"  [{d.meta.get('year')}] {d.meta.get('title', '')[:34]}")

# ── 3. 建索引 ──
store = VectorStore()
store.index(chunks)
print(
    f"\n═══ 索引 ═══\n  子块(可被搜到) {len(store.children)} | 父块(备返回) {len(store.parents)}"
)

# ── 4. 真检索 ──
for q in ["PBAT地膜的水蒸气阻隔性怎么样", "生物降解地膜对辣椒生长有什么影响"]:
    print("\n" + "═" * 74)
    print(f"❓ {q}")
    for child, parent, score in store.search(q, k=2):
        where = " > ".join(child.section_path[-1:]) or "(卷首)"
        print(
            f"\n  ── 命中子块 相似度 {score:.3f} | 章节【{where}】| 第{child.page_start}页"
        )
        print(f"     {child.text[:70]}")
        print("  ── 返回给大模型的【父块(整节)】:")
        print(f"     {parent.text[:130]}…")
