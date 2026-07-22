"""人工/Claude 主导出题的落盘工具。

评测集不再由 LLM 批量生成,而是由具备领域理解的作者(Claude)逐块阅读真实
内容后亲手撰写,保证:
  - 每道有答案题都锚定其依据的真实 chunk(chunk_id 即标准答案,无需再判可答性)
  - user 题措辞口语化但答案确在库内(消除措辞偏差且不落空)
  - noans 题依据领域常识撰写,库内确实无答案(测拒答)

本脚本只做机械落盘:把作者提供的 (chunk_id, strategy, persona, question) 列表
与真实 chunk 绑定、拉取答案摘要、校验一致性,追加写入 evalset.jsonl。

用法:被批次脚本 import,调用 append_batch(items) / append_noans(items)。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sqlalchemy import text

from ruixue_agent.persistence.engine import get_engine

OUT = Path("data/eval/evalset.jsonl")
_LEAK = re.compile(r"段\s*[1-9]|根据(上|前)述|上文|前文|文中(所述|提到)")


def _load() -> list[dict]:
    if not OUT.exists():
        return []
    return [json.loads(x) for x in OUT.read_text(encoding="utf-8").splitlines()]


def _save(items: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for o in items:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def append_batch(items: list[tuple]) -> None:
    """追加有答案题。items: [(chunk_ids, strategy, persona, question), ...]

    chunk_ids 可为单个字符串(fact/user)或列表(multihop,多个依据块)。
    """
    existing = _load()
    seen_q = {o["question"] for o in existing}
    engine = get_engine()
    added, dup, leak = 0, 0, 0
    with engine.connect() as conn:
        for cids, strat, persona, q in items:
            if q in seen_q:
                dup += 1
                continue
            if _LEAK.search(q):
                leak += 1
                print(f"  ✗ 段号泄漏,跳过: {q[:40]}")
                continue
            ids = [cids] if isinstance(cids, str) else list(cids)
            row = conn.execute(
                text("""
                SELECT c.text, c.section_path, d.title, d.source
                FROM chunks c JOIN documents d ON d.document_id=c.document_id
                WHERE c.chunk_id=:c
            """),
                {"c": ids[0]},
            ).first()
            assert row, f"块不存在: {ids[0]}"
            existing.append({
                "question": q, "strategy": strat, "persona": persona,
                "primary_gold": ids[0], "gold_chunk_ids": ids, "has_answer": True,
                "gold_text": row[0][:200], "section_path": row[1],
                "title": row[2], "source": row[3], "author": "claude",
            })
            seen_q.add(q)
            added += 1
    _save(existing)
    print(f"  有答案题 +{added}(重复跳过 {dup},泄漏跳过 {leak}),累计 {len(existing)}")


def append_noans(items: list[tuple]) -> None:
    """追加无答案题。items: [(persona, question), ...]"""
    existing = _load()
    seen_q = {o["question"] for o in existing}
    added = 0
    for persona, q in items:
        if q in seen_q:
            continue
        existing.append({
            "question": q, "strategy": "noans", "persona": persona,
            "primary_gold": None, "gold_chunk_ids": [], "has_answer": False,
            "author": "claude",
        })
        seen_q.add(q)
        added += 1
    _save(existing)
    print(f"  拒答题 +{added},累计 {len(existing)}")


def summary() -> None:
    from collections import Counter

    items = _load()
    print(f"\n评测集共 {len(items)} 道")
    for s, c in Counter(o["strategy"] for o in items).most_common():
        print(f"  {s:9} {c}")
