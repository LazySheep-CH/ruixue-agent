"""把 PG 里的子块 embed 后灌进 Milvus。

数据从PG读,不是从 JSON 读 —— PG 才是 source of truth。
JSON 是管道的中间缓存,可能和库里不一致。

支持断点续传:挂了直接重跑,已灌的跳过(连 embed 都省了)。
Milvus 没有事务 —— 挂了不会回滚、会留下半截数据,幂等是唯一的恢复手段。

用法:
    uv run python scripts/load_milvus.py            # 灌(可重复跑)
    uv run python scripts/load_milvus.py --restart  # 推倒重来
"""

import sys
import time

from sqlalchemy import text

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.rag.milvus_store import MilvusVectorStore

sys.stdout.reconfigure(encoding="utf-8")

_READ_BATCH = 5000  # 一次从 PG 读多少条


def main() -> None:
    store = MilvusVectorStore()
    if "--restart" in sys.argv:
        print("--restart:删掉 collection 重来")
        store.drop()
    store.ensure_collection(index_type="FLAT")

    engine = get_engine()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM chunks WHERE kind='child'")).scalar()
    print(f"PG 里有 {total:,} 个子块要 embed")
    print(f"Milvus 里已有 {store.count():,} 个 —— 这些会跳过\n")

    t0 = time.time()
    done = skipped = 0
    offset = 0

    while offset < total:
        # 从 PG 读一批。join documents 拿 year/source ——
        # 这两个字段要复制一份进 Milvus 做前过滤(它们属于索引,不是第二份数据源)
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text("""
                SELECT c.chunk_id, c.text, d.year, d.source
                FROM chunks c JOIN documents d ON d.document_id = c.document_id
                WHERE c.kind = 'child'
                ORDER BY c.chunk_id          -- 固定顺序,分页才不会漏/重
                LIMIT :lim OFFSET :off
            """),
                    {"lim": _READ_BATCH, "off": offset},
                )
                .mappings()
                .all()
            )
        offset += len(rows)
        if not rows:
            break

        # 断点续传:先问 Milvus 哪些已经有了,跳过 —— 省下的是 embed 的算力
        have = store.existing_ids([r["chunk_id"] for r in rows])
        todo = [dict(r) for r in rows if r["chunk_id"] not in have]
        skipped += len(rows) - len(todo)

        if todo:
            store.index(todo)
            done += len(todo)

        el = time.time() - t0
        rate = done / el if done and el else 0
        eta = (total - offset) / rate / 60 if rate else 0
        print(
            f"  {offset:>7,}/{total:,}  已灌 {done:>7,}  跳过 {skipped:>7,}"
            f"  {rate:>5.0f} 条/秒  剩约 {eta:>4.1f} 分钟"
        )

    print(f"\n灌完:{done:,} 条(跳过 {skipped:,}),耗时 {time.time() - t0:.0f}s")
    print(f"Milvus 现有:{store.count():,}")
    print(f"对账:PG 子块 {total:,}  ==  Milvus {store.count():,} ?")


if __name__ == "__main__":
    main()
