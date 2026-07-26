"""回填 chunks.text_tokens —— 给已有的 26 万块补上 jieba 分词。

为什么是独立脚本而不是写在 migration 里:
    migration 管【结构】,数据迁移管【内容】。26 万行的回填跑在 migration 里,
    中途挂了整个 migration 事务回滚 —— 前面跑的全白费。
    独立脚本可以分批提交、可以断点续传、可以重跑。

幂等:只处理 text_tokens IS NULL 的行,重跑自动跳过已完成的。

用法:
    uv run python scripts/backfill_tokens.py
    uv run python scripts/backfill_tokens.py --redo   # 全部重算(改了词典后用)
"""

import argparse
import sys
import time

from sqlalchemy import text

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.rag.text_segment import tokenize

sys.stdout.reconfigure(encoding="utf-8")

_BATCH = 5000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true", help="全部重算(改了领域词典后用)")
    args = ap.parse_args()

    engine = get_engine()

    # --redo:先把 text_tokens 全清空,再走"NULL 才处理"的正常路径。
    #
    # 不能用 "WHERE TRUE + LIMIT 分页" 重算,已知陷阱:
    #   分页没有 OFFSET,每批都从头查符合条件的块;第一批更新后那些块仍满足
    #   WHERE TRUE,下一批又查到它们 → 有的块被重复处理、有的从没被碰。
    #   现象:处理数超过总数(265,000/262,782)。
    #   正确做法是让"已处理"这件事改变 WHERE 的判定 —— 清空后用 text_tokens IS NULL,
    #   处理一个就少一个,天然收敛、不重不漏。
    if args.redo:
        with engine.begin() as conn:
            conn.execute(text("UPDATE chunks SET text_tokens = NULL"))
        print("--redo:已清空 text_tokens,全部重算")

    with engine.connect() as conn:
        todo = conn.execute(text("SELECT count(*) FROM chunks WHERE text_tokens IS NULL")).scalar()
    print(f"待分词 {todo:,} 块")
    if not todo:
        print("没有要做的。")
        return

    t0 = time.time()
    done = 0
    while True:
        with engine.begin() as conn:
            # text_tokens IS NULL:处理一个就少一个,天然收敛(不用 OFFSET,不重不漏)
            rows = conn.execute(
                text("""
                SELECT chunk_id, text FROM chunks
                WHERE text_tokens IS NULL
                ORDER BY chunk_id LIMIT :lim
            """),
                {"lim": _BATCH},
            ).all()
            if not rows:
                break

            # 分词在 Python 侧;写回后触发器自动把它转成 tsvector
            payload = [{"cid": cid, "tok": tokenize(txt)} for cid, txt in rows]
            conn.execute(
                text("UPDATE chunks SET text_tokens = :tok WHERE chunk_id = :cid"),
                payload,
            )

        done += len(rows)
        el = time.time() - t0
        rate = done / el
        print(
            f"  {done:>7,}/{todo:,}  {rate:>6.0f} 块/秒  剩约 {(todo - done) / rate / 60:>4.1f} 分钟"
        )

    print(f"\n完成 {done:,} 块,耗时 {time.time() - t0:.0f}s")

    # ── 验收:触发器真的生效了吗 ──
    with engine.connect() as conn:
        n_tok = conn.execute(
            text("SELECT count(*) FROM chunks WHERE text_tokens IS NOT NULL")
        ).scalar()
        n_tsv = conn.execute(
            text("SELECT count(*) FROM chunks WHERE text_tsv IS NOT NULL AND text_tsv != ''")
        ).scalar()
        print(f"  有分词的: {n_tok:,}")
        print(f"  有 tsv 的: {n_tsv:,}   ← 该和上面一样(触发器自动生成的)")

        # 真查一把 —— 这是 0001 时【完全查不出来】的
        for q in ("地膜 厚度", "PBAT", "全生物降解地膜"):
            hit = conn.execute(
                text("""
                SELECT count(*) FROM chunks
                WHERE text_tsv @@ plainto_tsquery('simple', :q)
            """),
                {"q": tokenize(q)},
            ).scalar()
            print(f"  搜「{q}」命中 {hit:,} 块")


if __name__ == "__main__":
    main()
