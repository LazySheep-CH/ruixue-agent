"""建词频表 —— BM25 挑词的依据。

为什么需要:词法检索的成败取决于只用罕见词查。实测(40 题):
    全部词 OR       R@10 0.350   8497 ms
    只留 DF<1% 的   R@10 0.525    140 ms    ← 召回 +50%,快 60 倍
一个罕见词和七个常见词做 OR,常见词会把候选灌到几万,罕见词被淹没。

PG 的 ts_stat 能算文档频率,但它是全表扫,不能每次查询都跑 → 物化成表。

派生数据,和索引一个性质:丢了重跑就有。语料变了要重跑。

用法:
    uv run python scripts/build_term_df.py
"""

import sys
import time

from sqlalchemy import text

from ruixue_agent.persistence.engine import get_engine

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    engine = get_engine()
    t0 = time.time()

    with engine.begin() as conn:
        total = conn.execute(text("SELECT count(*) FROM chunks WHERE kind='parent'")).scalar()
        print(f"统计 {total:,} 个父块的词频(ts_stat 全表扫,要等一下)…")

        conn.execute(text("TRUNCATE term_df"))
        conn.execute(
            text("""
            INSERT INTO term_df (term, ndoc)
            SELECT word, ndoc
            FROM ts_stat('SELECT text_tsv FROM chunks WHERE kind=''parent''')
        """)
        )

    with engine.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM term_df")).scalar()
        print(f"  {n:,} 个词,耗时 {time.time() - t0:.0f}s\n")

        print("最高频的词(它们几乎没有区分度):")
        for w, nd in conn.execute(
            text("SELECT term, ndoc FROM term_df ORDER BY ndoc DESC LIMIT 8")
        ):
            print(f"  {w:<10} {nd:>6,} 块 ({nd / total * 100:>4.1f}%)")

        print("\n分布(BM25 只该用左边这些):")
        for lo, hi, label in (
            (0, 0.1, "DF < 0.1%   极罕见"),
            (0.1, 0.5, "0.1~0.5%    罕见"),
            (0.5, 2, "0.5~2%      较罕见"),
            (2, 100, "> 2%        常见(该丢掉)"),
        ):
            c = conn.execute(
                text("SELECT count(*) FROM term_df WHERE ndoc >= :lo AND ndoc < :hi"),
                {"lo": total * lo / 100, "hi": total * hi / 100},
            ).scalar()
            print(f"  {label:<22} {c:>7,} 个词")


if __name__ == "__main__":
    main()
