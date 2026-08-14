"""词法检索:PostgreSQL 全文索引 + BM25 类打分,补语义检索在精确匹配上的短板。

核心策略是查询词先按文档频率过滤,只保留罕见词(DF < 1%)再做 OR 检索。
BM25 依赖 IDF 区分文档,常见词会把候选集撑到数万行、逐行打分拖垮延迟,
同时淹没罕见词的区分度。实测(150 题评测集):
    全部词 OR       R@10 0.350   8497 ms
    只留 DF<1%      R@10 0.525    140 ms
延迟正比于候选数,候选数正比于查询词的常见程度 —— 剔除常见词同时改善
召回与延迟,不是取舍。

实现注意:
- 查询语义必须是 OR + 排序。plainto_tsquery 是 AND 语义,长查询要求全部
  词共现,实际问句几乎必然 0 命中。
- 候选不做 LIMIT 截断:截断若无对应排序,取到的是任意子集而非高分子集。
  罕见词过滤已把候选压到千级以内,无需再截。
- 查询串与索引使用同一分词器(text_segment.tokenize)。

依赖:chunks.text_tokens(migration 0002)、term_df 词频表(migration 0003,
语料更新后需重建,见 scripts/build_term_df.py)。
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from ruixue_agent.rag.text_segment import tokenize

# ts_rank_cd 归一化位掩码:
#   2  = 除以文档长度的对数(长文档不因词多占优)
#   32 = rank/(rank+1),压缩到 0~1
_NORM = 2 | 32

# 文档频率上限(百分比)。实测 1% 与 0.3% 召回相同(R@10 0.525),
# 0.3% 的 MRR 略好但慢 3.7 倍;词法路与约 190ms 的向量路并行,取 1% 保延迟。
_MAX_DF_PCT = 1.0

# 查询词全为常见词时,保底取最罕见的几个。
# 不能回退为"全部词"—— 那会让阈值越严、回退越频繁,反而更慢。
_MIN_TERMS = 3

# 停用词:纯语法功能词。领域高频词(如"地膜"出现于 44% 的父块)不放这里,
# 由 DF 阈值统一处理 —— 硬编码剔除会伤害"地膜厚度"这类正常查询的语义。
_STOP = frozenset(
    "的 了 是 在 和 与 及 或 等 中 上 下 对 为 以 于 由 而 也 都 就 要 有 无 "
    "个 之 其 此 该 后 前 较 并 且 但 则 从 到 被 把 让 使 给 向 更 最 很 "
    "我 你 他 我们 什么 多少 怎么 如何 哪些 哪个 哪 请问 是否 能否 吗 呢 "
    "一个 一种 这个 那个 以及 或者 什么样 多大 多长 多久".split()
)


class Bm25Search:
    """PG 全文检索。返回 (chunk_id, 分数),与 MilvusVectorStore 同一契约。"""

    def __init__(self, session: Session, max_df_pct: float = _MAX_DF_PCT) -> None:
        self.session = session
        self.max_df_pct = max_df_pct
        self._total: int | None = None  # 父块总数,懒加载

    def _parent_count(self) -> int:
        if self._total is None:
            self._total = self.session.execute(
                text("SELECT count(*) FROM chunks WHERE kind='parent'")
            ).scalar()
        return self._total

    def _rare_terms_query(self, query: str) -> str:
        """挑出罕见词,拼成 tsquery 的 OR 表达式。"""
        tokens = list(
            dict.fromkeys(t for t in tokenize(query).split() if t not in _STOP and len(t) > 1)
        )
        if not tokens:
            return ""

        # 词频表中不存在的词按 DF=0 处理(未见过的词恰是区分度最高的)
        df = dict(
            self.session.execute(
                text("SELECT term, ndoc FROM term_df WHERE term = ANY(:t)"),
                {"t": tokens},
            ).all()
        )
        cutoff = self._parent_count() * self.max_df_pct / 100
        keep = [t for t in tokens if df.get(t, 0) <= cutoff]

        if not keep:
            keep = sorted(tokens, key=lambda t: df.get(t, 0))[:_MIN_TERMS]

        # 词内可能含 / . - 等 tsquery 语法字符(标准号 gb/t35795-2017),须加引号
        return " | ".join("'" + t.replace("'", "''") + "'" for t in keep)

    def search(
        self,
        query: str,
        k: int = 10,
        year_min: int | None = None,
        source: str | None = None,
    ) -> list[tuple[str, float]]:
        """检索父块,按 BM25 分数降序返回 (chunk_id, 分数)。"""
        tq = self._rare_terms_query(query)
        if not tq:
            return []

        conds = ["c.kind = 'parent'", "c.text_tsv @@ q"]
        params: dict = {"q": tq, "k": k}
        if year_min is not None:
            conds.append("d.year >= :year_min")
            params["year_min"] = year_min
        if source is not None:
            conds.append("d.source = :source")
            params["source"] = source

        rows = self.session.execute(
            text(f"""
            SELECT c.chunk_id, ts_rank_cd(c.text_tsv, q, {_NORM}) AS score
            FROM chunks c
            JOIN documents d ON d.document_id = c.document_id,
                 to_tsquery('simple', :q) AS q
            WHERE {" AND ".join(conds)}
            ORDER BY score DESC
            LIMIT :k
        """),
            params,
        ).all()
        return [(r[0], float(r[1])) for r in rows]
