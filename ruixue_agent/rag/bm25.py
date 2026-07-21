"""词法检索(lexical search)—— 靠罕见词精确定位,补向量检索的短板。

═══ 核心洞察:只用罕见词查 ═══

BM25 靠 IDF 定位 —— 一个词越罕见,它越能锁定文档。但把罕见词和常见词
一起 OR,常见词会把候选灌到几万,罕见词的定位能力被【淹没】。

实测(40 题评测集):
    全部词 OR         R@10 0.350   MRR 0.171   8497 ms
    只留 DF<1% 的词   R@10 0.525   MRR 0.217    140 ms   ← 召回 +50%,快 60 倍
    只留 DF<0.3%      R@10 0.525   MRR 0.272    513 ms

为什么快 60 倍:延迟正比于候选数,候选数正比于查询词的常见程度。
    「GB/T 35795 对断裂标称应变的规定」   候选    825 块    100 ms
    「残膜对土壤微生物有什么影响」        候选 22,468 块   2680 ms
BM25 恰好【在它有用的时候快,在它没用的时候慢】—— 挑掉常见词等于
同时拿到速度和准确率,不是权衡。

═══ 我在这上面栽过三次(留档,别重蹈)═══

① 用 plainto_tsquery —— 它把所有词用 AND 连接。长问句要求 13 个词全部命中
   (包括"的""是""多少"),真实查询几乎必然 0 结果。词法检索是 OR + 排序,
   不是布尔 AND。
② 为压延迟加了 `LIMIT 3000` 却没有 ORDER BY —— 取的是【任意】3000 个候选,
   不是最相关的。R@10 因此从 0.350 掉到 0.250。
③ 用启发式挑词(有没有数字、词长)而不是真实文档频率。
   并且误判"这些查询里没有罕见词" —— 我只统计了索引里【完全没有】的词(df=0),
   漏掉了 df=50/39396(0.13%)这种明明很罕见的。实测 94% 的查询都含极罕见词。

三个 bug 叠起来,让我一度得出"词法检索在这个语料上无效"的结论。
教训:先用最朴素的实现量天花板,再优化。我反过来干了。

═══ 依赖 ═══
    chunks.text_tokens   jieba 分词(migration 0002 + scripts/backfill_tokens.py)
    term_df              词频表(migration 0003 + scripts/build_term_df.py)
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from ruixue_agent.rag.text_segment import tokenize

# ts_rank_cd 的 normalization 位掩码:
#   2  = 除以文档长度的对数     ← 长文档不因为词多而占便宜
#   32 = rank/(rank+1)          ← 压到 0~1
# 这两位一起用,行为最接近标准 BM25。
_NORM = 2 | 32

# 只用文档频率低于这个比例的词。实测 1% 和 0.3% 召回相同(0.525),
# 0.3% 的 MRR 更好(0.272 vs 0.217)但慢 3.7 倍。取 1% 是速度优先的选择 ——
# 因为它要和 190ms 的向量检索并行跑,不能拖后腿。
_MAX_DF_PCT = 1.0

# 查询里全是常见词时,至少保留最罕见的几个(而不是退回"全部词")。
# 踩过:fallback 用全部词,导致严格阈值反而更慢(触发 fallback 更频繁)。
_MIN_TERMS = 3

# 停用词:纯语法功能词,任何查询里都没有检索价值。
# 领域高频词(地膜 44%、土壤 38.5%)【不放这里】—— 它们由 DF 阈值自动过滤,
# 而且用户真会问"地膜厚度",硬删语义就残了。
_STOP = frozenset(
    "的 了 是 在 和 与 及 或 等 中 上 下 对 为 以 于 由 而 也 都 就 要 有 无 "
    "个 之 其 此 该 后 前 较 并 且 但 则 从 到 被 把 让 使 给 向 更 最 很 "
    "我 你 他 我们 什么 多少 怎么 如何 哪些 哪个 哪 请问 是否 能否 吗 呢 "
    "一个 一种 这个 那个 以及 或者 什么样 多大 多长 多久".split()
)


class Bm25Search:
    """PG 全文检索。只还 (chunk_id, 分数) —— 和 MilvusVectorStore 同一个契约。"""

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
        """挑出罕见词,拼成 OR 查询。这是整个文件的关键。"""
        tokens = list(
            dict.fromkeys(
                t for t in tokenize(query).split() if t not in _STOP and len(t) > 1
            )
        )
        if not tokens:
            return ""

        # 一次查回所有词的文档频率。索引里没有的词 = df 0 = 最罕见
        df = dict(
            self.session.execute(
                text("SELECT term, ndoc FROM term_df WHERE term = ANY(:t)"),
                {"t": tokens},
            ).all()
        )
        cutoff = self._parent_count() * self.max_df_pct / 100
        keep = [t for t in tokens if df.get(t, 0) <= cutoff]

        # 全是常见词 → 退而求其次:取最罕见的几个。
        # 不能退回"全部词",那会把候选灌爆(踩过)。
        if not keep:
            keep = sorted(tokens, key=lambda t: df.get(t, 0))[:_MIN_TERMS]

        # 每个词单引号包起来 —— 词里可能有 / . - 这些 tsquery 语法字符
        # (标准号 gb/t35795-2017),不引会解析报错
        return " | ".join("'" + t.replace("'", "''") + "'" for t in keep)

    def search(
        self,
        query: str,
        k: int = 10,
        year_min: int | None = None,
        source: str | None = None,
    ) -> list[tuple[str, float]]:
        """查父块,按 BM25 分数降序。

        ⚠ 查询串和索引必须用【同一个分词器】—— 索引里存的是 "地膜 厚度",
          查询传 "地膜厚度" 就匹配不到。这是全文检索的铁律。
        """
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

        # 全量打分再排序 —— 不截候选。
        # 挑罕见词已经把候选压到几百~几千,不需要再截(截了还会漏掉高分的)。
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
