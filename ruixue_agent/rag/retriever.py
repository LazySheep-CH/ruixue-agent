"""Retriever:把 Milvus 和 PG 拼起来,还给上层一个完整答案。

    MilvusVectorStore  只知道"哪些 chunk_id 像"     (向量的事)
    PgRepository       只知道"chunk_id 的文本是啥"   (数据的事)
    Retriever          把两边接上,并处理接缝上的问题  ← 这个文件

为什么单独一层,不塞进 MilvusVectorStore:
    那样 Milvus 那个类就得认识 PG,两个本该独立的东西缠在一起。
    Retriever 只依赖【接口】(有 search 的东西、有 get_parents 的东西),
    所以测试能塞假的进去 —— 不连任何数据库就能测编排逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

from ruixue_agent.rag.fusion import rrf

# 超取倍数:要 k 个父块,就向 Milvus 要 k * _FANOUT 个子块。
# 为什么要超取见 search() 里的说明。
# 3 是个拍脑袋的起点 —— 等有了评测集,用【数据】定这个值,别信我这个数。
_FANOUT = 3

# RRF 里 BM25 那一路的权重(向量固定 1.0)。
#
# ★ 这个值不是"精确最优",是【一个平台区里的合理选择】。用企业标准的方法选的
#   (交叉验证 + 自助置信区间,见 scripts/tune_weight.py),结论:
#
#   ① 权重 0.2~0.7 的 R@3 全在 0.82~0.833 之间 —— 系统对权重【不敏感】,
#      是个平台不是尖峰。所以选 0.3 还是 0.5 没有统计意义上的区别。
#   ② 5 折交叉验证各自选出的"最优"是 [0.35,0.65,0.35,0.35,0.25] —— 到处跳,
#      证明"全量最优 0.4"是在拟合这 150 题的噪声,不能当真常数。
#   ③ 但【混合优于纯向量】是显著的:自助 1000 次重抽,R@3 提升的 95% 置信区间
#      [+0.027,+0.127] 整个在 0 以上。这个结论可信,权重的精确值不可信。
#   ④ 权重 >1.0 是悬崖:1.1 时 R@3 从 0.827 崩到 0.627(弱的那路话语权过大)。
#
# 取 0.3:在平台区内、离悬崖远、偏保守(向量是更强的那路,让它主导)。
# 换语料/换成真实问题都要重跑 tune_weight.py —— 这个值不跨数据集通用。
_BM25_WEIGHT = 0.3


@dataclass
class Hit:
    """还给上层(最终是 LLM)的一条结果。

    为什么不直接还 ChunkRow:
        ChunkRow 是【数据库的形状】(带 text_tsv、created_at 这些 LLM 不关心的东西)。
        Hit 是【检索结果的形状】—— 多了 score,少了数据库杂物。
        这和 Repository 那层的思想一样:别让底层的形状泄漏到上层。
    """

    chunk_id: str
    text: str  # 父块正文 —— 给 LLM 看的
    score: float  # 相关度
    document_id: str  # 出处:哪篇
    section_path: list[str]  # 出处:哪一节
    page_start: int
    page_end: int


class Retriever:
    def __init__(
        self, store, repo, bm25=None, weights=(1.0, _BM25_WEIGHT), reranker=None
    ) -> None:
        """store: 有 .search() 的东西    repo: 有 .get_chunks()/.get_parents() 的东西
        bm25:     可选。给了就走【混合检索】(向量 + BM25,RRF 融合)
        reranker: 可选。给了就在最后加一道【精排】(cross-encoder 重排候选)

        每个可选组件默认 None = 不启用,行为和没加它之前完全一致。
        这样才能用同一把尺子做 A/B:每一层都是"开 vs 不开,只差一个参数"。
        这是能【逐层量化贡献】的前提 —— 检索多少分、混合加多少、rerank 再加多少。
        """
        self.store = store
        self.repo = repo
        self.bm25 = bm25
        self.weights = weights  # (向量权重, BM25权重)
        self.reranker = reranker

    def search(
        self,
        query: str,
        k: int = 3,
        year_min: int | None = None,
        source: str | None = None,
    ) -> list[Hit]:
        """查询 → 命中子块 → 换成父块 → 还给上层。

        整个函数就四步,每步对应一组测试。
        """
        # ① 向 store 要子块 —— 要【k * _FANOUT】个,不是 k 个。
        #
        #    为什么超取:检索的粒度是【子块】,交付的粒度是【父块】,两者多对一。
        #    只要 3 个子块的话,它们可能全来自同一个父块(同一节里三句话都相关)
        #    → 去重完只剩 1 个,而用户要 3 个。
        #
        #    year_min/source 原样传下去:前过滤必须发生在 Milvus【内部】。
        #    如果在这儿自己筛(后过滤),就可能筛完剩 0 条。
        hits = self.store.search(query, k=k * _FANOUT, year_min=year_min, source=source)
        if not hits:
            return []

        # ② 记下每个子块的分数,后面第 ③ 步要用。
        #    hits 是 [("P1_c0", 0.82), ...] → {"P1_c0": 0.82, ...}
        score_by_child = {cid: s for cid, s in hits}

        # ③ 子块 → 父块,同时算出每个父块的分数。
        #
        #    ★ 这一步是整个函数的核心,一个 dict 干了三件事:
        #        去重      —— 同一个父块只留一个 key
        #        保序      —— Python 3.7+ 的 dict 保留【插入顺序】,
        #                     而 get_chunks 保证按相似度顺序还子块,
        #                     所以父块的顺序 = 它最相关子块的排名 = 正确的排序
        #        取最高分  —— 父块自己没有分数(Milvus 只给子块打分)。
        #                     P1 命中 0.82 和 0.71 两个子块 → 取 0.82。
        #                     为什么不取平均:那会惩罚长节 ——
        #                     长节里必然混着很多不相关的句子,不公平。
        #
        #    注意这里用 get_chunks 而不是 get_parents:我们需要子块身上的
        #    parent_id 来建立"哪个子块属于哪个父块"的对应关系,而 get_parents
        #    把这个信息扔掉了。(见文件末尾的备注)
        children = self.repo.get_chunks(list(score_by_child))
        best_score: dict[str, float] = {}
        for c in children:
            if c.parent_id is None:
                continue
            s = score_by_child[c.chunk_id]
            if s > best_score.get(c.parent_id, float("-inf")):
                best_score[c.parent_id] = s

        # ④ 排名:显式按分数排序。
        #
        #    ⚠ 不能依赖 dict 的插入顺序碰巧是对的。
        #      纯向量路径下 best_score 的插入顺序确实等于排名(因为 get_chunks 保序),
        #      但那是个【隐形契约】—— 下面加了 RRF 之后就不成立了。
        #      靠运气对齐的代码,改一处就会静默出错(而且不报错,只是答案变差)。
        ranked = sorted(best_score.items(), key=lambda kv: -kv[1])

        # ⑤ 混合检索:BM25 也来投一票,用 RRF 融合两边的【排名】。
        #
        #    为什么在这一步融合,而不是在 ① 那里:
        #    向量侧检索的是【子块】,BM25 检索的是【父块】—— 粒度不同,没法直接合。
        #    等向量侧折算成父块之后,两边才在同一个粒度上,才谈得上融合。
        #
        #    为什么用 RRF 而不是加权分数:向量给余弦(0.4~0.9),BM25 给 ts_rank(0.0x),
        #    量纲和分布完全不同,直接相加等于让尺度大的那方说了算。
        #    RRF 只看排名,不看分数 —— 详见 fusion.py。
        if self.bm25 is not None:
            lex_ranking = self.bm25.search(
                query, k=k * _FANOUT, year_min=year_min, source=source
            )
            ranked = rrf([ranked, lex_ranking], weights=list(self.weights))

        # ⑥ 取父块正文。
        #    get_chunks 按【给定顺序】返回,并跳过查不到的 id:
        #    索引指向 PG 里已删的块时不崩,这是正常的最终一致,不是错误。
        parents = self.repo.get_chunks([cid for cid, _ in ranked])

        # ⑦ rerank:最后一道精排(可选)。
        #
        #    为什么在【取完正文之后】做:cross-encoder 要把 query 和文档拼起来打分,
        #    必须有文本。前面的检索都只在 chunk_id 层面工作,到这里才有正文。
        #
        #    为什么在【截到 k 之前】做:rerank 的价值就是重排 —— 把检索排在
        #    第 5、第 8 的正确答案顶到前面。如果先截到 k 再 rerank,那些候选
        #    就已经被扔了,rerank 无米下锅。所以这里 parents 是超取来的
        #    (k * _FANOUT ≈ 30 个),rerank 完再截。
        #
        #    rerank 只能重排检索捞出来的 —— 找不到检索漏掉的。天花板是检索的召回率。
        if self.reranker is not None and parents:
            new_order = self.reranker.rerank(
                query, [(p.chunk_id, p.text) for p in parents]
            )
            by_id = {p.chunk_id: p for p in parents}
            best_score = dict(new_order)
            parents = [by_id[cid] for cid, _ in new_order]
        else:
            best_score = dict(ranked)

        # ⑧ 包成 Hit,截到 k 个。
        #    Hit 带 document_id / section_path 是【出处】:LLM 光有文本不够,
        #    用户要能核实"这话哪儿来的"。地膜标准答错要赔钱,没出处不能上生产。
        return [
            Hit(
                chunk_id=p.chunk_id,
                text=p.text,
                score=best_score.get(p.chunk_id, 0.0),
                document_id=p.document_id,
                section_path=p.section_path,
                page_start=p.page_start,
                page_end=p.page_end,
            )
            for p in parents
        ][:k]


# ── 评审八股 ──────────────────────────────────────────
# ① Small-to-Big 为什么有效?
#    检索要【短】才准(短文本向量语义集中,长文本被平均掉);
#    生成要【长】才全(LLM 需要上下文)。两个需求相反 →
#    用短的去搜、还长的回来,各取所长。
# ② 为什么要超取(fanout)?
#    检索的粒度是子块,交付的粒度是父块,两者是多对一 →
#    k 个子块 ≠ k 个父块。不超取就可能还不够数。
# ③ 为什么 Retriever 不自己 new 一个 MilvusVectorStore?
#    依赖注入 —— 依赖接口不依赖实现。换成别的向量库、或测试塞假的,
#    这个文件都不用改。
