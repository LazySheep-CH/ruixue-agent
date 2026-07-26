"""检索编排:组合向量检索(Milvus)、词法检索(BM25)与重排,输出带出处的结果。

MilvusVectorStore 只回答"哪些块相似",PgRepository 只负责按 ID 取文本,
本层把两者接起来并处理接缝:子块到父块的换算、多路融合、精排、截断。
仅依赖鸭子类型接口(有 search / get_chunks 的对象即可),测试可注入伪实现,
不连库即可覆盖全部编排逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

from ruixue_agent.rag.fusion import rrf

# 超取倍数:交付 k 个父块需要检索 k * _FANOUT 个子块。
# 检索粒度是子块、交付粒度是父块,多对一 —— 命中的子块可能集中在同一父块,
# 不超取则去重后不足 k 个。=2 由 fanout 扫描定:候选太多会淹没 rerank 精排,
# 2 是甜点(Recall 不掉、rerank 计算省近半),见 docs/检索优化记录.md。
_FANOUT = 2

# RRF 中词法路的权重(向量路固定 1.0)。由交叉验证 + bootstrap 置信区间选定
# (scripts/tune_weight.py),150 题评测集上的结论:
#   - 0.2~0.7 区间内 R@3 稳定在 0.82~0.833,是平台而非尖峰,取值不敏感;
#   - 五折交叉验证各折最优在 0.25~0.65 之间波动,说明单点"最优值"是噪声;
#   - "混合优于纯向量"本身显著:R@3 提升 95% CI [+0.027, +0.127],不含 0;
#   - 权重 >1.0 后 R@3 从 0.827 跌至 0.627 —— 弱路(词法单独 R@10 0.525)
#     不应与强路(向量 0.853)等权。
# 取平台区内偏保守的 0.3。语料或查询分布变化后需重新标定。
_BM25_WEIGHT = 0.3


@dataclass
class Hit:
    """单条检索结果。

    不直接暴露 ChunkRow:那是数据库行的形状(含 text_tsv、created_at 等
    检索消费方不关心的字段)。Hit 是检索结果的形状 —— 正文、相关度、出处。
    """

    chunk_id: str
    text: str  # 父块正文,交给生成层
    score: float  # 相关度
    document_id: str  # 出处:文档
    section_path: list[str]  # 出处:章节路径
    page_start: int
    page_end: int


class Retriever:
    def __init__(
        self,
        store,
        repo,
        bm25=None,
        weights=(1.0, _BM25_WEIGHT),
        reranker=None,
        rewriter=None,
    ) -> None:
        """store: 向量检索(须有 .search());repo: 文本存取(须有 .get_chunks())。

        bm25、reranker、rewriter 均为可选组件,None 即关闭,行为与未引入前完全一致。
        每层能力都是"开/关只差一个参数",评测时可逐层 A/B、单独量化各层增益。
        rewriter 带闸门只改口语查询,实测 338 题 user R@3 +0.060、fact/multihop 零误伤。
        """
        self.store = store
        self.repo = repo
        self.bm25 = bm25
        self.weights = weights  # (向量权重, 词法权重)
        self.reranker = reranker
        self.rewriter = rewriter

    def search(
        self,
        query: str,
        k: int = 3,
        year_min: int | None = None,
        source: str | None = None,
    ) -> list[Hit]:
        """检索并返回 top-k 父块,附相关度与出处。"""
        # 0. 查询改写(可选):口语→术语再检索,补 embedding 对口语的弱势。改写只用于
        #    【检索】(向量+词法);重排仍喂【原始问题】—— 重排匹配的是用户真实意图,
        #    不是改写后的措辞。改写器自带闸门(术语题原样放过),失败自动回退原查询。
        retr_query = self.rewriter.rewrite(query) if self.rewriter else query

        # 1. 向量检索子块,按超取倍数放大。过滤条件透传给 Milvus 做前过滤,
        #    在本层筛(后过滤)可能把候选筛空。
        hits = self.store.search(retr_query, k=k * _FANOUT, year_min=year_min, source=source)
        if not hits:
            return []

        score_by_child = {cid: s for cid, s in hits}

        # 2. 子块折算父块:同父去重,父块得分取其命中子块的最高分。
        #    不取平均 —— 长章节必然混有无关句子,均值会系统性惩罚长父块。
        #    这里用 get_chunks 而非 get_parents,因为需要子块的 parent_id
        #    来建立归属关系。
        children = self.repo.get_chunks(list(score_by_child))
        best_score: dict[str, float] = {}
        for c in children:
            if c.parent_id is None:
                continue
            s = score_by_child[c.chunk_id]
            if s > best_score.get(c.parent_id, float("-inf")):
                best_score[c.parent_id] = s

        # 3. 显式按分数排序。不依赖 dict 插入顺序恰好等于排名 —— 纯向量路径下
        #    碰巧成立,但引入 RRF 后即失效,隐式约定会静默产生错误排序。
        ranked = sorted(best_score.items(), key=lambda kv: -kv[1])

        # 4. 词法路融合(可选)。融合放在折算父块之后:向量路检索子块、
        #    词法路检索父块,粒度对齐后才可合并。
        #    用 RRF 而非分数加权:余弦(0.4~0.9)与 ts_rank(0.0x)量纲不同,
        #    直接相加等于让量纲大的一方主导。RRF 只比较排名,见 fusion.py。
        if self.bm25 is not None:
            lex_ranking = self.bm25.search(
                retr_query, k=k * _FANOUT, year_min=year_min, source=source
            )
            ranked = rrf([ranked, lex_ranking], weights=list(self.weights))

        # 5. 取父块正文。get_chunks 按给定顺序返回,查不到的 ID 跳过
        #    (索引与 PG 短暂不同步属正常最终一致,不视为错误)。
        parents = self.repo.get_chunks([cid for cid, _ in ranked])

        # 6. 重排(可选)。cross-encoder 需要正文,故必须在取文之后;
        #    且必须在截断之前 —— 重排的价值正是把检索排在 k 之外的正确
        #    答案提上来,先截断则无候选可提。重排只作用于已召回的候选,
        #    上限受检索召回率约束。
        if self.reranker is not None and parents:
            new_order = self.reranker.rerank(query, [(p.chunk_id, p.text) for p in parents])
            by_id = {p.chunk_id: p for p in parents}
            best_score = dict(new_order)
            parents = [by_id[cid] for cid, _ in new_order]
        else:
            best_score = dict(ranked)

        # 7. 组装并截断。出处字段(document_id / section_path / 页码)必须携带:
        #    生成层要标引用,使用方要能核实答案来源。
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
