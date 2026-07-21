"""重排(rerank)—— 检索的最后一道精排。

═══ 和前面所有检索的根本区别 ═══

向量检索 / BM25:query 和文档【各自独立】编码,再比距离。
    编码文档时根本不知道 query 是什么 —— 快(能预先算好 26 万个),但粗。

rerank(cross-encoder):query 和文档【拼在一起】喂进模型,一次前向出一个分。
    模型能看到"query 的这个词 对上 文档的那个词" —— 精,但慢:
    每个候选都要一次模型前向,没法预先算(query 是现来的)。

所以两者是【分工】不是替代:
    检索负责【召回】—— 从 26 万里把对的捞进前 N(我们的 R@10=0.900)
    rerank 负责【精排】—— 把这 N 个里最对的顶到第 1(冲 R@1)

    rerank 只能重排检索捞出来的东西 —— 它是天花板下的优化,
    不可能找到检索漏掉的。R@10 是它的上限。

═══ 为什么只对少量候选做 ═══

实测:30 个候选 97ms。如果对全部 26 万做,就是几百秒 —— 荒谬。
所以流程永远是:检索粗筛出 N 个(N=20~30)→ 只 rerank 这 N 个。
这也是它排在最后的原因:前面的便宜方法先把范围缩小。
"""

from __future__ import annotations

_MODEL = "BAAI/bge-reranker-v2-m3"  # 多语言 cross-encoder,中文强
_model = None


def get_reranker():
    """懒加载单例。模型 568M,首次加载慢(要下载),之后复用。"""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CrossEncoder(_MODEL, device=device, max_length=512)
    return _model


class Reranker:
    """给 (query, 候选文本) 打分并重排。只依赖模型,不碰数据库 —— 好测、好换。"""

    def __init__(self, model=None) -> None:
        self._model = model  # 允许注入假模型做测试

    @property
    def model(self):
        return self._model if self._model is not None else get_reranker()

    def rerank(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """candidates: [(chunk_id, 文本), ...] → [(chunk_id, 分数), ...] 按分降序。

        文本从哪来:检索返回的是父块,父块正文就是要给 LLM 的东西,
        也正是该拿去 rerank 的东西 —— 粒度天然一致。
        """
        if not candidates:
            return []
        pairs = [(query, text) for _cid, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            ((cid, float(s)) for (cid, _t), s in zip(candidates, scores)),
            key=lambda kv: -kv[1],
        )
        return ranked
