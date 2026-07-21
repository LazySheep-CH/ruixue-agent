"""重排:cross-encoder 对检索候选做最终精排。

向量检索与 BM25 都是双塔式打分 —— query 与文档各自独立编码再比距离,
文档侧可离线预计算,快但粗。cross-encoder 把 query 和文档拼接后整体
过一次模型,能捕捉词级对应关系,精度显著更高,但每个候选都需一次前向,
无法预计算,只能作用于检索粗筛后的少量候选(30 个约 97ms)。

分工:检索负责召回(把正确答案捞进候选池),重排负责精排(把最相关的
顶到前列)。重排只能调整已召回的候选,上限受检索召回率约束。
实测(150 题评测集):在混合检索之上加重排,R@1 0.647 → 0.807。
"""

from __future__ import annotations

_MODEL = "BAAI/bge-reranker-v2-m3"  # 多语言 cross-encoder,中文效果好
_model = None


def get_reranker():
    """懒加载单例。模型 568M 参数,首次加载慢,进程内复用。"""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CrossEncoder(_MODEL, device=device, max_length=512)
    return _model


class Reranker:
    """对 (query, 候选文本) 打分并重排。不依赖数据库,模型可注入以便测试。"""

    def __init__(self, model=None) -> None:
        self._model = model

    @property
    def model(self):
        return self._model if self._model is not None else get_reranker()

    def rerank(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """candidates: [(chunk_id, 文本), ...] → [(chunk_id, 分数), ...],按分降序。

        候选文本用父块正文 —— 与最终交给生成层的粒度一致。
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
