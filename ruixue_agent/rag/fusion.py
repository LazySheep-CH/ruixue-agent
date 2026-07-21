"""RRF(Reciprocal Rank Fusion):把多路检索的排名融合为一个排名。

不能直接把各路分数相加:向量给余弦相似度(集中在 0.4~0.9),词法给
ts_rank_cd(多在 0.0x),量纲与分布都不同,加权和会被量纲大的一方主导,
而量纲大小与检索质量无关。min-max 归一化也不可靠 —— 每次查询的分数范围
不同,归一化后跨查询不可比,且离群高分会压扁其余分数。

RRF 只使用排名,不使用分数:

    score(d) = Σ_i  w_i / (k + rank_i(d))

- 排名是序数,天然量纲无关;
- 单路打分异常不影响其他路;
- k 取 60(Cormack et al. 2009),跨数据集表现稳定,基本无需调参。

代价:丢弃分数中的置信度信息 —— 第 1 名领先第 2 名多少,RRF 不感知。
k 控制头部差距的压缩程度:k 越大越看重"多路共同投票",越小越看重头名。
"""

from __future__ import annotations

_K = 60


def rrf(
    rankings: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    k: int = _K,
) -> list[tuple[str, float]]:
    """融合多路 [(id, 分数), ...] 排名,返回按融合分降序的列表。

    rankings 各路须已按自身分数排好序;weights 缺省为等权。
    入参分数仅用于确定顺序,不参与融合计算。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"weights 有 {len(weights)} 个,但 rankings 有 {len(rankings)} 路"
        )

    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, (cid, _score) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank)

    return sorted(fused.items(), key=lambda kv: -kv[1])
