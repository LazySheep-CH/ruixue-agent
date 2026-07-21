"""RRF(Reciprocal Rank Fusion)—— 把多个检索器的排名合成一个。

═══ 为什么不能直接加分数 ═══

向量给的是余弦相似度(0~1,通常挤在 0.4~0.9),BM25 给的是 ts_rank_cd
(0~1,但分布完全不同,大多数在 0.0x)。两个【量纲不同、分布不同】的分数
直接相加或加权平均,等于让分数尺度大的那一方说了算 —— 而尺度大小
跟"谁更准"毫无关系。

归一化(min-max / z-score)能缓解,但引入新问题:每次查询的分数范围都不同,
归一化后的值不可比;而且一个离群的高分会把其他所有分数压扁。

═══ RRF 的做法:只看排名,不看分数 ═══

    score(d) = Σ  1 / (k + rank_i(d))
               i

每个检索器里排第 1 的贡献 1/(k+1),第 2 名 1/(k+2)…… 求和。

好处:
  ① 【量纲无关】—— 排名是序数,不存在尺度问题
  ② 【鲁棒】—— 某个检索器打分离谱不影响别人
  ③ 【无需调参】—— k 取 60 是原论文(Cormack et al. 2009)的经验值,
     在很多数据集上都稳,几乎不用调

代价:丢掉了分数里的【置信度】信息 —— 第 1 名比第 2 名领先很多,
      和只领先一点点,RRF 里是一样的。

k 的作用:压平头部差距。k 越大,第 1 名和第 10 名的差距越小(更看重"两边都投了票"),
         k 越小越看重"谁排第一"。60 是个偏保守的值。
"""

from __future__ import annotations

_K = 60  # Cormack et al. 2009 的经验值,跨数据集都稳


def rrf(
    rankings: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    k: int = _K,
) -> list[tuple[str, float]]:
    """把多个 [(id, 分数), ...] 排名融合成一个,按融合分降序。

    rankings: 每个检索器的结果(已按各自的分数排好序)
    weights:  每路的权重,不给就等权。给了的话长度要和 rankings 一致。

    注意入参的分数【只用来确定顺序】,不参与计算 —— 这正是 RRF 的关键:
    它只信排名,不信分数。
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
