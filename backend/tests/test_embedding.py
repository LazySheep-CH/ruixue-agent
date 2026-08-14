"""embedding 的测试。

真加载模型跑一次(慢,但这是契约的一部分:维度、归一化、语义方向都不能变)。
下游全指望这三条:Milvus 建表写死了 512 维、检索靠"归一化后点积=余弦"、
Small-to-Big 靠"相关的分高"。任何一条破了,整个检索静默失效。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ruixue_agent.rag.embedding import embed

# 这几条要真加载模型(会下载 ~100MB),本地默认跑,但 CI 快速通道跳过:
# CI 里设了环境变量 SKIP_MODEL_TESTS=1 就整文件跳过,避免每次 CI 下大模型。
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_MODEL_TESTS") == "1",
    reason="CI 快速通道:跳过需要下载模型的测试",
)


def test_dimension_is_512():
    """维度必须是 512 —— Milvus collection 建表时写死了这个数。

    换模型换维度的话,这个测试先红,提醒你 Milvus 也要跟着重建,
    而不是等灌库时才报维度不匹配。
    """
    v = embed(["测试"])
    assert v.shape == (1, 512)


def test_normalized():
    """向量已归一化(模长=1)—— 这样点积就等于余弦相似度。

    milvus_store 用 COSINE、numpy 侧直接点积,都依赖这个前提。
    """
    v = embed(["地膜降解性能"])
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-5


def test_semantic_direction():
    """相关的分高、无关的分低 —— 检索能工作的最基本前提。"""
    v = embed(["PBAT地膜的降解性能", "生物降解薄膜的透水性", "今天天气很好"])
    related = float(v[0] @ v[1])  # 都讲膜
    unrelated = float(v[0] @ v[2])  # 天气
    assert related > unrelated
    assert related > 0.3  # 相关的要有实际相似度,不只是"比无关高一点"


def test_batch():
    v = embed(["一", "二", "三"])
    assert v.shape[0] == 3
