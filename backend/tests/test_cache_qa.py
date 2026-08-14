"""问答缓存测试。

核心不是"能不能命中",而是绝不能错误命中 —— 本项目刻意放弃语义缓存,
就是因为实测「地膜降解太快怎么办」↔「地膜降解太慢怎么办」相似度高达 0.9727,
语义匹配会给出相反的农事建议。这里的测试把这条底线焊死。
"""

import time

import pytest

from ruixue_agent.rag import cache


@pytest.fixture(autouse=True)
def _clean():
    cache.clear()
    yield
    cache.clear()


KB = "v1"


# ── 基本读写 ──────────────────────────────────────────────────
def test_miss_then_hit():
    assert cache.get("地膜降解率标准", KB) is None
    cache.put("地膜降解率标准", KB, "答案A")
    assert cache.get("地膜降解率标准", KB) == "答案A"


# ── 归一化:表达上无差别的写法应命中同一条 ────────────────────
@pytest.mark.parametrize(
    "variant",
    [
        "地膜降解率标准?",  # 句末问号
        "地膜降解率标准？",  # 全角问号
        "  地膜降解率标准  ",  # 首尾空白
        "地膜 降解率 标准",  # 中间空格(中文里无语义)
    ],
)
def test_normalization_variants_hit_same_entry(variant):
    cache.put("地膜降解率标准", KB, "答案A")
    assert cache.get(variant, KB) == "答案A"


# ── 安全底线:意思不同的问题【绝不能】互相命中 ────────────────
@pytest.mark.parametrize(
    "q1,q2",
    [
        ("地膜降解太快怎么办", "地膜降解太慢怎么办"),  # 反义!语义相似度 0.9727
        ("黑色地膜的作用", "白色地膜的作用"),
        ("地膜厚度标准是多少", "地膜幅宽标准是多少"),
        ("PBAT的降解性能如何", "PLA的降解性能如何"),
        ("覆膜前要做什么准备", "揭膜后要做什么处理"),
    ],
)
def test_different_questions_never_collide(q1, q2):
    """这是放弃语义缓存的原因所在 —— 精确匹配必须保证零误命中。"""
    cache.put(q1, KB, f"{q1} 的答案")
    assert cache.get(q2, KB) is None, f"错误命中!「{q2}」拿到了「{q1}」的答案"


# ── 失效:知识库版本变了,旧答案不该再用 ──────────────────────
def test_kb_version_change_invalidates():
    cache.put("地膜降解率标准", "v1", "旧知识库的答案")
    assert cache.get("地膜降解率标准", "v2") is None


# ── 过期 ──────────────────────────────────────────────────────
def test_expired_entry_is_dropped(monkeypatch):
    cache.put("问题", KB, "答案")
    monkeypatch.setattr(cache, "TTL_SECONDS", -1)  # 立刻过期
    assert cache.get("问题", KB) is None


# ── 容量上限:不能无限涨 ──────────────────────────────────────
def test_evicts_when_full(monkeypatch):
    monkeypatch.setattr(cache, "MAX_ENTRIES", 3)
    for i in range(5):
        cache.put(f"问题{i}", KB, f"答案{i}")
        time.sleep(0.001)  # 保证写入时间有先后,FIFO 才可判定
    assert cache.stats()["entries"] <= 3
    assert cache.get("问题4", KB) == "答案4"  # 最新的还在
    assert cache.get("问题0", KB) is None  # 最旧的被淘汰
