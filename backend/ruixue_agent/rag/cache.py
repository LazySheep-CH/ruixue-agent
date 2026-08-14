"""问答缓存:同一个问题不重复走"检索 + 重排 + 生成"这条贵路径。

为什么是【精确匹配】而不是语义缓存:
原计划做语义缓存(问题向量相近就复用答案),实测发现在本领域不安全:

    「地膜降解太快怎么办」 ↔ 「地膜降解太慢怎么办」  余弦相似度 0.9727

比所有真正的同义问题(最高 0.946)还高 —— embedding 分不清反义词。两句字面
几乎一样,答案却完全相反(一个要加厚/降 PBAT,一个要减薄/升 PBAT)。若上语义缓存,
农户问"太慢"会拿到"太快"的建议,直接给出相反的农事决策。

实测在任何阈值下都无法兼顾:阈值 0.95 时同义命中 0/5,错误命中仍有 1/8。
故放弃语义匹配,改用归一化后精确匹配:命中率低一些,但零误答风险。
实际使用中"同一个标准问题被反复问"很常见(不同用户问同样的国标/规程),收益是实的。

失效(invalidation):
知识库更新后旧答案会过时。故缓存键里带上知识库版本(chunk 总数的指纹):
灌新数据 → 指纹变 → 旧缓存自然不再命中,无需手工清理。
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata

# 内存缓存:{key: (答案, 写入时间)}。
# 为什么先用内存而不是 Redis/Milvus:单进程即可验证收益,且答案可再生(丢了只是多花一次钱)。
# 多 worker 部署时改挂 Redis —— 换的是这里的存储实现,调用方不动。
_CACHE: dict[str, tuple[str, float]] = {}

MAX_ENTRIES = 500  # 上限,防内存无限增长
TTL_SECONDS = 7 * 24 * 3600  # 一周过期:知识库版本没变时,答案也不宜无限期陈放


def normalize_question(q: str) -> str:
    """问题归一化:让"表达上无差别"的写法命中同一个键。

    只做【无损】的规整 —— 大小写、全角半角、空白、句末标点。
    绝不做同义词替换那类"猜意思"的处理:那正是语义缓存不安全的根源。
    """
    s = unicodedata.normalize("NFKC", q).strip().lower()
    s = re.sub(r"\s+", "", s)  # 中文里空格无语义,去掉
    # 去掉句末标点(用正则而非 rstrip:rstrip 是逐字符剥离,语义容易误解)
    return re.sub(r"[?？。.!！,,、;;：:]+$", "", s)


def make_key(question: str, kb_version: str) -> str:
    """缓存键 = 归一化问题 + 知识库版本(版本变则旧缓存自然失效)。"""
    raw = f"{kb_version}|{normalize_question(question)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def get(question: str, kb_version: str) -> str | None:
    """命中返回答案,否则 None。过期项顺手清掉。"""
    key = make_key(question, kb_version)
    item = _CACHE.get(key)
    if item is None:
        return None
    answer, ts = item
    if time.time() - ts > TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return answer


def put(question: str, kb_version: str, answer: str) -> None:
    """写入缓存;超出上限时淘汰最旧的一条(简单 FIFO,够用)。"""
    if len(_CACHE) >= MAX_ENTRIES:
        oldest = min(_CACHE, key=lambda k: _CACHE[k][1])
        _CACHE.pop(oldest, None)
    _CACHE[make_key(question, kb_version)] = (answer, time.time())


def clear() -> None:
    """清空(测试与运维用)。"""
    _CACHE.clear()


def stats() -> dict:
    return {"entries": len(_CACHE), "max_entries": MAX_ENTRIES}
