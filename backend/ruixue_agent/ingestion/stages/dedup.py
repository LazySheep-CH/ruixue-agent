"""dedup:文档级近似去重(切块前)。

sha256(organize_raw 已做)只抓"字节全同";这里抓"内容几乎一样"——同一篇的不同水印/重下版本。
在稳定单元(整篇)上去重;chunk 级因边界不稳,改到检索时用向量语义去重。

用 MinHash + LSH(行业标准:Google网页去重、Common Crawl / LLM训练数据清洗都是这套):

  1) MinHash:把几千个 shingle 的大集合,压成 128 个数的【签名】。
     数学保证:两个签名"对应位置相同的比例" ≈ 两个集合的 Jaccard 相似度。
     → 比 128 个数,而不是比 5000 个 shingle。

  2) LSH:把签名分段哈希进桶。数学保证:相似的文档【至少有一段会落进同一个桶】。
     → 只和"同桶的候选"比,而不是和所有人比。

  复杂度 O(n²) → 约 O(n)。实测(见 tests):1733 篇从【48.5分钟】降到【秒级】。

  代价:近似算法,有极小的漏检/误检概率(num_perm 越大越准、越慢)。
        精确的 jaccard() 保留在下面,供小规模场景和测试用。
"""

from __future__ import annotations

from datasketch import MinHash, MinHashLSH

from ruixue_agent.ingestion.schema import Document

_NUM_PERM = 128  # 签名长度:越大越准越慢。128 是 datasketch 的常用默认


def _doc_text(doc: Document) -> str:
    return "".join(e.text for e in doc.elements)


def _shingles(text: str, k: int = 4) -> set[str]:
    """把文本切成 k 字滑动片段(去空白),作为内容"指纹原料"。

    为什么滑动重叠:插入/删除一个字会让"不重叠的切法"整体错位、全变;
    滑动窗口只影响附近几个片段,其余不动 → 对小改动免疫。
    """
    text = "".join(text.split())
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """精确 Jaccard = 交集/并集。1=完全相同,0=毫无重合。(小规模/测试用)"""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _minhash(shingles: set[str]) -> MinHash:
    """一个 shingle 集合 → 一个 MinHash 签名。"""
    m = MinHash(num_perm=_NUM_PERM)
    m.update_batch([s.encode("utf-8") for s in shingles])
    return m


def dedup_documents(
    docs: list[Document], threshold: float = 0.9
) -> tuple[list[Document], list[Document]]:
    """近似去重:保留首次出现,丢弃与已保留者相似度 ≥ threshold 的近似重复。

    返回 (保留, 丢弃)。丢弃的应记入 manifest —— 不悄悄丢数据。
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=_NUM_PERM)
    kept: list[Document] = []
    dropped: list[Document] = []

    for doc in docs:
        sig = _minhash(_shingles(_doc_text(doc)))
        # LSH 直接给出"可能相似"的候选;有候选就说明撞上了已保留的某篇
        if lsh.query(sig):
            dropped.append(doc)
        else:
            lsh.insert(doc.document_id, sig)  # 只有保留的才进索引
            kept.append(doc)
    return kept, dropped
