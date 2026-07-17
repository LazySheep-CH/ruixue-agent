"""向量库(Chunk 版):Small-to-Big 检索。

子块 → embedding 建索引(小、精确,负责【被搜到】)
父块 → 存字典备查,不做 embedding(大、完整,负责【被返回给大模型】)
查询:query → 找最像的子块 → 顺 parent_id 取回父块

用 numpy 暴力算余弦(向量都归一化过,点积=余弦)。百万级再换 Milvus,现在上是过早优化。
"""

from __future__ import annotations

import numpy as np

from ruixue_agent.ingestion.schema import Chunk
from ruixue_agent.rag.embedding import embed


class VectorStore:
    def __init__(self) -> None:
        self.children: list[Chunk] = []  # 被索引的子块(下标与 vectors 对应)
        self.parents: dict[str, Chunk] = {}  # chunk_id → 父块,按 id 备查
        self.vectors: np.ndarray | None = None  # 子块向量矩阵

    def index(self, chunks: list[Chunk]) -> None:
        """把混在一起的父块/子块分开:子块做 embedding 建索引,父块存字典。

        轮到你写(三步):
          1. self.children = 只留 parent_id 不是 None 的(那是子块)
          2. self.parents  = {c.chunk_id: c for ...} 只收 parent_id 是 None 的(那是父块)
          3. self.vectors  = embed([每个子块的 text])
        """
        # 你写
        for chunk in chunks:
            if chunk.parent_id is not None:
                self.children.append(chunk)
            else:
                self.parents[chunk.chunk_id] = chunk

        texts = []
        for chunk in self.children:
            texts.append(chunk.text)

        self.vectors = embed(texts)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, Chunk | None, float]]:
        """查询 → 命中子块 → 顺 parent_id 取父块。返回 [(子块, 父块, 相似度), ...]。

        轮到你写(四步):
          1. qv = embed([query])[0]              把问题也变成向量
          2. sims = self.vectors @ qv            所有子块 · 问题 = 每个子块的相似度
          3. top = np.argsort(-sims)[:k]         相似度从高到低取前 k 个的下标
          4. 对每个下标 i:child = self.children[i]
                          parent = self.parents.get(child.parent_id)
                          收集 (child, parent, float(sims[i]))
        """
        # 你写
        qv = embed([query])[0]
        sims = self.vectors @ qv
        top = np.argsort(-sims)[:k]
        results = []
        for i in top:
            child = self.children[i]  # 下标 i → 第 i 个子块
            parent = self.parents.get(child.parent_id)  # 顺 parent_id 取父块
            results.append((child, parent, float(sims[i])))
        return results
