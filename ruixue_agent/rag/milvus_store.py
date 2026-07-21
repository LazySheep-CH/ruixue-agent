"""Milvus 向量索引 —— 只管"哪些块像",不管"文本是什么"。

职责边界(这个文件最重要的一句话):
    Milvus 里【没有文本】。它只存 chunk_id + 向量 + 过滤字段。
    文本在 PG。所以 search() 还给你的是 (chunk_id, 相似度),不是 Chunk 对象。
    这和 numpy 版 VectorStore 的接口【不一样】,是刻意的 ——
    那一版手里握着全部东西所以能还 Chunk;这一版存储拆开了,接口必须跟着变。

为什么 year/source 要【复制】一份到 Milvus:
    不是"把数据搬到向量库",PG 那份一个字段都没少。这份是【副本】,属于索引的一部分,
    PG 改了就重建 —— 和向量本身一个待遇(向量不也是从 PG 的文本算出来的吗)。
    复制它的理由是【前过滤】:
      后过滤 = 先搜 top10 再筛 2020 年后的 → 可能筛完剩 0 条
      前过滤 = Milvus 内部先圈定 2020 年后的子集,在子集里搜 top10 → 稳拿 10 条
    前过滤要求过滤字段在 Milvus 里。这是标准做法,不是妥协。

    还是那条规矩:数据是资产(PG),索引是派生物(Milvus)。副本属于派生物那一侧。
"""

from __future__ import annotations

from pymilvus import DataType, MilvusClient

from ruixue_agent.rag.embedding import embed

_DIM = 512  # BAAI/bge-small-zh-v1.5 的维度。实测确认过,不是查手册来的
_URI = "http://localhost:19530"
_BATCH = 2000  # 一次 upsert 塞多少条

# ⚠ Milvus 是【最终一致】的 —— upsert 返回 ≠ 数据立刻可见。
#   PG 提交完立刻能查到;Milvus 默认(Bounded)有几秒延迟窗口。
#   实测:灌完 3 条立刻 count(*) 得到 0,过一会儿才变 3。
#
#   Strong = 读之前先跟时间戳服务同步,保证看得见所有已提交的写入。
#   代价是每次查询多一次同步往返。
#   我们的负载是"批量灌完,之后只读",这个代价可以忽略 —— 但要用数据确认,不是拍脑袋。
_CONSISTENCY = "Strong"


class MilvusVectorStore:
    def __init__(self, collection: str = "chunks", uri: str = _URI) -> None:
        self.client = MilvusClient(uri=uri)
        self.collection = collection

    # ── 建表 ──────────────────────────────────────────

    def ensure_collection(self, index_type: str = "FLAT") -> None:
        """建 collection。已存在就跳过 —— 每次启动都会调,不能炸。

        为什么先用 FLAT(暴力,全部算一遍):
            FLAT 的结果是【100% 正确】的 —— 它就是真值(ground truth)。
            HNSW 快,但召回率是 98% 还是 80%?没有 FLAT 做对照你【永远不知道】。
            召回率不是查手册查来的,是拿同一批查询、两边比对算出来的。
            而且 22 万条 FLAT 本来就够快 —— 先量,再决定要不要换。

        index_type 是参数不是硬编码:换索引 = drop_index + 重建,向量【不用重算】。
        贵的是 embed(一次),索引想建几次建几次。
        """
        if self.client.has_collection(self.collection):
            return

        schema = MilvusClient.create_schema(auto_id=False)
        # chunk_id 做主键 → upsert 靠它判断"是不是同一条" → 幂等的锚
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=_DIM)
        # nullable:实测 238/1578 篇没有 year。抽不到是常态,不是异常
        schema.add_field("year", DataType.INT64, nullable=True)
        schema.add_field("source", DataType.VARCHAR, max_length=32, nullable=True)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            # COSINE:我们的向量 embed 时已经归一化(normalize_embeddings=True),
            # 所以点积 == 余弦。写 COSINE 而不是 IP 是为了【自解释】——
            # 哪天有人把归一化去掉了,COSINE 还是对的,IP 就悄悄错了。
            metric_type="COSINE",
        )
        # year/source 建标量索引:前过滤要靠它才快
        index_params.add_index(field_name="year")
        index_params.add_index(field_name="source")

        self.client.create_collection(
            self.collection, schema=schema, index_params=index_params
        )

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    # ── 灌数据 ────────────────────────────────────────

    def index(self, rows: list[dict]) -> int:
        """把子块 embed 后灌进去。rows 里每条要有 chunk_id / text / year / source。

        幂等:靠 upsert + chunk_id 主键。同一条灌两次 = 库里还是一条。

        ⚠ Milvus【没有事务】—— 灌到一半挂了不会回滚,会留下半截数据。
          所以幂等在这里比在 PG 里更要命:它是【唯一】的恢复手段。
          配合 existing_ids() 可以做断点续传,连重新 embed 都省了。
        """
        if not rows:
            return 0

        for i in range(0, len(rows), _BATCH):
            batch = rows[i : i + _BATCH]
            vectors = embed([r["text"] for r in batch])  # GPU 自动生效
            self.client.upsert(
                self.collection,
                data=[
                    {
                        "chunk_id": r["chunk_id"],
                        "vector": v.tolist(),
                        "year": r.get("year"),
                        "source": r.get("source"),
                        # 注意没有 text —— 文本是 PG 的活
                    }
                    for r, v in zip(batch, vectors, strict=True)
                ],
            )
        return len(rows)

    def existing_ids(self, chunk_ids: list[str]) -> set[str]:
        """这批 ID 里,哪些已经在库里了 —— 断点续传的基础。

        22 万条重跑时,已经灌进去的连 embed 都不用再算。
        """
        if not chunk_ids:
            return set()
        found: set[str] = set()
        for i in range(0, len(chunk_ids), _BATCH):
            rows = self.client.query(
                self.collection,
                filter=f"chunk_id in {chunk_ids[i : i + _BATCH]}",
                output_fields=["chunk_id"],
                consistency_level=_CONSISTENCY,  # 漏读 → 白重算一遍,必须准
            )
            found.update(r["chunk_id"] for r in rows)
        return found

    def count(self) -> int:
        """真实行数。

        ⚠ 不能用 get_collection_stats()["row_count"] —— 它会骗人:
          ① 异步更新,刚灌完是 0
          ② 把【软删除】的行也算进去。upsert = 删+插,所以同样 3 条灌两遍它报 6。
          (踩过:测试里 assert count()==3 拿到 6,一度以为 upsert 没去重)
        """
        rows = self.client.query(
            self.collection,
            filter="",
            output_fields=["count(*)"],
            consistency_level=_CONSISTENCY,
        )
        return rows[0]["count(*)"]

    # ── 检索 ──────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 3,
        year_min: int | None = None,
        source: str | None = None,
    ) -> list[tuple[str, float]]:
        """返回 [(chunk_id, 相似度), ...],按相似度从高到低。

        还的是 ID 不是文本 —— 拿这些 ID 去 PgRepository.get_parents() 取父块正文。
        谁来组合这两边?上层的 Retriever(下一步)。

        year_min / source 是【前过滤】:Milvus 内部先圈定子集再搜,
        不是搜完再筛(那样可能筛完剩 0 条)。
        """
        conds = []
        if year_min is not None:
            conds.append(f"year >= {year_min}")
        if source is not None:
            conds.append(f'source == "{source}"')

        hits = self.client.search(
            self.collection,
            data=[embed([query])[0].tolist()],
            filter=" and ".join(conds),  # 空列表 → "" → 不过滤
            limit=k,
            output_fields=["chunk_id"],
            consistency_level=_CONSISTENCY,
        )
        return [(h["entity"]["chunk_id"], float(h["distance"])) for h in hits[0]]
