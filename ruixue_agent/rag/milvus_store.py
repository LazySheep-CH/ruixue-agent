"""Milvus 向量索引。

只存 chunk_id + 向量 + 过滤字段(year/source),不存文本 —— 文本的
source of truth 在 PostgreSQL,Milvus 是可随时重建的派生索引。若把文本
也放进来,索引就兼任了数据源,重建索引即丢数据。

year/source 是从 PG 复制来的副本,用于前过滤:在向量检索内部先圈定子集
再取 top-k。若采用后过滤(先取 top-k 再筛),候选可能被筛空。副本属于
索引的一部分,PG 侧变更后随索引一起重建即可。

search() 只返回 (chunk_id, 相似度),取正文由 PgRepository 负责,
两者在 Retriever 层组合。
"""

from __future__ import annotations

from pymilvus import DataType, MilvusClient

from ruixue_agent.rag.embedding import embed

_DIM = 512  # BAAI/bge-small-zh-v1.5 输出维度
_URI = "http://localhost:19530"
_BATCH = 2000  # 单次 upsert 批大小

# Milvus 是最终一致:upsert 返回不代表数据可见(默认 Bounded 有秒级延迟窗口,
# 实测灌入后立即 count 得 0)。读路径统一用 Strong,保证读到全部已提交写入;
# 代价是每次查询多一次时间戳同步。当前负载为批量写入 + 在线只读,可接受。
_CONSISTENCY = "Strong"


class MilvusVectorStore:
    def __init__(self, collection: str = "chunks", uri: str = _URI) -> None:
        self.client = MilvusClient(uri=uri)
        self.collection = collection

    # ── 建表 ──────────────────────────────────────────

    def ensure_collection(self, index_type: str = "FLAT") -> None:
        """创建 collection,已存在则跳过(启动时幂等调用)。

        默认 FLAT(暴力检索):结果是精确真值,可作为评测 HNSW 等近似索引
        召回率的基准;22 万级数据量下延迟已满足需求。换索引类型只需
        drop 后重建,向量无需重算。
        """
        if self.client.has_collection(self.collection):
            return

        schema = MilvusClient.create_schema(auto_id=False)
        # chunk_id 主键是 upsert 幂等的判重依据
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=_DIM)
        # nullable:约 15% 文档抽取不到 year,属正常情况
        schema.add_field("year", DataType.INT64, nullable=True)
        schema.add_field("source", DataType.VARCHAR, max_length=32, nullable=True)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            # 向量在 embed 时已归一化,点积等价余弦;显式用 COSINE 而非 IP,
            # 归一化逻辑将来若变动,COSINE 仍正确
            metric_type="COSINE",
        )
        # 过滤字段建标量索引,前过滤才不退化为全扫
        index_params.add_index(field_name="year")
        index_params.add_index(field_name="source")

        self.client.create_collection(self.collection, schema=schema, index_params=index_params)

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    # ── 写入 ────────────────────────────────────────

    def index(self, rows: list[dict]) -> int:
        """embed 后批量写入。rows 各项需含 chunk_id / text / year / source。

        通过 upsert + 主键实现幂等。Milvus 无事务,中途失败会留下部分数据,
        幂等重跑是唯一恢复手段;配合 existing_ids() 可断点续传,跳过已入库
        的块(连 embed 也省掉)。
        """
        if not rows:
            return 0

        for i in range(0, len(rows), _BATCH):
            batch = rows[i : i + _BATCH]
            vectors = embed([r["text"] for r in batch])
            self.client.upsert(
                self.collection,
                data=[
                    {
                        "chunk_id": r["chunk_id"],
                        "vector": v.tolist(),
                        "year": r.get("year"),
                        "source": r.get("source"),
                    }
                    for r, v in zip(batch, vectors, strict=True)
                ],
            )
        return len(rows)

    def existing_ids(self, chunk_ids: list[str]) -> set[str]:
        """返回给定 ID 中已入库的子集,供断点续传跳过。"""
        if not chunk_ids:
            return set()
        found: set[str] = set()
        for i in range(0, len(chunk_ids), _BATCH):
            rows = self.client.query(
                self.collection,
                filter=f"chunk_id in {chunk_ids[i : i + _BATCH]}",
                output_fields=["chunk_id"],
                consistency_level=_CONSISTENCY,
            )
            found.update(r["chunk_id"] for r in rows)
        return found

    def count(self) -> int:
        """实际行数。

        不能用 get_collection_stats 的 row_count:该值异步更新(刚写入时为 0),
        且包含软删除的行 —— upsert 实现为删+插,同批数据写两遍会翻倍。
        count(*) 查询才是准确值。
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
        """向量检索,返回 [(chunk_id, 相似度)],按相似度降序。

        year_min / source 为前过滤条件,在检索内部生效。
        """
        conds = []
        if year_min is not None:
            conds.append(f"year >= {year_min}")
        if source is not None:
            conds.append(f'source == "{source}"')

        hits = self.client.search(
            self.collection,
            data=[embed([query])[0].tolist()],
            filter=" and ".join(conds),
            limit=k,
            output_fields=["chunk_id"],
            consistency_level=_CONSISTENCY,
        )
        return [(h["entity"]["chunk_id"], float(h["distance"])) for h in hits[0]]
