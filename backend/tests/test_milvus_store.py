"""MilvusVectorStore 的规格说明书。

注意:和 numpy 版 VectorStore 的接口不一样,这是刻意的:
    numpy 版手里握着全部东西(children/parents/vectors),所以 search() 能直接还你 Chunk。
    Milvus 版只有向量,文本在 PG 里 —— 它根本没有 Chunk 可还。
    存储拆开了,接口就得跟着变。

    MilvusVectorStore.search()  →  [(chunk_id, 相似度), ...]     只管"哪些块像"
    PgRepository.get_parents()  →  父块文本                       只管"文本是什么"
    两者由上层的 Retriever 组合(下一步的事)。

    这不是妥协,是职责分离:向量的事归 Milvus,文本的事归 PG。

跑这些测试要 Milvus 起着(docker compose up -d)。没起就 skip,不是报错。
"""

from __future__ import annotations

import pytest

from ruixue_agent.rag.milvus_store import MilvusVectorStore

# 测试用的独立 collection —— 绝不碰生产那个。
# (和 conftest 里"测完回滚"同一个思想:测试不能污染真实数据)
_TEST_COLLECTION = "test_chunks_tmp"


@pytest.fixture
def store():
    try:
        s = MilvusVectorStore(collection=_TEST_COLLECTION)
        s.client.list_collections()
    except Exception:
        pytest.skip("Milvus 没起(cd docker && docker compose up -d)")
    s.drop()  # 从干净状态开始 —— 上次跑挂了留下的残留不能影响这次
    s.ensure_collection()
    yield s
    s.drop()


def _rows():
    """三条子块,文本取自真实语料 —— 构造数据难以覆盖真实分布。"""
    return [
        {
            "chunk_id": "d1_s0_c0",
            "text": "PBAT 牌号 TH801T,由蓝山屯河提供,熔融指数 3.5 g/10min。",
            "year": 2025,
            "source": "期刊论文",
        },
        {
            "chunk_id": "d1_s0_c1",
            "text": "PLA 牌号 4032D,由 NatureWorks 提供,密度 1.24 g/cm³。",
            "year": 2018,
            "source": "期刊论文",
        },
        {
            "chunk_id": "d2_s0_c0",
            "text": "地膜厚度应不小于 0.010 mm,断裂标称应变纵横向均不小于 300%。",
            "year": 2023,
            "source": "标准规范",
        },
    ]


# ────────────────── 1. 建表 ──────────────────


def test_ensure_collection_is_idempotent(store):
    """建表要能重复调 —— 每次启动都调一次,已存在就跳过,不能炸。"""
    store.ensure_collection()
    store.ensure_collection()  # 再来一次
    assert _TEST_COLLECTION in store.client.list_collections()


def test_ensure_collection_self_heals_missing_index(store):
    """回归表存在但缺索引时,ensure_collection 要能自愈补建。

    踩过的坑:旧实现只判断 has_collection 就 return —— 若上次运行在建表与建索引
    之间中断,会留下"有表无索引"的残留,后续 search 报 `index not found`,
    而且这种残留会一直卡住,直到手工 drop。故这里人为制造坏状态验证自愈。
    """
    from pymilvus import DataType, MilvusClient

    from ruixue_agent.rag.milvus_store import _DIM

    store.drop()
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=_DIM)
    schema.add_field("year", DataType.INT64, nullable=True)
    schema.add_field("source", DataType.VARCHAR, max_length=32, nullable=True)
    store.client.create_collection(_TEST_COLLECTION, schema=schema)  # 故意不建索引
    assert store.client.list_indexes(_TEST_COLLECTION) == []  # 确认坏状态造出来了

    store.ensure_collection()  # 自愈

    assert "vector" in set(store.client.list_indexes(_TEST_COLLECTION))
    store.index(_rows())
    assert store.search("地膜", k=1)  # 能真正搜出来,不再 index not found


def test_schema_has_filter_fields(store):
    """year/source 必须在 Milvus 里 —— 否则只能后过滤:
    先搜 top10 再筛 2020 年后的,可能筛完剩 0 条。
    前过滤要求过滤字段在库里。
    """
    fields = {f["name"] for f in store.client.describe_collection(_TEST_COLLECTION)["fields"]}
    assert {"chunk_id", "vector", "year", "source"} <= fields


# ────────────────── 2. 灌数据 ──────────────────


def test_index_and_count(store):
    n = store.index(_rows())
    assert n == 3
    assert store.count() == 3


def test_index_is_idempotent(store):
    """22 万条灌到一半挂了必须能直接重跑。

    Milvus没有事务—— 挂了不会回滚,会留下半截数据。
    所以幂等在这里比在 PG 里更要命:它是唯一的恢复手段。
    """
    store.index(_rows())
    store.index(_rows())  # 全部撞主键
    assert store.count() == 3  # 不是 6


def test_index_empty_is_noop(store):
    assert store.index([]) == 0
    assert store.count() == 0


def test_existing_ids_enables_resume(store):
    """断点续传的基础:先问"哪些已经在库里了",跳过它们。

    22 万条重新 embed 一遍是几分钟的浪费 —— 能跳就跳。
    """
    store.index(_rows()[:2])
    have = store.existing_ids([r["chunk_id"] for r in _rows()])
    assert have == {"d1_s0_c0", "d1_s0_c1"}  # 第三条还没进去


# ────────────────── 3. 检索 ──────────────────


def test_search_finds_semantically_closest(store):
    """问 PBAT 的牌号,PBAT 那条要排第一。"""
    store.index(_rows())
    hits = store.search("PBAT 用的什么牌号?", k=3)
    assert hits[0][0] == "d1_s0_c0"
    assert hits[0][1] > hits[1][1]  # 相似度递减
    assert len(hits) == 3


def test_search_returns_ids_not_chunks(store):
    """契约:只还 (chunk_id, 相似度)。

    Milvus 里没有文本—— 文本是 PG 的活。
    这个测试锁住职责边界:哪天有人想往 Milvus 塞 text 字段,这里就该红。
    """
    store.index(_rows())
    hits = store.search("地膜厚度", k=1)
    assert isinstance(hits[0][0], str) and isinstance(hits[0][1], float)


def test_search_prefilters_by_year(store):
    """前过滤:在 2020 年后的子集里搜,不是搜完再筛。

    区别看这里:PBAT 那条(2025)语义上最像"牌号",PLA 那条(2018)次之。
    过滤 year>=2020 后,PLA 那条根本不该出现。
    """
    store.index(_rows())
    hits = store.search("牌号", k=5, year_min=2020)
    ids = [h[0] for h in hits]
    assert "d1_s0_c1" not in ids  # PLA 是 2018 年的,被前过滤挡掉


def test_search_filters_by_source(store):
    store.index(_rows())
    hits = store.search("地膜", k=5, source="标准规范")
    assert [h[0] for h in hits] == ["d2_s0_c0"]


def test_search_on_empty_collection(store):
    """空库查询:返回空列表,不是崩。"""
    assert store.search("随便问点啥", k=3) == []
