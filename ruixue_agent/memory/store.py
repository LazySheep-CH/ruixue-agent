"""记忆的存与取。PG 是权威,Milvus 是可重建的语义索引。

## 为什么是两套存储(面经必追的一问)

"既然存的是用户偏好标签,为什么不用 MySQL?"——因为记忆有两种查法:

    精确查:"这个用户的地块在哪"          → 一条 SQL 就够
    语义查:"当前问题和他以前说过的什么相关" → 只能靠向量

第二种关系库做不到:用户说过"我这边风大",现在问"选哪种配方",
关键词一个都不重合,但语义上高度相关(风大 → 该看拉伸强度)。

分工和文档那套完全一致:**PG 存事实(权威、可查、可删、可审计),
Milvus 只存 id + 向量(索引,丢了重建即可)。**
"""

from __future__ import annotations

import hashlib
import logging

from pymilvus import DataType
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.models import MemoryRow
from ruixue_agent.rag.embedding import embed
from ruixue_agent.rag.milvus_store import _DIM, _URI

logger = logging.getLogger("ruixue.memory")

# 单独一个 collection,不和文档块混在一起。
# 为什么必须分开:两者的"相关"含义不同 —— 文档块问的是"哪段资料能回答这个问题",
# 记忆问的是"这个用户以前说过什么和这有关"。混在一个库里检索会互相污染,
# 而且记忆是【按用户隔离】的,文档是全局共享的,过滤逻辑也不一样。
MEMORY_COLLECTION = "memories"

# 一次召回几条。取 5 是权衡:太少漏掉相关背景,太多会稀释注意力、挤占上下文。
RECALL_TOP_K = 5
# 相似度低于这个值就不注入 —— 宁可不给,也不要给一段不相关的"记忆"
# 让模型据此瞎推理(那是主动制造幻觉)。
#
# ⚠ 【已知问题,未调优】0.35 是拍的,实测有两类偏差:
#     漏召:问"选哪种配方比较好"时只召回了偏好,漏掉了地块和作物 ——
#           而这两条恰恰是选配方最需要的(地点定环境、作物定生育期)。
#     误召:问"今天天气怎么样"会召回"用户所在地风大"(语义确实沾边,但没用)。
#   根因是"当前问题"和"历史事实"本来就不是同一种文本,直接算余弦相似度
#   并不贴合"这条背景对回答这个问题有没有帮助"。
#   正确的调法是【先有评测再调阈值】—— 造一批"问题→应召回哪几条"的标注,
#   像调检索那样量 Recall@k,而不是凭感觉挪这个数字。
#   在有评测之前,这个数字保持保守(宁可漏,不可乱注入)。
RECALL_MIN_SCORE = 0.35


def _memory_id(user_id: str, text: str) -> str:
    """内容寻址:同一个用户重复说同一件事,只存一条。

    和文档的 document_id 同一个思路 —— 幂等靠内容哈希,不靠中心协调。
    """
    return hashlib.sha256(f"{user_id}|{text}".encode()).hexdigest()[:16]


def _client():
    from pymilvus import MilvusClient

    return MilvusClient(uri=_URI)


def ensure_collection() -> None:
    """建记忆库并确保已加载(启动时幂等调用,搜索前也会兜底调一次)。

    ⚠ Milvus 的 collection 建好、索引建好【还不能搜】,必须先 load 进内存。
      漏了这一步报的是 `collection not loaded`,而不是"没数据" —— 我第一版就漏了,
      表现为"存进去了但一条都召不回",而且因为我们对召回失败做了优雅降级
      (静默不注入),这个 bug 在生产里会表现成"记忆功能好像没生效"。
      同一个坑 rag/milvus_store.py 里早有注释,我写新模块时没复用经验。

    做成【自愈】而不只在建库时 load:Milvus 重启后 collection 会回到未加载状态,
    只在启动时 load 一次的话,重启 Milvus 就再也搜不了了。
    """
    c = _client()
    if not c.has_collection(MEMORY_COLLECTION):
        schema = c.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("memory_id", DataType.VARCHAR, is_primary=True, max_length=16)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=_DIM)
        # 用户 id 作为标量字段:检索时【必须】按它过滤 —— 否则会召回别人的记忆,
        # 这不只是效果问题,是数据泄露。
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        c.create_collection(MEMORY_COLLECTION, schema=schema)
        idx = c.prepare_index_params()
        idx.add_index("vector", index_type="FLAT", metric_type="COSINE")
        c.create_index(MEMORY_COLLECTION, idx)
        logger.info("已创建记忆 collection")
    c.load_collection(MEMORY_COLLECTION)  # 索引建好要 load 才能搜


def remember(user_id: str, facts: list[tuple[str, str]], run_id: str | None = None) -> int:
    """存入记忆。facts 是 (kind, text) 列表。返回【新增】条数。

    幂等:同一用户说过同样的话不会重复存(内容寻址)。
    """
    if not facts:
        return 0
    rows = [
        MemoryRow(
            memory_id=_memory_id(user_id, text),
            user_id=user_id,
            kind=kind,
            text=text,
            source_run_id=run_id,
        )
        for kind, text in facts
    ]
    new_ids: list[str] = []
    with Session(get_engine()) as s:
        existing = set(
            s.scalars(
                select(MemoryRow.memory_id).where(
                    MemoryRow.memory_id.in_([r.memory_id for r in rows])
                )
            ).all()
        )
        fresh = [r for r in rows if r.memory_id not in existing]
        if not fresh:
            return 0
        s.add_all(fresh)
        s.commit()
        new_ids = [r.memory_id for r in fresh]
        texts = [r.text for r in fresh]

    # 向量后写:PG 成功了才写索引。反过来的话 PG 失败会留下"孤儿向量" ——
    # 检索命中一个 PG 里不存在的 id,只能跳过,白占空间还查不出原因。
    try:
        ensure_collection()
        vecs = embed(texts)
        _client().upsert(
            MEMORY_COLLECTION,
            [
                {"memory_id": mid, "vector": v, "user_id": user_id}
                for mid, v in zip(new_ids, vecs, strict=True)
            ],
        )
    except Exception:
        # 索引写失败不回滚 PG:事实本身已经记住了,向量可以重建。
        # 这正是"PG 权威、向量可重建"的好处 —— 失败的代价是"这条暂时召不回",
        # 而不是"这条丢了"。
        logger.warning("记忆向量写入失败,PG 已存;可重建索引", exc_info=True)
    return len(new_ids)


def recall(user_id: str, query: str, k: int = RECALL_TOP_K) -> list[MemoryRow]:
    """按当前问题召回这个用户的相关记忆。"""
    try:
        # 兜底:Milvus 重启后 collection 会回到未加载状态。这里调一次幂等的
        # ensure_collection,代价是一次轻量 RPC,换来"重启 Milvus 后记忆还能用"。
        ensure_collection()
        vec = embed([query])[0]
        hits = _client().search(
            MEMORY_COLLECTION,
            data=[vec],
            limit=k,
            # ⚠ 这个过滤条件是【安全边界】,不是优化:漏了它就会召回别人的记忆。
            filter=f'user_id == "{user_id}"',
            output_fields=["memory_id"],
        )
    except Exception:
        # 记忆是锦上添花:召回挂了就当没有记忆,不能拖垮回答
        logger.warning("记忆召回失败,本次不注入", exc_info=True)
        return []

    ids = [
        h["entity"]["memory_id"]
        for h in (hits[0] if hits else [])
        if h["distance"] >= RECALL_MIN_SCORE
    ]
    if not ids:
        return []
    with Session(get_engine()) as s:
        rows = s.scalars(
            select(MemoryRow).where(
                MemoryRow.memory_id.in_(ids),
                MemoryRow.user_id == user_id,  # 双保险:PG 侧再校验一次归属
                MemoryRow.deleted.is_(False),
            )
        ).all()
        return list(rows)


def list_memories(user_id: str) -> list[MemoryRow]:
    """列出某用户的全部记忆。给"让用户看见并管理自己的记忆"用 ——
    记忆不该是黑箱:用户有权知道系统记住了什么,也有权删掉。"""
    with Session(get_engine()) as s:
        return list(
            s.scalars(
                select(MemoryRow)
                .where(MemoryRow.user_id == user_id, MemoryRow.deleted.is_(False))
                .order_by(MemoryRow.created_at.desc())
            ).all()
        )


def delete_memory(user_id: str, memory_id: str) -> bool:
    """软删一条记忆。返回是否删到。

    为什么软删:① 保留审计痕迹;② 硬删的话下次同样的话又会被抽出来存回去,
    用户会觉得"我删了它还记得"。软删 + 内容寻址 = 删过的不会复活。
    """
    with Session(get_engine()) as s:
        n = s.execute(
            update(MemoryRow)
            .where(MemoryRow.memory_id == memory_id, MemoryRow.user_id == user_id)
            .values(deleted=True)
        ).rowcount
        s.commit()
    return bool(n)
