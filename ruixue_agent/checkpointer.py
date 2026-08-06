"""持久化 checkpointer:agent 会话状态存 PostgreSQL(重启不丢、多 worker 共享)。

对标 InMemorySaver(进程内存、重启即失),这里换成 PostgresSaver。
换的是"状态存哪",不是"接口"—— 两者都实现 BaseCheckpointSaver,agent 侧几乎无感。
和 web 从"内存 session"演进到"Redis/DB session store"是同一个道理。
"""

import os
from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ruixue_agent.persistence.engine import database_url


@lru_cache(maxsize=1)
def get_checkpointer() -> PostgresSaver:
    """全进程共用一个 PG checkpointer(内含连接池)。首次会建好 checkpoint 表。"""
    # PostgresSaver 用 psycopg 直连(不走 SQLAlchemy),要【纯 psycopg 连接串】:
    # 去掉 SQLAlchemy 的方言后缀 "+psycopg"。
    conn_string = database_url().replace("postgresql+psycopg://", "postgresql://")

    # PostgresSaver 对连接有三个硬性要求(否则报错/行为异常):
    #   autocommit=True       每次 checkpoint 写入即时提交
    #   row_factory=dict_row  按列名取行(它内部按字段名读)
    #   prepare_threshold=0   关掉预处理语句缓存,和连接池配合更稳
    pool = ConnectionPool(
        conninfo=conn_string,
        # 池大小要 >= 后台运行并发(runs.MAX_CONCURRENT_RUNS,默认 8):
        # 每个正在跑的 agent 都要用 checkpointer 存取会话状态,池小于并发数
        # 就会互相等连接,表现为"并发一高就变慢"。留 2 条余量给恢复/查询。
        max_size=int(os.getenv("CHECKPOINTER_POOL_SIZE", "10")),
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        open=False,  # 不在构造时开(新版 psycopg_pool 推荐显式 open)
    )
    pool.open()

    saver = PostgresSaver(pool)
    saver.setup()  # 首次:建 checkpoint 相关表(幂等,已存在则跳过)
    return saver
