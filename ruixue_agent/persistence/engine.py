"""数据库连接管理。

为什么用【连接池】而不是每次新建连接:
    建一条 PG 连接要 TCP 握手 + 认证 + fork 后端进程,几十毫秒。
    灌 26 万 chunk 时如果每次新建,光建连接就能耗掉几十分钟。
    连接池 = 一开始建好 N 条,用完还回去,下次直接拿。

为什么用【同步】而不是 async(参考架构 用的是 async):
    参考架构 是 web 服务:大量并发请求、每个请求大部分时间在等 IO → async 能让一个线程服务上千请求。
    我们是 ETL 批处理:一个进程灌数据、瓶颈在数据库写入而不是等待 → async 只增加复杂度,不提速。
    选型看【工作负载】,不是看哪个词更时髦。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine

_DOCKER_ENV = Path(__file__).resolve().parent.parent.parent / "docker" / ".env"


def database_url() -> str:
    """从 docker/.env 拼出连接串 —— 凭据只有一处定义,不在代码里硬编码。

    (本地开发这样够用;上线要换成密钥管理服务,而不是把密码放文件里。)
    """
    load_dotenv(_DOCKER_ENV)
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    port = os.getenv("POSTGRES_PORT", "5432")
    # postgresql+psycopg = psycopg3 驱动(不是老的 psycopg2)
    return f"postgresql+psycopg://{user}:{password}@localhost:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """全进程共用一个 Engine(内含连接池)。lru_cache 保证只建一次。

    池大小怎么定 —— 要和【实际并发】对齐,不是拍个数:

        每个 worker 进程的用量 ≈ 后台运行并发(默认 8,见 runs.MAX_CONCURRENT_RUNS)
                                + SSE/查询等短请求若干
        本池 10+10 = 20 条,再加 checkpointer 自己的池(5 条)≈ 25 条/worker

        PostgreSQL 默认 max_connections=100 → 最多约 4 个 worker(100 ÷ 25)。
        起更多 worker 前必须同步调大 PG 的 max_connections,否则新 worker
        会拿不到连接、请求直接失败 —— 这是扩容时最容易踩的坑。

    池太小会怎样:请求排队等连接,表现为"莫名其妙变慢",很难查。
    池太大也不行:PG 每条连接都是一个进程,几百条会把数据库拖垮。
    """
    return create_engine(
        database_url(),
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),  # 常驻连接数
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),  # 高峰可临时再开
        pool_timeout=10,  # 等连接最多 10 秒 —— 超了快速失败,别让请求无限挂着
        pool_recycle=1800,  # 30 分钟回收一次:防止被数据库/中间件掐掉的僵死连接
        pool_pre_ping=True,  # 取连接前先 ping:防止拿到已被服务端断掉的死连接
        echo=False,  # 调试时改 True 可以看到每条真实 SQL
    )
