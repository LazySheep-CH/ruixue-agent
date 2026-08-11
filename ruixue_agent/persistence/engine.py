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
    load_dotenv(_DOCKER_ENV)  # 不覆盖已存在的环境变量 → 容器里 compose 注入的优先
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ["POSTGRES_DB"]
    # 主机名必须可配:本机开发连回环,但【容器里的 localhost 是容器自己】,
    # 得连服务名 postgres。写死的话 app 一进容器就连不上数据库。
    #
    # ⚠ 默认值用 127.0.0.1 而不是 localhost —— 实测差 200 倍(0.05s vs 10.13s):
    #   compose 里端口只绑了 IPv4 回环(安全加固),而 Windows 把 localhost 先解析成
    #   IPv6 的 ::1,每次建连都要先等 10 秒超时才退回 IPv4。连接池启动要开 4 条,
    #   4×10s 直接把 PoolTimeout(30s)打爆 —— 表现为"本机突然连不上库"。
    #   和 nginx 健康检查用 localhost 失败是同一个病:localhost ≠ 127.0.0.1。
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    # 同理端口:容器间走内网 5432,不受宿主机端口映射影响。
    port = os.getenv("POSTGRES_PORT", "5432") if host in ("localhost", "127.0.0.1") else "5432"
    # postgresql+psycopg = psycopg3 驱动(不是老的 psycopg2)
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """全进程共用一个 Engine(内含连接池)。lru_cache 保证只建一次。

    池大小怎么定 —— 要和【实际并发】对齐,不是拍个数:

        ⚠ 【2026-08-10 实测修正】原来这里按"每 worker 满池 30 条"推算,
          得出"最多 3~4 个 worker"。**压测实测发现这个算式过于保守约一倍**:

              4 个 worker 压测中,PG 实际连接数只有 54/100(不是预测的 120)

          原因:SQLAlchemy 的连接池是【懒建】的 —— pool_size=10 是"最多留 10 条
          空闲",不是"启动就占 10 条"。实际用多少建多少。
          所以那个算式是【最坏情况上界】,不是实际用量;拿它当扩容红线会
          白白限制住本可以加的 worker。

        真正的红线【不是连接数,是内存】—— 见 runs.py 里 MAX_CONCURRENT_RUNS
        上面的压测记录:每个 worker 要各自加载一份嵌入模型 + 三个树模型,
        约 520MB;4GB 上限下最多 7 个,8 个就撞顶、16 个直接雪崩。

        连接数仍要监控(下面这条 SQL),但它不是当前的瓶颈:
            SELECT count(*) FROM pg_stat_activity WHERE datname='ruixue';

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
