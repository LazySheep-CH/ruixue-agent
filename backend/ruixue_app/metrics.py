"""运行指标:把现有状态汇总成 Prometheus 文本格式,从 /metrics 暴露。

数据本来就有(runs 表、线程池计数、Redis、pg 表大小),这里只做汇总输出,
不引入新的采集或存储。格式选 Prometheus 而不是自定义 JSON,是为了以后接真的
Prometheus 时应用侧不用改;现阶段用 scripts/check_alerts.py 挂 cron 读它就够。

注意两点:
- 任何一块取数失败都不能让整个端点 5xx,否则数据库一抖,监控就误报服务挂了。
  失败的块只计入 ruixue_metrics_errors。
- 带窗口的指标把窗口写进名字(ruixue_runs_failed_1h),省得读的人猜口径。
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime, timedelta

logger = logging.getLogger("ruixue.metrics")

# 统计窗口。1 小时:短到能及时发现,长到不会因为几次抖动就报警。
WINDOW_HOURS = 1


def _fmt(name: str, value: float, help_text: str, labels: str = "", kind: str = "gauge") -> str:
    """渲染一条指标。labels 单独传,不能拼进 name。

    Prometheus 的格式规定:`# HELP` / `# TYPE` 后面跟的是【不带标签的指标名】,
    而且同一个名字只能声明一次。写成
        # HELP ruixue_dep_up{dep="redis"} ...
    是不合法的 —— 解析器会拒绝或行为未定义。
    这类错误在我们自己看输出时完全正常(肉眼读得懂),只有真接上采集器才会暴露。
    """
    body = f"{name}{labels} {value}\n"
    return f"# HELP {name} {help_text}\n# TYPE {name} {kind}\n{body}"


def _series(name: str, value: float, labels: str) -> str:
    """同名指标的后续样本:只出数据行,不重复 HELP/TYPE。"""
    return f"{name}{labels} {value}\n"


def _runs_window() -> dict:
    """最近 WINDOW_HOURS 小时的运行统计。数据库不可用时返回空 —— 见下方说明。"""
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    from ruixue_agent.persistence.engine import get_engine
    from ruixue_agent.persistence.models import RunRow

    since = datetime.now(UTC) - timedelta(hours=WINDOW_HOURS)
    with Session(get_engine()) as s:
        rows = s.execute(
            select(RunRow.status, func.count())
            .where(RunRow.created_at >= since)
            .group_by(RunRow.status)
        ).all()
    return {status: n for status, n in rows}


def _table_bytes() -> dict:
    """几张关键表的物理大小。checkpoints 是已知会无限增长的那张 ——
    没有这个指标,它涨到磁盘满之前不会有任何人发现。"""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from ruixue_agent.persistence.engine import get_engine

    with Session(get_engine()) as s:
        rows = s.execute(
            text(
                "select relname, pg_total_relation_size(relid) "
                "from pg_catalog.pg_statio_user_tables "
                "where relname in ('checkpoints','checkpoint_writes','checkpoint_blobs',"
                "'runs','memories','datasets','chunks')"
            )
        ).all()
    return {name: int(size) for name, size in rows}


def render() -> str:
    """生成 Prometheus 文本。

    任何一块取数失败都不能让整个 /metrics 挂掉:
    监控端点自己变成故障点是很讽刺的一件事:数据库抖一下,/metrics 返回 500,
    于是监控系统认为服务挂了,而实际上服务好好的。
    所以每一块单独 try,失败就少几个指标 + 把 `ruixue_metrics_errors` 加一,
    那个数本身就是"有些指标取不到"的信号。
    """
    out: list[str] = []
    errors = 0

    out.append(_fmt("ruixue_up", 1, "进程存活(能返回这个响应就是 1)"))

    # ── 运行成败:最能反映"服务活着但不干活" ──
    try:
        w = _runs_window()
        ok, failed, running = w.get("succeeded", 0), w.get("failed", 0), w.get("running", 0)
        total = ok + failed + running
        out.append(_fmt("ruixue_runs_total_1h", total, f"最近 {WINDOW_HOURS}h 的运行总数"))
        out.append(_fmt("ruixue_runs_succeeded_1h", ok, "成功数"))
        out.append(_fmt("ruixue_runs_failed_1h", failed, "失败数"))
        out.append(_fmt("ruixue_runs_running", running, "当前状态为 running 的运行数"))
        # 失败率单独给一个数,免得告警规则里再做除法(除零要处理,容易写错)
        out.append(
            _fmt(
                "ruixue_runs_failure_ratio_1h",
                round(failed / total, 4) if total else 0.0,
                "失败占比;总数为 0 时记 0(没有请求不等于有问题)",
            )
        )
    except Exception:
        errors += 1
        logger.warning("运行统计取数失败", exc_info=True)

    # ── 并发:队列积压说明容量不够 ──
    try:
        from ruixue_app import runs

        out.append(_fmt("ruixue_inflight_runs", runs._inflight, "在途运行数(排队中 + 执行中)"))
        out.append(
            _fmt(
                "ruixue_capacity_runs",
                runs.MAX_CONCURRENT_RUNS + runs.MAX_QUEUED_RUNS,
                "容量上限(并发 + 排队);在途逼近它就该扩容",
            )
        )
        out.append(_fmt("ruixue_shutting_down", int(runs._shutting_down), "是否正在优雅停机"))
    except Exception:
        errors += 1

    # ── 依赖是否可达。这三个是"服务活着但不干活"的直接原因 ──
    #
    # 首次探测会慢(Milvus 客户端冷启动约 11s),但生产环境启动时
    # lifespan 里的 ensure_collection() 已经把它建好了 —— 采集时永远是热的。
    up_lines, ms_lines = [], []
    for name, probe in (
        ("postgres", _probe_pg),
        ("redis", _probe_redis),
        ("milvus", _probe_milvus),
    ):
        t0 = time.perf_counter()
        try:
            alive = probe()
        except Exception:
            alive = False
        lb = f'{{dep="{name}"}}'
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        up_lines.append((int(alive), lb))
        ms_lines.append((elapsed, lb))

    for i, (v, lb) in enumerate(up_lines):
        out.append(
            _fmt("ruixue_dep_up", v, "依赖是否可达", lb)
            if i == 0
            else _series("ruixue_dep_up", v, lb)
        )
    for i, (v, lb) in enumerate(ms_lines):
        out.append(
            _fmt("ruixue_dep_probe_ms", v, "依赖探测耗时(毫秒)", lb)
            if i == 0
            else _series("ruixue_dep_probe_ms", v, lb)
        )

    # ── 存储增长:checkpoints 是已知的慢性病 ──
    try:
        for i, (tbl, size) in enumerate(sorted(_table_bytes().items())):
            lb = f'{{table="{tbl}"}}'
            out.append(
                _fmt("ruixue_table_bytes", size, "表物理大小(字节)", lb)
                if i == 0
                else _series("ruixue_table_bytes", size, lb)
            )
    except Exception:
        errors += 1

    out.append(_fmt("ruixue_metrics_errors", errors, "本次采集失败的指标块数;>0 说明有东西取不到"))
    return "".join(out)


def _probe_pg() -> bool:
    from sqlalchemy import text

    from ruixue_agent.persistence.engine import get_engine

    with get_engine().connect() as c:
        c.execute(text("SELECT 1"))
    return True


def _probe_redis() -> bool:
    url = os.getenv("REDIS_URL", "")
    if not url:
        return False
    import redis

    return bool(redis.Redis.from_url(url, socket_connect_timeout=2).ping())


def _probe_milvus() -> bool:
    """探测 Milvus。必须复用已缓存的客户端,不能每次新建。

    实测:`MilvusClient(uri=...)` 每次新建耗时 10.4 秒(建连接 + 握手),
    而复用缓存客户端首次 41ms、之后 6~8ms —— 差 250 倍。

    一个 10 秒的 /metrics 端点是不可用的:采集器会超时,于是
    监控系统认为服务挂了,而服务好好的。监控自己变成故障源,
    比没有监控更坏 —— 它会制造假警报,然后所有人开始忽略警报。

    (`memory.store._client` 带 lru_cache,进程内单例。)
    """
    from ruixue_agent.memory.store import _client

    _client().list_collections()
    return True
