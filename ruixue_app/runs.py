"""Run(一次 agent 运行)的生命周期:创建 → 后台执行 → 事件流 → 查询/重连。

## 解决什么问题

原先 agent 直接在 SSE 请求里跑:客户端一断(刷新页面、切网络、手机锁屏),
生成器被取消 → agent 半路停下 → **钱花了、结果没有、用户还得重问**。

改成:
    创建 Run(PG) → 【后台任务】跑 agent(不绑请求生命周期)
                  → 事件写 Redis Stream → SSE 从 Stream 读并推给客户端

客户端断了 agent 照跑完;重连时凭 run_id 把已产生的事件**从头补发**,
再接着推后续的 —— 刷新页面不丢进度,也不用重新花钱。

## 为什么事件放 Redis 而不是内存

上线会起多个 worker。用户刷新后的请求**可能落到另一个 worker**,
若事件存在产生它的那个进程的内存里,另一个进程读不到 —— 重连就失败了。
Redis Stream 是跨进程共享的,谁都能读。

## 各自的职责边界

    PostgreSQL(runs 表) = 权威状态,持久;重启后靠它知道谁是残留
    Redis Stream         = 事件缓存,有 TTL,丢了只影响"回放过程",不影响最终答案
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import UTC, datetime, timedelta
from functools import cache

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.models import RunRow

logger = logging.getLogger("ruixue.runs")

STREAM_TTL = 3600  # 事件流保留 1 小时:够用户刷新/重连,又不会把 Redis 撑爆
# 超过这个时长仍是 running 的,判定为进程异常退出留下的残留(见 reap_stale)
STALE_AFTER = timedelta(minutes=15)


@cache
def _redis():
    """Redis 客户端(懒加载单例)。没配 REDIS_URL 返回 None —— 退化为不可重连。"""
    url = os.getenv("REDIS_URL", "")
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url, decode_responses=True)


def _stream_key(run_id: str) -> str:
    return f"ruixue:run:{run_id}"


# ── 事件流 ────────────────────────────────────────────────────
def publish(run_id: str, event: dict) -> None:
    """把一个事件追加到该 Run 的流里。Redis 不可用时静默跳过(不影响 agent 跑)。"""
    r = _redis()
    if r is None:
        return
    try:
        key = _stream_key(run_id)
        r.xadd(key, {"d": json.dumps(event, ensure_ascii=False)})
        r.expire(key, STREAM_TTL)
    except Exception as e:  # 事件流是"锦上添花",挂了不能拖垮正在跑的 agent
        logger.warning("事件发布失败(%s),忽略", type(e).__name__)


def read_events(run_id: str, last_id: str = "0-0", block_ms: int = 15000):
    """阻塞读取事件。从 last_id 之后开始 —— 传 "0-0" 就是【从头补发】(重连场景)。

    每次返回 (事件id, 事件内容) 列表;超时返回空列表(让调用方有机会发心跳)。
    """
    r = _redis()
    if r is None:
        return []
    try:
        resp = r.xread({_stream_key(run_id): last_id}, count=100, block=block_ms)
    except Exception as e:
        logger.warning("事件读取失败(%s)", type(e).__name__)
        return []
    out = []
    for _key, entries in resp or []:
        for eid, fields in entries:
            try:
                out.append((eid, json.loads(fields["d"])))
            except Exception:
                continue
    return out


# ── Run 记录(PostgreSQL:权威状态)─────────────────────────────
def create_run(user_id: str, thread_id: str, question: str) -> str:
    run_id = str(uuid.uuid4())
    with Session(get_engine()) as s:
        s.add(RunRow(run_id=run_id, user_id=user_id, thread_id=thread_id, question=question[:4000]))
        s.commit()
    return run_id


def finish_run(run_id: str, *, answer: str | None = None, error: str | None = None) -> None:
    """标记结束。error 必须是【脱敏】的短原因 —— 堆栈只进日志,不给用户看。"""
    with Session(get_engine()) as s:
        s.execute(
            update(RunRow)
            .where(RunRow.run_id == run_id)
            .values(
                status="succeeded" if error is None else "failed",
                answer=answer,
                error=error[:255] if error else None,
                finished_at=datetime.now(UTC),
            )
        )
        s.commit()


def get_run(run_id: str, user_id: str) -> RunRow | None:
    """按 run_id 取,并校验归属 —— 否则猜到别人的 run_id 就能看别人的对话(越权)。"""
    with Session(get_engine()) as s:
        return s.scalar(select(RunRow).where(RunRow.run_id == run_id, RunRow.user_id == user_id))


def reap_stale() -> int:
    """把"跑太久还 running"的 Run 标记为失败。启动时调用一次。

    为什么需要:执行是进程内的后台任务,进程被 kill / 机器重启时,任务没了但
    数据库里还写着 running —— 用户会一直等一个永远不会完成的 Run。
    """
    cutoff = datetime.now(UTC) - STALE_AFTER
    with Session(get_engine()) as s:
        result = s.execute(
            update(RunRow)
            .where(RunRow.status == "running", RunRow.created_at < cutoff)
            .values(
                status="failed",
                error="服务重启导致本次运行中断,请重新提问",
                finished_at=datetime.now(UTC),
            )
        )
        s.commit()
        n = result.rowcount or 0
    if n:
        logger.warning("清理了 %d 个残留的 running 运行(进程异常退出所致)", n)
    return n


# ── 后台执行 ──────────────────────────────────────────────────
def start_background(run_id: str, target, *args) -> None:
    """在后台线程里执行 target(*args)。

    为什么用线程而不是 asyncio task:agent 调用链(模型、Milvus、树模型)全是
    同步阻塞的,放进事件循环会把整个 worker 卡死。线程池本来就是 FastAPI 跑
    同步端点的方式,这里只是把它从"请求生命周期"里解绑。

    daemon=True:进程退出时不等它 —— 残留状态由 reap_stale 兜底,
    比让进程卡住等一个长任务更可取。
    """
    threading.Thread(target=target, args=args, daemon=True, name=f"run-{run_id[:8]}").start()
