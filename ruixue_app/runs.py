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
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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
# 并发上限:同时最多跑几个 agent。
#
# 【必须有上限】—— 用裸 threading.Thread 是没有上限的:100 个用户同时提问就起
# 100 个线程,checkpointer 连接池(5 条)瞬间排满、100 个并发 LLM 调用把账单和
# 上游限流一起打爆。有界线程池把压力挡在门口,而不是让系统雪崩。
#
# 8 的取法:单机几十人场景下,同时真正在等回答的很少;而每个运行要占
# 1 条 checkpointer 连接 + 1 条业务连接,8 并发对应 ~16 条,PG 上限 100
# 下即使 4 个 worker 也留有余量。可用 MAX_CONCURRENT_RUNS 调。
#
# ⚠⚠ 这个数【绑死在"模型走外部 API"这个前提上】。改成自部署推理服务
#    (vLLM / SGLang)后,上面的理由有一半不成立,必须重新推导:
#
#      · "打爆账单" —— 不存在了。自部署是固定成本,GPU 闲着也是烧钱,
#        压低并发反而是浪费。
#      · "打爆上游限流" —— 变成打爆自己的 GPU 队列,而 vLLM 用
#        continuous batching,【并发越高吞吐越高】(到饱和点为止)。
#        把 app 层死卡在 8 会让 GPU 大量闲置。
#
#    自部署下正确的形状是:**让推理服务器自己排队**(它比我们会调度),
#    app 层的闸门从"限并发"改成"限排队深度 + 超时"。
#    重试策略也要跟着改:API 的 503 是别人限流,退避重试是对的;
#    自部署的排队超时是【自己容量不够】,重试只会让队列更长,该拒绝或扩容。
#
#    真正不变的约束只有一条:PG 的 max_connections(见 engine.py 的算式)。
#
# ── 压测实测记录(2026-08-10)────────────────────────────────
#
# 之前这里写着"这些数字从未压测验证过"。现在压过了,结论和当初的猜测【全都不一样】:
#
#   闸门+DB 路径(不打模型)的饱和曲线,机器 64 核 / 容器限 4GB:
#       worker=2   1.07 GiB   457 req/s   p99  110ms
#       worker=4   2.09 GiB   630 req/s   p99  130ms   ← 最优
#       worker=8   ~4   GiB   341 req/s   p99 2266ms   ← 撞内存顶,开始抖
#       worker=16  4/4  GiB    42 req/s   p99 53428ms  ← 雪崩
#
# **瓶颈不是 CPU(64 核大量闲置),不是数据库(压测中 PG 只有 1 个 active 连接、
#   32 个 idle),是【内存】** —— 每个 worker 要各自加载一份嵌入模型 + 三个树模型,
#   实测约 520MB/worker(线性)。4GB 上限 → 最多 ~7 个。
#
# 这条对 AI 服务是普适的、和普通 Web 服务不一样的经验:
#   **worker 数不能按 CPU 核数拍,它是内存约束,不是 CPU 约束。**
#   照"核数×2"配 128 个 worker,会在启动阶段就把机器打死。
#
# 想再往上扩,正解不是加内存,是【把嵌入模型抽成独立服务】——
# 让所有 worker 共享一份,而不是每个进程各存一份。那之后 worker 数才回到 CPU 约束。
#
# 复现:uv run python scripts/ops/loadtest.py --mode gate --concurrency 80 --requests 2400
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "8"))
# 等待队列上限:超出就直接拒绝,而不是让用户排一个看不到头的队。
# 宁可明确告诉他"现在忙,稍后再试",也不要让他盯着转圈等 5 分钟。
MAX_QUEUED_RUNS = int(os.getenv("MAX_QUEUED_RUNS", "16"))

# 收到停止信号后,最多等在途运行多久(秒)。
#
# 上界由【容器编排给的宽限期】决定:docker stop 默认 10s 后 SIGKILL,
# compose 里已把它调到 60s(stop_grace_period)。这里取 45s,留 15s 给
# 收尾落库 —— 等超过宽限期毫无意义,那时进程已经被强杀了。
SHUTDOWN_DRAIN_SECONDS = int(os.getenv("SHUTDOWN_DRAIN_SECONDS", "45"))

# 线程池【惰性创建】而不是模块加载时就建。
# 理由:shutdown() 会把它彻底关掉,关掉的池不能再 submit。惰性创建让"停机后
# 再启动"这件事天然成立 —— 测试要反复走停机流程,而生产上万一有人写了
# 重启逻辑也不会拿到一个死池。
_executor: ThreadPoolExecutor | None = None
_inflight = 0  # 已提交但未完成的运行数(排队中 + 执行中)
_inflight_lock = threading.Lock()
_shutting_down = False
# 本进程当前负责的 run_id。停机时要能把没跑完的这些【立刻】标记失败 ——
# 否则它们会在库里挂着 running,用户对着转圈等到 15 分钟后才被 reap_stale 清理。
_owned: set[str] = set()


class CapacityError(RuntimeError):
    """系统当前容量已满 —— 调用方应转成 503 告诉用户稍后再试。"""


def _get_executor() -> ThreadPoolExecutor:
    """取线程池,没有就建。调用方已持有 _inflight_lock。"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RUNS, thread_name_prefix="run")
    return _executor


def start_background(run_id: str, target, *args) -> None:
    """把一次运行提交到有界线程池。容量满 / 正在停机时抛 CapacityError。

    为什么用线程而不是 asyncio task:agent 调用链(模型、Milvus、树模型)全是
    同步阻塞的,放进事件循环会把整个 worker 卡死。
    """
    global _inflight
    with _inflight_lock:
        # 已经在停机了就别再收新活 —— 收了也跑不完,只是白花一次模型钱。
        if _shutting_down:
            raise CapacityError("服务正在重启")
        if _inflight >= MAX_CONCURRENT_RUNS + MAX_QUEUED_RUNS:
            raise CapacityError("并发已达上限")
        _inflight += 1
        _owned.add(run_id)
        pool = _get_executor()

    def _release(_future):
        """名额归还。挂在 future 的完成回调上,而不是写在任务体的 finally 里。

        为什么:停机时 cancel_futures=True 会取消【还没开始跑】的任务 ——
        那些任务体一次都不执行,写在 finally 里的归还永远不会发生,计数就永久
        泄漏了:系统明明空闲,却一直以为满载,之后所有请求被 503 挡在门外。
        done_callback 对【正常完成 / 抛异常 / 被取消】三种结局都会触发。
        """
        global _inflight
        with _inflight_lock:
            _inflight -= 1
            _owned.discard(run_id)

    pool.submit(target, *args).add_done_callback(_release)


def inflight_count() -> int:
    """当前在跑 + 排队的运行数(供 /health 或排查用)。"""
    return _inflight


def shutdown(timeout: int = SHUTDOWN_DRAIN_SECONDS) -> int:
    """停机流程:停收新活 → 等在途跑完 → 剩下的立刻标记失败。返回被标记失败的条数。

    ## 不做这件事会怎样

    每次重新部署都会发 SIGTERM。uvicorn 只等【HTTP 请求】排空,而 agent 跑在
    后台线程里、**不绑任何 HTTP 请求** —— 于是线程被直接砍掉:
        · 用户的答案没了,而模型的钱已经花掉了
        · 库里那条记录还写着 running,用户对着转圈等到下次启动 reap_stale
          才被标记失败 —— 那是 15 分钟以后的事

    所以停机要主动做三件事,而不是"什么都不做等被杀":
        1. 关门:新请求直接 503(顺带让负载均衡把流量切走)
        2. 等:给在途的运行一段有界时间跑完 —— 大多数几十秒内能结束
        3. 兜底:还没完的立刻落库为失败,并给用户看得懂的原因

    第 3 步是关键。**宁可明确告诉用户"服务重启了,请重发",也不要让他等一个
    永远不会回来的答案。**

    超时上界由容器宽限期决定,等超过宽限期没有意义 —— 那时已经被 SIGKILL 了。
    """
    global _shutting_down, _executor
    with _inflight_lock:
        _shutting_down = True
        pool, _executor = _executor, None
    # cancel_futures:还在排队、一次都没跑过的直接取消,别浪费宽限期
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _inflight_lock:
            if _inflight == 0:
                logger.info("停机:在途运行已全部完成")
                return 0
        time.sleep(0.2)

    with _inflight_lock:
        stragglers = list(_owned)
    for rid in stragglers:
        try:
            finish_run(rid, error="服务重启,本次运行未能完成,请重新提问")
            publish(rid, {"type": "error", "message": "服务重启,请重新提问"})
        except Exception:  # 停机路径上不能因为一条记录写失败就卡住
            logger.warning("停机标记运行 %s 失败", rid, exc_info=True)
    if stragglers:
        logger.warning("停机:%d 个运行超过 %ds 未完成,已标记失败", len(stragglers), timeout)
    return len(stragglers)
