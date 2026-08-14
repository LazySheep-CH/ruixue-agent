"""异步运行(Run)测试:断线不丢结果 + 重连 + 越权防线 + 残留清理。

要解决的问题:原先 agent 直接在 SSE 请求里跑,客户端一断(刷新页面、切网络、
锁屏),生成器被取消 → agent 半路停下 → 钱花了、结果没有、用户还得重问。
"""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

import ruixue_app.auth as auth
import ruixue_app.main as main_mod
import ruixue_app.runs as runs

_KEY = "runs-test-key"
_H = {"X-API-Key": _KEY}
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")


def _deps_up() -> bool:
    """需要 Redis(事件流)+ PostgreSQL(run 记录)。"""
    try:
        import redis
        from sqlalchemy import text

        from ruixue_agent.persistence.engine import get_engine

        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _deps_up(), reason="需要 Redis + PostgreSQL(docker compose up -d)"
)


class _SlowAgent:
    """慢慢吐字的假 agent —— 好让测试在"跑到一半"时断开。"""

    def stream(self, state, config, stream_mode=None):
        for t in ["新疆", "尉犁", "适合PBAT70"]:
            time.sleep(0.3)
            yield AIMessageChunk(content=t), {}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setattr(auth, "API_KEYS", {_KEY: "alice"})
    monkeypatch.setattr(main_mod, "_agent", _SlowAgent())
    runs._redis.cache_clear()
    yield TestClient(main_mod.app)
    runs._redis.cache_clear()


def _start_and_disconnect(client, after: int = 3) -> str:
    """发起对话,读几个事件就断开,返回 run_id(模拟用户刷新页面)。"""
    run_id = ""
    with client.stream(
        "POST", "/chat/stream", headers=_H, json={"thread_id": "t1", "message": "选配方"}
    ) as r:
        n = 0
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev.get("type") == "run":
                run_id = ev["run_id"]
            n += 1
            if n >= after:
                break
    assert run_id, "流的开头必须下发 run_id,否则客户端无法重连"
    return run_id


# ── 核心:断线后 agent 继续跑完 ────────────────────────────────
def test_agent_finishes_after_client_disconnects(client):
    """这是本次改造的全部意义:客户端断了,agent 照跑完,钱不白花。"""
    run_id = _start_and_disconnect(client)
    for _ in range(40):  # 最多等 4 秒
        time.sleep(0.1)
        r = client.get(f"/chat/runs/{run_id}", headers=_H).json()
        if r["status"] != "running":
            break
    assert r["status"] == "succeeded"
    assert r["answer"] == "新疆尉犁适合PBAT70", "断线后结果应完整落库"


def test_resume_returns_final_answer(client):
    """重连:已结束的运行直接给最终答案,不必重放整个过程。"""
    run_id = _start_and_disconnect(client)
    for _ in range(40):
        time.sleep(0.1)
        if client.get(f"/chat/runs/{run_id}", headers=_H).json()["status"] != "running":
            break
    texts = []
    with client.stream("GET", f"/chat/runs/{run_id}/stream", headers=_H) as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                texts.append(json.loads(line[5:]))
    assert any("PBAT70" in str(e.get("text", "")) for e in texts)
    assert texts[-1]["type"] == "done"


# ── 越权:只能看自己的运行 ────────────────────────────────────
def test_cannot_read_others_run(client, monkeypatch):
    """猜到别人的 run_id 也看不了 —— 否则就能翻别人的对话。"""
    run_id = _start_and_disconnect(client)
    monkeypatch.setattr(auth, "API_KEYS", {_KEY: "alice", "bob-key": "bob"})
    r = client.get(f"/chat/runs/{run_id}", headers={"X-API-Key": "bob-key"})
    assert r.status_code == 404  # 不是 403:不泄露"这个 run 存在"


def test_run_query_requires_auth(client):
    assert client.get("/chat/runs/whatever").status_code == 401


def test_unknown_run_returns_404(client):
    assert client.get("/chat/runs/does-not-exist", headers=_H).status_code == 404


# ── 残留清理:进程被 kill 后不能留下永远 running 的运行 ─────────
def test_reap_stale_marks_orphans_failed(client, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import update
    from sqlalchemy.orm import Session

    from ruixue_agent.persistence.engine import get_engine
    from ruixue_agent.persistence.models import RunRow

    rid = runs.create_run("alice", "alice:t9", "残留测试")
    # 手工把创建时间调老,模拟"进程被 kill 前留下的运行"
    with Session(get_engine()) as s:
        s.execute(
            update(RunRow)
            .where(RunRow.run_id == rid)
            .values(created_at=datetime.now(UTC) - timedelta(hours=2))
        )
        s.commit()

    runs.reap_stale()
    row = runs.get_run(rid, "alice")
    assert row.status == "failed"
    assert "重启" in (row.error or "")  # 给用户看得懂的原因,不是堆栈


# ── 容量闸门:并发必须有上限 ──────────────────────────────────
@pytest.fixture
def drained():
    """容量类测试共享全局在途计数,跑之前/之后都要等它清空,否则互相干扰
    (前一个测试没跑完的任务会把后一个直接顶到容量上限)。"""

    # 这些用例共享模块级全局(计数、线程池、停机标志)。不复位的话,
    # 前一个用例的残留会让后一个直接顶到容量上限或被"停机中"拒绝。
    runs._shutting_down = False

    def wait():
        for _ in range(60):
            if runs.inflight_count() == 0:
                return
            time.sleep(0.1)
        raise AssertionError("在途任务未清空,可能有名额泄漏")

    wait()
    yield
    runs._shutting_down = False
    wait()


def test_capacity_gate_rejects_when_full(drained, monkeypatch):
    """裸线程没有上限:100 人同时提问就起 100 个线程,连接池排满、
    并发 LLM 调用把账单和上游限流一起打爆。必须在门口挡住。"""
    monkeypatch.setattr(runs, "MAX_CONCURRENT_RUNS", 2)
    monkeypatch.setattr(runs, "MAX_QUEUED_RUNS", 2)

    def slow(_):
        time.sleep(0.8)

    accepted = rejected = 0
    for i in range(8):
        try:
            runs.start_background(f"cap{i}", slow, i)
            accepted += 1
        except runs.CapacityError:
            rejected += 1
    assert accepted == 4 and rejected == 4  # 容量 = 并发 + 队列


def test_inflight_returns_to_zero(drained, monkeypatch):
    """在途计数必须归零 —— 泄漏会让系统在"看起来空闲"时拒绝新请求。"""
    monkeypatch.setattr(runs, "MAX_CONCURRENT_RUNS", 2)
    monkeypatch.setattr(runs, "MAX_QUEUED_RUNS", 4)
    for i in range(3):
        runs.start_background(f"zero{i}", lambda _: time.sleep(0.1), i)
    for _ in range(50):
        time.sleep(0.1)
        if runs.inflight_count() == 0:
            break
    assert runs.inflight_count() == 0


def test_failed_task_also_releases_slot(drained, monkeypatch):
    """任务抛异常也要释放名额,否则几次失败就把容量占死。"""
    monkeypatch.setattr(runs, "MAX_CONCURRENT_RUNS", 2)
    monkeypatch.setattr(runs, "MAX_QUEUED_RUNS", 2)

    def boom(_):
        raise RuntimeError("模拟运行失败")

    runs.start_background("boom1", boom, 1)
    for _ in range(30):
        time.sleep(0.05)
        if runs.inflight_count() == 0:
            break
    assert runs.inflight_count() == 0, "失败的任务没有释放名额"


# ── 停机:每次重新部署都会走这条路 ──────────────────────────────
def test_shutdown_rejects_new_runs(drained, monkeypatch):
    """停机时还收新活 = 白花一次模型钱(它注定跑不完)。"""
    monkeypatch.setattr(runs, "_shutting_down", True)
    try:
        with pytest.raises(runs.CapacityError, match="重启"):
            runs.start_background("sd1", lambda: None)
    finally:
        monkeypatch.setattr(runs, "_shutting_down", False)


def test_shutdown_waits_for_inflight_then_returns_zero(drained, monkeypatch):
    """在途的要给机会跑完 —— 大多数几十秒内能结束,不该一律砍掉。"""
    monkeypatch.setattr(runs, "MAX_CONCURRENT_RUNS", 2)
    runs.start_background("sd2", lambda: time.sleep(0.3))
    try:
        assert runs.shutdown(timeout=10) == 0, "在途跑完了就不该有被标记失败的"
    finally:
        runs._shutting_down = False


def test_shutdown_marks_stragglers_failed_immediately(drained, monkeypatch):
    """超过宽限期还没完的,必须立刻落库为失败。

    不做的话它在库里挂着 running,用户要对着转圈等到 15 分钟后 reap_stale 才知道。
    宁可明确说"服务重启了,请重发",也不要让人等一个永远不回来的答案。
    """
    monkeypatch.setattr(runs, "MAX_CONCURRENT_RUNS", 2)
    rid = runs.create_run("alice", "alice:sd", "停机测试")
    runs.start_background(rid, lambda: time.sleep(3))
    try:
        assert runs.shutdown(timeout=1) == 1
        row = runs.get_run(rid, "alice")
        assert row.status == "failed" and "重启" in (row.error or "")
    finally:
        runs._shutting_down = False
        for _ in range(60):
            if runs.inflight_count() == 0:
                break
            time.sleep(0.1)
