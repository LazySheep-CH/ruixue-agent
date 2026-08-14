"""用量配额测试:跨进程共享 + 降级 + 端点接入。

为什么需要配额:限流(20次/分)管的是【频率】,不管【总量】——
按 20/分钟老实调,一天也能调 28800 次,足够烧光整月的模型额度。
频率限流防"瞬间打爆",配额防"细水长流地烧钱",两者都要。
"""

import os

import pytest
from fastapi import HTTPException

import ruixue_app.quota as quota

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")


def _redis_up() -> bool:
    try:
        import redis

        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
        return True
    except Exception:
        return False


needs_redis = pytest.mark.skipif(not _redis_up(), reason="Redis 未启动(docker compose up -d redis)")


@pytest.fixture
def q(monkeypatch):
    """每个测试用独立用户名 + 干净计数,避免相互干扰。"""
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    quota._redis.cache_clear()
    yield quota
    quota._redis.cache_clear()


# ── 未配置 Redis:配额不启用(本地开发)────────────────────────
def test_disabled_without_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    quota._redis.cache_clear()
    assert quota.consume("anyone") == (True, 0)
    quota._redis.cache_clear()


# ── 计数与拦截 ────────────────────────────────────────────────
@needs_redis
def test_counts_and_blocks_over_limit(q, monkeypatch):
    monkeypatch.setattr(q, "DAILY_LIMIT", 3)
    user = "quota-test-block"
    q._redis().delete(q._today_key(user))
    assert [q.consume(user) for _ in range(3)] == [(True, 1), (True, 2), (True, 3)]
    allowed, used = q.consume(user)
    assert allowed is False and used == 4


@needs_redis
def test_shared_across_processes(q, monkeypatch):
    """核心:计数必须跨进程共享,否则多 worker 时配额 ×N 形同虚设。

    用 cache_clear() 重建连接来模拟"另一个进程"。
    """
    monkeypatch.setattr(q, "DAILY_LIMIT", 4)
    user = "quota-test-shared"
    q._redis().delete(q._today_key(user))

    [q.consume(user) for _ in range(2)]  # worker A
    q._redis.cache_clear()  # 换个"进程"
    _, used = q.consume(user)  # worker B
    assert used == 3, "计数没有跨进程共享"


@needs_redis
def test_users_are_isolated(q, monkeypatch):
    monkeypatch.setattr(q, "DAILY_LIMIT", 2)
    for u in ("quota-iso-a", "quota-iso-b"):
        q._redis().delete(q._today_key(u))
    [q.consume("quota-iso-a") for _ in range(2)]
    assert q.consume("quota-iso-b")[0] is True  # B 不受 A 影响


@needs_redis
def test_key_is_per_day(q):
    """按 UTC 日期分桶 —— 天然按天重置,不需要定时任务清零。"""
    assert q._today_key("u") != q._today_key("u").replace("2", "9", 1)
    from datetime import UTC, datetime

    assert datetime.now(UTC).strftime("%Y%m%d") in q._today_key("u")


# ── 降级:Redis 挂了要放行(配额是成本控制,不是安全边界)──────
def test_degrades_open_when_redis_fails(monkeypatch):
    """不使用 q fixture:这里要把 _redis 换成普通函数,fixture 的 cache_clear 会失效。"""

    class Broken:
        def incr(self, *a, **k):
            raise ConnectionError("redis down")

    monkeypatch.setattr(quota, "_redis", lambda: Broken())
    allowed, _ = quota.consume("anyone")
    assert allowed is True, "Redis 挂了不该把真实用户挡在门外"


# ── 依赖注入:超限抛 429 ──────────────────────────────────────
def test_enforce_raises_429(monkeypatch):
    monkeypatch.setattr(quota, "consume", lambda uid: (False, 999))
    with pytest.raises(HTTPException) as e:
        quota.enforce_quota("someone")
    assert e.value.status_code == 429
    assert "上限" in e.value.detail


def test_enforce_passes_through_user_id(monkeypatch):
    monkeypatch.setattr(quota, "consume", lambda uid: (True, 1))
    assert quota.enforce_quota("alice") == "alice"
