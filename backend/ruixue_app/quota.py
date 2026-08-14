"""用量配额:限制每个用户【每天能用多少】,护住 API 账单。

为什么限流不够:
限流(20 次/分钟)管的是频率,不管总量。一个用户老老实实按 20/分钟 调,
一天也能调 28800 次 —— 足够烧光整月的模型额度。频率限流防的是"瞬间打爆",
配额防的是"细水长流地烧钱"。两者都要。

为什么必须放 Redis:
和限流同理:配额计数若存进程内存,多 worker 就是各算各的,配额直接失效。
且配额是"花掉多少钱"的账,进程重启不该归零(Redis 开了 appendonly 持久化)。

降级策略:
Redis 挂了怎么办?这里选择放行并告警,而不是拒绝服务:
配额是成本控制手段,不是安全边界 —— 因为计数器挂了就把所有真实用户挡在门外,
代价大于短时间内多花一点钱。安全相关的门(认证)才必须 fail closed。
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from functools import cache

from fastapi import Depends, HTTPException, status

from ruixue_app.auth import get_current_user

logger = logging.getLogger("ruixue.quota")

# 每用户每日对话次数上限。取值依据:正常用户一天问几十次已属重度使用;
# 200 次足够覆盖真实需求,又能挡住脚本刷量。可用环境变量调。
DAILY_LIMIT = int(os.getenv("DAILY_CHAT_QUOTA", "200"))
KEY_PREFIX = "ruixue:quota:chat"


@cache
def _redis():
    """Redis 客户端(懒加载单例)。没配 REDIS_URL 返回 None —— 配额不启用。"""
    url = os.getenv("REDIS_URL", "")
    if not url:
        return None
    import redis

    return redis.Redis.from_url(url, decode_responses=True)


def _today_key(user_id: str) -> str:
    """按 UTC 日期分桶:天然按天重置,不需要定时任务去清零。"""
    return f"{KEY_PREFIX}:{datetime.now(UTC):%Y%m%d}:{user_id}"


def consume(user_id: str) -> tuple[bool, int]:
    """消耗一次配额。返回 (是否放行, 今日已用次数)。

    用 INCR + EXPIRE:INCR 是原子操作,多 worker 并发下计数也不会丢
    (若用"读-改-写"就会有竞态,两个 worker 同时读到 99 都写 100)。
    """
    r = _redis()
    if r is None:
        return True, 0  # 未配置 Redis:配额不启用(本地开发)
    key = _today_key(user_id)
    try:
        used = r.incr(key)
        if used == 1:
            r.expire(key, 2 * 24 * 3600)  # 首次计数时设 2 天过期,自动清理旧桶
    except Exception as e:
        # 降级放行:配额是成本控制,不是安全边界。计数器挂了不该挡住真实用户。
        logger.warning("配额计数失败(%s),本次放行", type(e).__name__)
        return True, 0
    return used <= DAILY_LIMIT, used


def enforce_quota(user_id: str = Depends(get_current_user)) -> str:
    """FastAPI 依赖:检查并消耗配额,超限返回 429。放行时返回 user_id。

    放在 /chat 这类【会花钱】的端点上;查询类端点(健康检查、查土壤)不必。
    """
    allowed, used = consume(user_id)
    if not allowed:
        logger.warning("用户 %s 超出每日配额(%d/%d)", user_id, used, DAILY_LIMIT)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"已达每日使用上限({DAILY_LIMIT} 次),请明天再试",
        )
    return user_id
