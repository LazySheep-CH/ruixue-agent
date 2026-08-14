"""JWT 访问令牌:签发与校验。

为什么用 JWT 而不是服务端 session:
    session 要在服务端存状态,多实例部署得共享存储(Redis);
    JWT 把身份信息签名后交给客户端保管,服务端无状态,水平扩容天然友好。
    代价:签发后无法立即作废(除非维护黑名单)——所以有效期不能太长。

安全要点:
- 密钥从环境变量读(RUIXUE_JWT_SECRET),绝不硬编码进代码;未配置时生成随机密钥,
  这样重启即失效,提醒你生产必须显式配置。
- 只放 user_id / username / 过期时间,不放敏感信息 —— JWT 是签名不是加密,
  任何人都能解开看内容,它保证的是"没被篡改",不是"看不见"。
"""

from __future__ import annotations

import os
import secrets
from datetime import UTC, datetime, timedelta

import jwt

ALGORITHM = "HS256"
TOKEN_TTL_DAYS = int(os.getenv("RUIXUE_TOKEN_TTL_DAYS", "7"))

# 未配置密钥时用随机值:开发能跑,但重启后旧 token 全失效 —— 逼你上线前配好。
_SECRET = os.getenv("RUIXUE_JWT_SECRET") or secrets.token_urlsafe(32)


class TokenError(Exception):
    """令牌无效/过期。"""


def create_access_token(user_id: int, username: str) -> str:
    """签发访问令牌。sub=用户ID(JWT 标准字段),exp=过期时间(库会自动校验)。"""
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(days=TOKEN_TTL_DAYS),
    }
    return jwt.encode(payload, _SECRET, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """校验并解析令牌;失败抛 TokenError(不泄露具体原因给外部)。"""
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as e:
        raise TokenError("登录已过期,请重新登录") from e
    except jwt.InvalidTokenError as e:
        raise TokenError("登录凭证无效") from e

    if "sub" not in payload:
        raise TokenError("登录凭证无效")
    return payload
