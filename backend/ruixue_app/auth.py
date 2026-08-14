"""认证:识别"当前是谁",返回用于数据隔离的 user_id。

支持两种凭证(按优先级):
  1. **JWT**(Authorization: Bearer xxx)—— 网页用户登录后走这条
  2. **API Key**(X-API-Key)—— 程序化调用/脚本/演示走这条

为什么保留两条:网页用户要的是"注册登录"体验;而机器对机器调用用 API Key
更简单(不需要登录流程)。这也是 OpenAI / Anthropic 等平台的通行做法。

返回的 user_id 会拼进 thread_id 做【命名空间隔离】(见 main.py),
保证用户只能读写自己的会话。
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from ruixue_app.security.tokens import TokenError, decode_access_token


# ── 程序化调用的 API Key(从环境变量读,不再硬编码)──────────────
# 格式:RUIXUE_API_KEYS="key1:user1,key2:user2";不配则只能用账号登录。
def _load_api_keys() -> dict[str, str]:
    raw = os.getenv("RUIXUE_API_KEYS", "")
    keys: dict[str, str] = {}
    for item in raw.split(","):
        if ":" in item:
            k, _, u = item.partition(":")
            if k.strip() and u.strip():
                keys[k.strip()] = u.strip()
    return keys


API_KEYS = _load_api_keys()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# auto_error=False:没带 token 时不直接报错,交给我们统一处理(好给中文提示)
_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="未登录或凭证无效",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer),
    api_key: str | None = Security(_api_key_header),
) -> str:
    """FastAPI 依赖:校验凭证,返回 user_id(字符串)。失败抛 401。"""
    # ① JWT 优先(网页登录用户)
    if bearer is not None and bearer.credentials:
        try:
            payload = decode_access_token(bearer.credentials)
        except TokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            ) from e
        return f"u{payload['sub']}"  # 加前缀,避免与 API Key 的用户名撞车

    # ② API Key(程序化调用)
    if api_key and api_key in API_KEYS:
        return API_KEYS[api_key]

    raise _UNAUTHORIZED
