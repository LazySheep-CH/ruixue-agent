"""认证：从请求头取 API Key，校验后返回 user_id。

对标 Spring Security 的认证过滤器：验身份 → 拿到当前用户。
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# 演示用的"合法 API Key → user_id"表。
# 生产上这里换成查数据库/密钥服务（就像 Spring 的 UserDetailsService）。
_API_KEYS: dict[str, str] = {
    "demo-key-alice": "alice",
    "demo-key-bob": "bob",
}

# 声明"从请求头 X-API-Key 取值"。auto_error=False：取不到我们自己处理(返回更清楚的错)。
# 它还会让 /docs 自动出现一个"填 API Key"的输入框。
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_user(api_key: str | None = Security(_api_key_header)) -> str:
    """FastAPI 依赖：校验 API Key，返回 user_id；无效则抛 401。"""
    # ===== (你写这几行) =====
    # 1. 如果 api_key 是 None 或不在 _API_KEYS 里 → 抛 401：
    #      raise HTTPException(
    #          status_code=status.HTTP_401_UNAUTHORIZED,
    #          detail="无效的 API Key",
    #      )
    # 2. 否则返回对应的 user_id：return _API_KEYS[api_key]
    if api_key is None or api_key not in _API_KEYS:
        raise HTTPException(
                 status_code=status.HTTP_401_UNAUTHORIZED,
                 detail="无效的 API Key",
             )
    return _API_KEYS[api_key]