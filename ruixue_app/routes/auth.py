"""认证路由:注册 / 登录 / 查当前用户。

放在 routes/ 而不是塞进 main.py:main 是装配入口,路由按领域分文件,
将来加"会话历史""设置"等路由各占一个文件,不会让 main 膨胀成几千行。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ruixue_app.auth import get_current_user
from ruixue_app.security.tokens import TOKEN_TTL_DAYS, create_access_token
from ruixue_app.security.users import UserError, authenticate, create_user

router = APIRouter(prefix="/auth", tags=["认证"])


class Credentials(BaseModel):
    """注册/登录的请求体。长度上限防超长输入。"""

    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_in_days: int = TOKEN_TTL_DAYS


@router.post("/register", response_model=TokenResponse, summary="注册")
def register(body: Credentials) -> TokenResponse:
    """注册成功后直接签发令牌 —— 免去"注册完还要再登录一次"的多余步骤。"""
    try:
        user = create_user(body.username, body.password)
    except UserError as e:
        # 409 Conflict:资源冲突(用户名已被占用),比笼统的 400 更准确
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    return TokenResponse(
        access_token=create_access_token(user.id, user.username), username=user.username
    )


@router.post("/login", response_model=TokenResponse, summary="登录")
def login(body: Credentials) -> TokenResponse:
    try:
        user = authenticate(body.username, body.password)
    except UserError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e
    return TokenResponse(
        access_token=create_access_token(user.id, user.username), username=user.username
    )


@router.get("/me", summary="当前用户")
def me(user_id: str = Depends(get_current_user)) -> dict:
    """前端启动时用它校验本地令牌是否还有效(过期就跳登录页)。"""
    return {"user_id": user_id}
