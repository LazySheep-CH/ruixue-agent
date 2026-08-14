"""用户仓储:注册、按用户名查、校验登录。

放在 app/security 而非 persistence/repository:那边是 RAG 文档的仓储,职责不同。
这里只做"用户"这一个聚合,依赖 persistence 的 engine(复用连接池)。
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.models import UserRow
from ruixue_app.security.password import hash_password, verify_password

USERNAME_MIN = 3
PASSWORD_MIN = 6


class UserError(Exception):
    """业务性错误(用户名已存在等),消息可直接给用户看。"""


def _session() -> Session:
    return Session(bind=get_engine())


def create_user(username: str, password: str) -> UserRow:
    """注册。用户名唯一;密码只存哈希。"""
    username = username.strip()
    if len(username) < USERNAME_MIN:
        raise UserError(f"用户名至少 {USERNAME_MIN} 个字符")
    if len(password) < PASSWORD_MIN:
        raise UserError(f"密码至少 {PASSWORD_MIN} 个字符")

    with _session() as s:
        if s.query(UserRow).filter_by(username=username).first():
            raise UserError("该用户名已被注册")
        user = UserRow(username=username, password_hash=hash_password(password))
        s.add(user)
        s.commit()
        s.refresh(user)
        s.expunge(user)  # 脱离 session 后仍可读属性
        return user


def authenticate(username: str, password: str) -> UserRow:
    """登录校验。

    安全用户名不存在与密码错误返回同一句提示—— 否则攻击者能靠不同的
    错误信息枚举出哪些用户名是存在的(用户枚举漏洞)。
    """
    with _session() as s:
        user = s.query(UserRow).filter_by(username=username.strip()).first()
        if user is None or not verify_password(password, user.password_hash):
            raise UserError("用户名或密码错误")
        if not user.is_active:
            raise UserError("该账号已被停用")
        s.expunge(user)
        return user


def get_user(user_id: int) -> UserRow | None:
    with _session() as s:
        user = s.get(UserRow, user_id)
        if user is not None:
            s.expunge(user)
        return user
