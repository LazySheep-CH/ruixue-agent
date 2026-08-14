"""认证体系测试:密码哈希 · JWT · 注册登录 · 数据隔离。

密码/JWT 部分纯离线;注册登录需要数据库(users 表),没起 PG 就跳过。
"""

import time

import pytest
from fastapi.testclient import TestClient

from ruixue_app.main import app
from ruixue_app.security.password import hash_password, verify_password
from ruixue_app.security.tokens import TokenError, create_access_token, decode_access_token

client = TestClient(app)


def _username() -> str:
    """每次用不同用户名,避免测试之间互相污染。"""
    return f"u{int(time.time() * 1000) % 10_000_000}"


# ── 密码哈希(离线)────────────────────────────────────────────
def test_password_hash_is_not_plaintext():
    h = hash_password("ruixue123")
    assert h != "ruixue123"
    assert h.startswith("$2")  # bcrypt 格式


def test_password_verify_roundtrip():
    h = hash_password("ruixue123")
    assert verify_password("ruixue123", h)
    assert not verify_password("wrongpass", h)


def test_same_password_hashes_differ():
    """加盐 → 同一密码两次哈希结果不同(防彩虹表)。"""
    assert hash_password("same") != hash_password("same")


def test_long_password_not_truncated():
    """bcrypt 有 72 字节上限;我们做了 sha256 预哈希,超长密码不该被静默截断。"""
    base = "x" * 80
    h = hash_password(base + "AAA")
    assert not verify_password(base + "BBB", h)  # 只有第73字节起不同


def test_corrupted_hash_returns_false_not_raise():
    assert verify_password("any", "not-a-valid-hash") is False


# ── JWT(离线)──────────────────────────────────────────────────
def test_jwt_roundtrip():
    payload = decode_access_token(create_access_token(7, "alice"))
    assert payload["sub"] == "7"
    assert payload["username"] == "alice"


def test_tampered_jwt_rejected():
    token = create_access_token(7, "alice")
    with pytest.raises(TokenError):
        decode_access_token(token[:-3] + "abc")  # 改签名


def test_garbage_jwt_rejected():
    with pytest.raises(TokenError):
        decode_access_token("not.a.token")


# ── 端点:未认证一律 401 ────────────────────────────────────────
def test_protected_endpoints_require_auth():
    assert client.get("/auth/me").status_code == 401
    assert client.post("/chat", json={"thread_id": "t", "message": "hi"}).status_code == 401


def test_bad_bearer_token_rejected():
    r = client.get("/auth/me", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401


# ── 注册 / 登录(要数据库)──────────────────────────────────────
def _db_ok() -> bool:
    return client.get("/health/ready").status_code == 200


needs_db = pytest.mark.skipif(not _db_ok(), reason="PostgreSQL 未就绪")


@needs_db
def test_register_login_and_me():
    name = _username()
    r = client.post("/auth/register", json={"username": name, "password": "ruixue123"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    assert r.json()["username"] == name

    # 重复注册 → 409
    assert (
        client.post("/auth/register", json={"username": name, "password": "ruixue123"}).status_code
        == 409
    )
    # 正确密码 → 200;错误密码 → 401
    assert (
        client.post("/auth/login", json={"username": name, "password": "ruixue123"}).status_code
        == 200
    )
    assert (
        client.post("/auth/login", json={"username": name, "password": "nope12345"}).status_code
        == 401
    )

    # 用令牌查自己
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user_id"].startswith("u")


@needs_db
def test_unknown_user_and_wrong_password_same_message():
    """安全用户名不存在与密码错误必须返回同一句话,否则可枚举用户名。"""
    a = client.post("/auth/login", json={"username": "nosuchuser999", "password": "whatever1"})
    name = _username()
    client.post("/auth/register", json={"username": name, "password": "ruixue123"})
    b = client.post("/auth/login", json={"username": name, "password": "wrongpass1"})
    assert a.status_code == b.status_code == 401
    assert a.json()["detail"] == b.json()["detail"]


@needs_db
def test_two_users_get_different_namespaces():
    """不同用户的 user_id 不同 → thread_id 命名空间隔离(防越权读他人会话)。"""
    ids = []
    for _ in range(2):
        name = _username() + "x"
        r = client.post("/auth/register", json={"username": name, "password": "ruixue123"})
        tok = r.json()["access_token"]
        ids.append(
            client.get("/auth/me", headers={"Authorization": f"Bearer {tok}"}).json()["user_id"]
        )
        time.sleep(0.002)
    assert ids[0] != ids[1]


@needs_db
def test_short_username_or_password_rejected():
    assert (
        client.post("/auth/register", json={"username": "ab", "password": "ruixue123"}).status_code
        == 422
    )
    assert (
        client.post("/auth/register", json={"username": _username(), "password": "123"}).status_code
        == 422
    )
