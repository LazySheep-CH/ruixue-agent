"""CORS 测试:前端独立部署后要能跨域调后端,但只放行白名单来源。"""

from fastapi.testclient import TestClient

from ruixue_app.main import ALLOWED_ORIGINS, app

client = TestClient(app)


def test_allowed_origin_gets_cors_header():
    """白名单来源(前端开发端口)应拿到 CORS 放行头。"""
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_preflight_allows_api_key_header():
    """预检请求要放行自定义头 X-API-Key,否则前端带不了认证。"""
    resp = client.options(
        "/chat/stream",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-API-Key",
        },
    )
    assert resp.status_code == 200
    assert "x-api-key" in resp.headers.get("access-control-allow-headers", "").lower()


def test_unknown_origin_not_allowed():
    """【安全】非白名单来源不给放行头(不能用 * 通配)。"""
    resp = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example.com"
    assert resp.headers.get("access-control-allow-origin") != "*"


def test_default_origins_cover_frontend_dev_ports():
    assert "http://localhost:3000" in ALLOWED_ORIGINS
