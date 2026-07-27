"""前端页面托管测试:同源托管聊天页,且不影响任何 API 路由。"""

from fastapi.testclient import TestClient

from ruixue_app.main import app

client = TestClient(app)


def test_index_serves_chat_page():
    """根路径返回聊天页(同源托管 → 前端调 /chat/stream 免 CORS)。"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "瑞雪地膜智能助手" in resp.text
    assert "/chat/stream" in resp.text  # 页面确实接后端流式端点


def test_static_mount_does_not_shadow_api():
    """挂静态不能抢占 API 路由。"""
    assert client.get("/health").status_code == 200
    # 认证仍然生效(不会因为挂了静态就绕过)
    assert client.post("/chat", json={"thread_id": "t", "message": "hi"}).status_code == 401


def test_page_sends_api_key_header():
    """页面必须带 X-API-Key 调后端(否则会被 401)。"""
    assert "X-API-Key" in client.get("/").text
