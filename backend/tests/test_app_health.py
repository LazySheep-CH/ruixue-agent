"""健康检查测试:存活探针 + 就绪探针(含"数据库挂了"的失败路径)。

同样用不带 with 的 TestClient(不触发 lifespan、不建 agent)。
"""

from fastapi.testclient import TestClient

from ruixue_app.main import app

client = TestClient(app)


# ── 存活探针:永远 200,不碰任何外部依赖 ──────────────────────
def test_health_liveness_always_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── 就绪探针:数据库连不上 -> 503(用替身模拟"挂了")──────────
def test_readiness_returns_503_when_db_down(monkeypatch):
    """核心:数据库挂了,就绪探针必须回 503(而不是 500,也不是假装 ready)。

    我们不真去关数据库,而是用 monkeypatch 把 get_engine 换成一个"一调用就抛异常"
    的替身 —— 等价于"数据库连不上"。这样失败路径能确定性地测到,不靠运气。
    注意:被替换的是它被查找的地方ruixue_agent.persistence.engine.get_engine
    (就绪探针里就是从那儿 import 的)。
    """

    def boom():
        raise ConnectionError("模拟:数据库连不上")

    monkeypatch.setattr("ruixue_agent.persistence.engine.get_engine", boom)

    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready"}


# ── 就绪探针:数据库正常 -> 200 ready(没起 PG 就跳过)──────────
def test_readiness_returns_200_when_db_up(pg_available):
    if not pg_available:
        import pytest

        pytest.skip("PostgreSQL 没起(docker compose up -d postgres)")

    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}
