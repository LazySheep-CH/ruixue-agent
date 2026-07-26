"""可观测性测试:request_id 追踪(日志盖章 + 响应头 + 沿用调用方编号)。"""

import logging

from fastapi.testclient import TestClient

from ruixue_app.main import app
from ruixue_app.observability import RequestIdLogFilter, request_id_var

client = TestClient(app)


# ── 单元:日志过滤器把当前请求编号盖到日志记录上 ──────────────
def test_log_filter_stamps_request_id():
    """把"餐盘"设成某个值,过滤器就该把它盖到日志记录的 request_id 字段上。"""
    token = request_id_var.set("test-rid-123")
    try:
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hi", args=(), exc_info=None,
        )
        RequestIdLogFilter().filter(record)
        assert record.request_id == "test-rid-123"
    finally:
        request_id_var.reset(token)


# ── 集成:响应头一定带 X-Request-ID(没传就自动生成)──────────
def test_response_has_generated_request_id():
    resp = client.get("/health")
    rid = resp.headers.get("X-Request-ID")
    assert rid  # 非空
    assert len(rid) == 12  # 我们生成的是 uuid4().hex[:12]


# ── 集成:调用方自带编号时,原样沿用(跨服务追踪)──────────────
def test_request_id_is_echoed_when_provided():
    resp = client.get("/health", headers={"X-Request-ID": "trace-xyz"})
    assert resp.headers.get("X-Request-ID") == "trace-xyz"


# ── 集成:两个请求各自独立编号,不串 ──────────────────────────
def test_two_requests_get_different_ids():
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]
