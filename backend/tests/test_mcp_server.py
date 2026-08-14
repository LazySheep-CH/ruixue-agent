"""MCP server(提供方)的测试。

重点全在几条错了会出事的边界上,而不是"能不能返回结果":

- 默认必须关闭 —— 一个没人用的入口默认开着,是白送的攻击面。
- 无凭证必须 401 —— 子应用不吃 FastAPI 的 Depends,鉴权写在 ASGI 层。
  这条要是漏了,表现是接口一切正常,只是谁都能调,不会有任何报错。
- 白名单必须真的生效 —— 配了只暴露 A,结果 B 也出去了,等于配置形同虚设。
- 配额超限要返回人话而不是抛异常 —— 对方的 agent 才能转述给它的用户。
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from ruixue_app import mcp_server

# MCP streamable-http 要求客户端同时接受 JSON 与 SSE
_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}
_LIST_TOOLS = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


@pytest.fixture(autouse=True)
def _restore_global_state():
    """把这个文件弄脏的全局状态复原。

    为什么必须有(踩过):
    这里的测试要走真实的 app 生命周期(MCP 的会话管理器只在 lifespan 里启动),
    于是用了 `with TestClient(app)` —— 而退出这个 with 会触发优雅停机,
    `runs._shutting_down` 被永久置 True。后果是本文件之后的所有测试
    发起对话都得到 503,而报错信息完全指向别处("为什么 /chat/stream 挂了")。

    单跑 test_runs.py 全绿、和本文件一起跑就红 —— 这类"测试之间互相污染"
    是最难查的,因为失败的地方和肇事的地方毫无关联。

    模块重载同理:reload 会重建 API_KEYS 等模块级状态,不复位就带到下一个文件。
    """
    import ruixue_app.auth
    import ruixue_app.main
    from ruixue_app import runs

    yield
    runs._shutting_down = False
    runs._executor = None
    importlib.reload(ruixue_app.auth)
    importlib.reload(mcp_server)
    importlib.reload(ruixue_app.main)


def _reload_app(monkeypatch, **env):
    """按给定环境变量重建 app —— 挂载发生在模块导入期,必须重新 import。"""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import ruixue_app.auth
    import ruixue_app.main

    importlib.reload(ruixue_app.auth)
    importlib.reload(mcp_server)
    return importlib.reload(ruixue_app.main).app


# ── 默认关闭 ──────────────────────────────────────────────────


def test_disabled_by_default(monkeypatch):
    """没显式开启就完全不存在 —— 零开销、零攻击面。"""
    monkeypatch.delenv("RUIXUE_MCP_SERVER", raising=False)
    assert mcp_server.build_server() is None


@pytest.mark.parametrize("flag", ["0", "false", "no", ""])
def test_falsy_flags_keep_it_off(monkeypatch, flag):
    """只认明确的开启值。'RUIXUE_MCP_SERVER=0' 被当成"开"是经典事故。"""
    monkeypatch.setenv("RUIXUE_MCP_SERVER", flag)
    assert mcp_server.build_server() is None


# ── 鉴权:错了是静默无鉴权,最危险的一类 ────────────────────────


def test_no_credential_is_rejected(monkeypatch):
    """无凭证必须 401。子应用不走 FastAPI 依赖,这条只能靠 ASGI 网关。"""
    app = _reload_app(monkeypatch, RUIXUE_MCP_SERVER="1", RUIXUE_API_KEYS="k1:alice")
    with TestClient(app) as c:
        r = c.post("/mcp/", json=_LIST_TOOLS, headers=_MCP_HEADERS)
    assert r.status_code == 401


def test_wrong_api_key_is_rejected(monkeypatch):
    app = _reload_app(monkeypatch, RUIXUE_MCP_SERVER="1", RUIXUE_API_KEYS="k1:alice")
    with TestClient(app) as c:
        r = c.post("/mcp/", json=_LIST_TOOLS, headers={**_MCP_HEADERS, "X-API-Key": "wrong"})
    assert r.status_code == 401


def test_valid_key_can_list_tools(monkeypatch):
    app = _reload_app(monkeypatch, RUIXUE_MCP_SERVER="1", RUIXUE_API_KEYS="k1:alice")
    with TestClient(app) as c:
        r = c.post("/mcp/", json=_LIST_TOOLS, headers={**_MCP_HEADERS, "X-API-Key": "k1"})
    assert r.status_code == 200
    assert "predict_by_location" in r.text


def test_authenticate_returns_none_on_garbage():
    """认证是安全边界:任何解析失败都必须往严的方向倒,绝不返回默认用户。"""
    assert mcp_server.authenticate({}) is None
    assert mcp_server.authenticate({"authorization": "Bearer 不是token"}) is None
    assert mcp_server.authenticate({"x-api-key": ""}) is None


# ── 白名单 ────────────────────────────────────────────────────


def test_allowlist_actually_limits_exposure(monkeypatch):
    """配了只暴露预测模型,知识库就不该出去 —— 否则配置等于没写。"""
    app = _reload_app(
        monkeypatch,
        RUIXUE_MCP_SERVER="1",
        RUIXUE_API_KEYS="k1:alice",
        RUIXUE_MCP_TOOLS="predict_by_location,predict_degradation",
    )
    with TestClient(app) as c:
        r = c.post("/mcp/", json=_LIST_TOOLS, headers={**_MCP_HEADERS, "X-API-Key": "k1"})
    assert "predict_by_location" in r.text
    assert "search_knowledge" not in r.text, "白名单外的工具泄露了"


def test_unknown_tool_names_disable_the_server(monkeypatch):
    """白名单全打错字时不该"静默暴露零个工具"——那样对方连不上又查不出原因。"""
    monkeypatch.setenv("RUIXUE_MCP_SERVER", "1")
    monkeypatch.setenv("RUIXUE_MCP_TOOLS", "typo_tool_name")
    assert mcp_server.build_server() is None


# ── 真实调用:能列出工具 ≠ 能调通 ──────────────────────────────


def test_end_to_end_tool_call_returns_real_result(monkeypatch):
    """走完整条链路:鉴权 → 路由 → schema 解析 → 执行 → 回包。

    只测 tools/list 会漏掉一大片:参数 schema 生成错、类型转换错、
    返回值序列化错,全都要到真正调用时才暴露。
    这里用 estimate_film_usage —— 纯公式,不连库不连网,CI 里能稳定跑。
    """
    app = _reload_app(
        monkeypatch,
        RUIXUE_MCP_SERVER="1",
        RUIXUE_API_KEYS="k1:alice",
        RUIXUE_MCP_TOOLS="estimate_film_usage",
    )
    with TestClient(app) as c:
        r = c.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "estimate_film_usage",
                    "arguments": {"area_mu": 50, "film_type": "PE", "thickness_mm": 0.01},
                },
            },
            headers={**_MCP_HEADERS, "X-API-Key": "k1"},
        )
    assert r.status_code == 200, r.text
    # 50 亩 × 900 × 0.01 = 450 公斤
    assert "450" in r.text, r.text[:500]


# ── 配额:超限要返回人话,不能抛异常 ──────────────────────────


def test_over_quota_returns_readable_text_not_exception(monkeypatch):
    """抛异常的话,对方的 agent 只看到 "tool error",没法告诉它的用户发生了什么。"""
    monkeypatch.setattr(mcp_server.quota, "consume", lambda uid: (False, 999))
    wrapped = mcp_server._metered(lambda: "真结果", "fake_tool")
    out = wrapped()
    assert "上限" in out and "真结果" not in out


def test_within_quota_passes_through(monkeypatch):
    monkeypatch.setattr(mcp_server.quota, "consume", lambda uid: (True, 1))
    assert mcp_server._metered(lambda: "真结果", "fake_tool")() == "真结果"


def test_caller_identity_is_used_for_metering(monkeypatch):
    """计费必须记在解析出来的身份上,不能记在调用方自己声称的身份上。"""
    seen = []
    monkeypatch.setattr(mcp_server.quota, "consume", lambda uid: (seen.append(uid), (True, 1))[1])
    token = mcp_server._caller.set("alice")
    try:
        mcp_server._metered(lambda: "ok", "t")()
    finally:
        mcp_server._caller.reset(token)
    assert seen == ["alice"]
