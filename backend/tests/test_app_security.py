"""app 层(FastAPI 服务)安全测试:认证 + 输入上限。

不连数据库、不调 LLM。做得到的原因,正是 P1 的 lifespan 改动:
建 agent 挪进了 lifespan,所以【导入 main、不进入 lifespan】就不会连库。
我们用【不带 with 的 TestClient】发请求 —— lifespan 不触发,_agent 保持 None。
而这些请求都在【碰到 agent 之前】就被拦下(认证失败/输入超长),
所以 _agent 是不是 None 根本不影响,测试快、稳、可进 CI。
"""

import pytest
from fastapi.testclient import TestClient

import ruixue_app.auth as auth
from ruixue_app.main import app

client = TestClient(app)  # 不用 with -> 不触发 lifespan -> 不建 agent、不连库

# API Key 已改为从环境变量 RUIXUE_API_KEYS 读取(不再硬编码,安全改进)。
# 测试里注入一个临时 key,不依赖真实环境配置。
_TEST_KEY = "test-key-alice"
_GOOD_KEY = {"X-API-Key": _TEST_KEY}


@pytest.fixture(autouse=True)
def _inject_api_key(monkeypatch):
    """给每个测试注入一个可用的 API Key(测完自动还原)。"""
    monkeypatch.setattr(auth, "API_KEYS", {_TEST_KEY: "alice"})


# ── 认证:没带对钥匙,进不来(P0 第①道门)──────────────────────
def test_missing_api_key_returns_401():
    """没带 X-API-Key -> 401,连业务都不会执行。"""
    resp = client.post("/chat", json={"thread_id": "t1", "message": "你好"})
    assert resp.status_code == 401


def test_wrong_api_key_returns_401():
    """钥匙不对 -> 401。注意:HTTP 请求头只能放 ASCII,不能放中文,所以用英文假钥匙。"""
    resp = client.post(
        "/chat",
        headers={"X-API-Key": "wrong-key-123"},
        json={"thread_id": "t1", "message": "你好"},
    )
    assert resp.status_code == 401


# ── 输入上限:超长直接拒(P0 第④道门,防烧 token)──────────────
# 下面几个用【正确的钥匙】(见上方 fixture),让认证先通过,单独验证"输入校验"这一关。


def test_message_too_long_returns_422():
    """message 超过 2000 字 -> 422(Pydantic 的 Field(max_length) 拦下)。"""
    resp = client.post(
        "/chat",
        headers=_GOOD_KEY,
        json={"thread_id": "t1", "message": "字" * 2001},  # 2001 字,超一个
    )
    assert resp.status_code == 422


def test_thread_id_too_long_returns_422():
    """thread_id 超过 64 字 -> 422。"""
    resp = client.post(
        "/chat",
        headers=_GOOD_KEY,
        json={"thread_id": "x" * 65, "message": "你好"},
    )
    assert resp.status_code == 422


def test_valid_input_reaches_agent_and_returns_200(monkeypatch):
    """反面对照:合法请求应一路通到 agent 并拿到答案。

    我们不想在单测里真连库/真调 LLM,于是用【替身】:monkeypatch 把 main 里的
    _agent 临时换成一个假 agent(只会返回一句固定答案)。monkeypatch 是 pytest
    的夹具,它做的替换【只在这条测试内生效】,测完自动还原,不污染别的测试。

    这条测试证明的是【整条链路通了】:认证放行 -> 输入校验放行 -> 命名空间拼接 ->
    调 agent -> 取最后一条消息的 content 返回。任何一环断了,这里就不是 200。
    """
    from langchain_core.messages import AIMessage

    class FakeAgent:
        def invoke(self, state, config):
            # 断言命名空间隔离真的生效了:thread_id 被拼上了 user_id 前缀
            assert config["configurable"]["thread_id"] == "alice:t1"
            return {"messages": [AIMessage(content="这是替身返回的测试答案")]}

    monkeypatch.setattr("ruixue_app.main._agent", FakeAgent())

    resp = client.post(
        "/chat",
        headers=_GOOD_KEY,  # demo-key-alice -> 用户 alice
        json={"thread_id": "t1", "message": "你好"},
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "这是替身返回的测试答案"
