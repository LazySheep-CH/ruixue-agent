"""人工批准(Human-in-the-Loop)测试:中间件装配 + API 中断/恢复 + 越权防线。

绑定的场景是成本控制而非"防破坏"——本 agent 工具全为只读,
但 delegate_to_expert 会启动子 agent(嵌套 LLM 循环,最贵的单次操作),
让用户确认"要不要花这个钱"是真实需求。默认关闭,按需开启。
"""

import pytest
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

import ruixue_app.auth as auth
import ruixue_app.main as main_mod
from ruixue_agent.agents.builder import REQUIRE_APPROVAL_TOOLS, _build_middleware

_KEY = "hitl-test-key"


# ── 中间件装配 ────────────────────────────────────────────────
def test_approval_middleware_off_by_default():
    """默认关闭 —— 开启会改变 /chat 返回形态,需前端配合,不能默认打开。"""
    names = [type(m).__name__ for m in _build_middleware("deepseek-v4-flash", False)]
    assert "HumanInTheLoopMiddleware" not in names


def test_approval_middleware_inserted_when_enabled():
    names = [type(m).__name__ for m in _build_middleware("deepseek-v4-flash", True)]
    assert "HumanInTheLoopMiddleware" in names
    # 位置:在重试之后、计时之前—— 等人批准的时间不该算进工具耗时
    assert names.index("ModelRetryMiddleware") < names.index("HumanInTheLoopMiddleware")
    assert names.index("HumanInTheLoopMiddleware") < names.index("TimingLoggingMiddleware")


def test_expensive_tool_is_the_one_gated():
    """把关的是启动子 agent 的那个工具(最贵),不是随便挑的只读工具。"""
    assert REQUIRE_APPROVAL_TOOLS == ["delegate_to_expert"]


# ── API 中断 / 恢复 ───────────────────────────────────────────
class _FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def delegate_to_expert(expert: str, task: str) -> str:
    """派活给专家(测试替身)。"""
    return "专家结论:建议 PBAT70"


@pytest.fixture
def client(monkeypatch):
    """装一个开启了人工批准的假 agent,并注入测试 API Key。"""
    monkeypatch.setattr(auth, "API_KEYS", {_KEY: "alice"})
    model = _FakeModel(
        responses=[
            AIMessage(
                content="",
                id="a1",
                tool_calls=[
                    {
                        "name": "delegate_to_expert",
                        "args": {"expert": "配方优化专家", "task": "选配方"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="综合专家意见:推荐 PBAT70/PLA30"),
        ]
    )
    agent = create_agent(
        model,
        [delegate_to_expert],
        checkpointer=InMemorySaver(),
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"delegate_to_expert": True})],
    )
    monkeypatch.setattr(main_mod, "_agent", agent)
    return TestClient(main_mod.app)


_H = {"X-API-Key": _KEY}


def test_chat_returns_pending_instead_of_answer(client):
    r = client.post("/chat", headers=_H, json={"thread_id": "t1", "message": "帮我选配方"}).json()
    assert r["answer"] == ""  # 还没答,等批准
    assert r["pending"] and r["pending"][0]["tool"] == "delegate_to_expert"
    assert r["pending"][0]["args"]["expert"] == "配方优化专家"  # 让用户看清要执行什么


def test_resume_approve_completes_the_answer(client):
    client.post("/chat", headers=_H, json={"thread_id": "t1", "message": "帮我选配方"})
    r = client.post("/chat/resume", headers=_H, json={"thread_id": "t1", "approve": True}).json()
    assert "PBAT70" in r["answer"]
    assert r["pending"] is None


def test_resume_requires_auth(client):
    """批准端点同样要认证 —— 否则任何人都能替别人批准。"""
    r = client.post("/chat/resume", json={"thread_id": "t1", "approve": True})
    assert r.status_code == 401


def test_resume_is_namespaced_per_user(client, monkeypatch):
    """越权防线:B 用户的批准动不了A 用户会话里的待确认操作。

    验证方式是"A 的状态没被改变",而不是"B 拿没拿到答案"——
    后者会被测试替身干扰(共享的假模型剧本会往下走,给 B 返回下一句台词,
    那与 A 的会话无关,曾误判为越权)。真正的安全属性是状态隔离。
    """
    client.post("/chat", headers=_H, json={"thread_id": "t1", "message": "帮我选配方"})
    monkeypatch.setattr(auth, "API_KEYS", {_KEY: "alice", "bob-key": "bob"})

    # bob 试图批准 alice 会话里的操作(用同一个 thread_id "t1")
    client.post(
        "/chat/resume", headers={"X-API-Key": "bob-key"}, json={"thread_id": "t1", "approve": True}
    )

    # alice 的线程状态必须仍停在"待批准" —— 说明 bob 根本碰不到它
    state = main_mod._agent.get_state({"configurable": {"thread_id": "alice:t1"}})
    assert state.interrupts, "alice 的待批准操作被别的用户消费了 —— 越权!"
