"""长期记忆注入的【集成】测试:走真 agent、真 runtime,只把模型换成假的。

为什么必须单独有这一层:
`tests/test_memory.py` 里已经测过 `MemoryRecallMiddleware.before_model` 的逻辑,
而且测得很细(注入时机、去重、身份隔离)。但那些测试用的是手搓的假 runtime:

    class _Rt:
        def __init__(self, thread_id):
            self.config = {"configurable": {"thread_id": thread_id}}

它测的是"如果 runtime 长这样,逻辑对不对"。

而 LangGraph 的真 `Runtime` 没有 `config` 属性(官方文档:"Runtime does not
include config",要用 `langgraph.config.get_config()`)。于是:

    单元测试:全绿  ✅   ——  因为假 runtime 带 .config
    真实运行:全废  ❌   ——  真 runtime 没有 .config,user_id 恒为空,永不注入

2026-08-12 实测:长期记忆从上线起一次都没注入过。
失败方式是"什么都不做"——没身份就不注入本身是正确行为(宁可不给,不可给错人),
所以没有报错、没有异常日志,只是功能静静地不生效。

结论:凡是依赖框架对象形状的逻辑,必须有一条走真框架的测试。
假的输入只能验证你的 if-else,验证不了你对框架的假设。

这条测试用 FakeMessagesListChatModel:不花钱、确定性、能进 CI,
但 agent 的装配、中间件链、runtime 全是真的 —— 正好卡在该卡的地方。
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from ruixue_agent.agents.middlewares import MEMORY_HEADER


class _Fake(FakeMessagesListChatModel):
    """永远直接给最终答案的假模型 —— 我们要看的是【进模型前】的上下文,不是它答什么。"""

    def bind_tools(self, tools, **kw):
        return self


def _agent_with_fake_model(monkeypatch):
    """用真装配函数建 agent,只把模型换成假的。

    必须走 create_ruixue_agent —— 另拼一个 agent 就绕开了真正的中间件链,
    这条测试也就失去了意义。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    import ruixue_agent.agents.builder as builder

    fake = _Fake(responses=[AIMessage(content="好的。", id=f"a-{uuid.uuid4().hex[:6]}")])
    monkeypatch.setattr(builder, "create_model", lambda *a, **k: fake)
    return builder.create_ruixue_agent(checkpointer=InMemorySaver())


def _run(agent, thread_id: str, text: str):
    return agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )


@pytest.fixture
def _fake_recall(monkeypatch):
    """假的召回:不连 Milvus,固定返回一条记忆。测的是注入链路,不是召回质量。"""

    class _Row:
        kind, text = "plot", "用户在新疆尉犁县有 50 亩棉花地"

    seen: list[str] = []

    import ruixue_agent.memory as mem

    def fake(uid, q):
        seen.append(uid)
        return [_Row()]

    monkeypatch.setattr(mem, "recall", fake)
    return seen


def test_memory_is_actually_injected_through_the_real_runtime(monkeypatch, _fake_recall):
    """核心断言:记忆真的进了上下文。

    这一条要是挂了,说明 user_id 又取不到了 —— 长期记忆整个功能是废的。
    """
    agent = _agent_with_fake_model(monkeypatch)
    state = _run(agent, "alice:t1", "帮我算一下要买多少地膜。")

    injected = [m for m in state["messages"] if MEMORY_HEADER in str(getattr(m, "content", ""))]
    assert injected, "记忆没有注入 —— user_id 大概率又取不到了(真 Runtime 没有 .config)"
    assert "尉犁" in str(injected[0].content)


def test_user_id_comes_from_thread_prefix(monkeypatch, _fake_recall):
    """召回必须按【解析出来的 user_id】查,查错人就是数据泄露。"""
    agent = _agent_with_fake_model(monkeypatch)
    _run(agent, "alice:t1", "帮我算一下要买多少地膜。")
    assert _fake_recall == ["alice"], f"recall 收到的 user_id 不对:{_fake_recall}"


def test_memory_still_injects_when_a_skill_also_injects(monkeypatch, _fake_recall):
    """技能注入不能把记忆注入挤掉 —— 2026-08-12 实测踩过。

    中间件链是 技能 → 记忆。技能命中时会往消息尾部追加一条 SystemMessage,
    于是记忆中间件原来那句 `messages[-1] must be HumanMessage` 不再成立,
    直接 return None:

        "帮我算一下要买多少地膜"     不触发技能 → 记忆注入 ✅
        "帮我在赤峰选个合适的配方"   触发技能   → 记忆被挡 ❌

    最讽刺的是最需要记忆的问题(选型/配方/推荐)恰恰最容易触发技能,
    这个 bug 精准地打掉了记忆最有价值的那部分场景。

    这里用一句同时命中技能触发词("配方")的提问,断言两者都在。
    """
    # SKILL_HEADER 是带 {name} 占位符的【模板】,不是字面量 —— 用它自己的解析函数。
    from ruixue_agent.skills.loader import injected_names

    agent = _agent_with_fake_model(monkeypatch)
    state = _run(agent, "alice:t1", "帮我在赤峰选个合适的降解膜配方。")

    contents = [str(getattr(m, "content", "")) for m in state["messages"]]
    assert any(injected_names(c) for c in contents), "技能没注入,这道题就测不到冲突"
    assert any(MEMORY_HEADER in c for c in contents), (
        "技能注入后记忆被挤掉了 —— 中间件之间又靠'谁在最后'这种隐式约定耦合了"
    )


def test_neither_is_injected_twice(monkeypatch, _fake_recall):
    """改成"往回找最后一条 HumanMessage"之后,同一轮里 before_model 可能被调多次。

    去重判据必须扫【全部】消息;只扫 messages[:-1] 会漏掉刚注入的那条,
    于是同一段内容塞两遍 —— 白花 token,还稀释注意力。
    """
    from ruixue_agent.skills.loader import injected_names

    agent = _agent_with_fake_model(monkeypatch)
    state = _run(agent, "alice:t1", "帮我在赤峰选个合适的降解膜配方。")

    contents = [str(getattr(m, "content", "")) for m in state["messages"]]
    assert sum(MEMORY_HEADER in c for c in contents) == 1
    assert sum(bool(injected_names(c)) for c in contents) == 1


def test_no_user_prefix_means_no_injection(monkeypatch, _fake_recall):
    """thread_id 里没有 "用户:" 前缀时一条都不给 —— 宁可不给,不可给错人。"""
    agent = _agent_with_fake_model(monkeypatch)
    state = _run(agent, "no-prefix-thread", "帮我算一下要买多少地膜。")

    assert not [m for m in state["messages"] if MEMORY_HEADER in str(getattr(m, "content", ""))]
    assert _fake_recall == [], "没有身份却仍然去查了记忆"
