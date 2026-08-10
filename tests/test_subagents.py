"""子智能体(多 Agent)测试:验证委派机制,不花钱、不连库。

手法:用 monkeypatch 把 subagents 里的 create_model 换成【假模型】,
这样专家子 agent 用的就是我们能精确控制"说什么"的假模型。
再单独给主 agent 一个假模型,让它"决定派活"。整条链就能在毫秒级跑通。
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

import ruixue_agent.subagents as sub


class ScriptedModel(FakeMessagesListChatModel):
    """照剧本说话的假模型(自带那个没实现 bind_tools,这里补上)。"""

    def bind_tools(self, tools, **kwargs):
        return self


# ── 委派工具本身:名字不对时不崩,给出可选清单 ────────────────
def test_delegate_unknown_expert_returns_hint():
    result = sub.delegate_to_expert.invoke({"expert": "不存在的专家", "task": "随便"})
    assert "没有名为" in result
    assert "文献检索专家" in result  # 提示里带上可选清单,便于模型自我纠正


# ── 委派工具:把子任务交给专家子 agent,返回其最终答案 ─────────
def test_delegate_runs_expert_and_returns_answer(monkeypatch):
    """核心:delegate 里真的建了个专家子 agent、让它跑完、把结论返回。"""
    # 专家的假模型:直接给出结论(不调工具,简化)
    expert_model = ScriptedModel(responses=[AIMessage(content="据文献,PBAT 降解率约 90%。")])
    # 把 subagents 内部用来造专家的 create_model 换掉 -> 专家就用这个假模型
    monkeypatch.setattr(sub, "create_model", lambda name: expert_model)

    result = sub.delegate_to_expert.invoke({"expert": "文献检索专家", "task": "查 PBAT 降解性能"})
    assert result == "据文献,PBAT 降解率约 90%。"


# ── 整条链:主 agent 决定派活 -> 专家执行 -> 主 agent 汇总 ──────
def test_lead_agent_delegates_to_expert(monkeypatch):
    """把 delegate_to_expert 挂到一个主 agent 上,验证"主派活、拿回结论"闭环。"""
    expert_model = ScriptedModel(responses=[AIMessage(content="降解率约 90%。")])
    monkeypatch.setattr(sub, "create_model", lambda name: expert_model)

    # 主 agent 的假模型:先决定调 delegate 工具,拿到结论后再汇总
    lead_model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                id="lead_1",
                tool_calls=[
                    {
                        "name": "delegate_to_expert",
                        "args": {"expert": "文献检索专家", "task": "查 PBAT 降解性能"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="综合专家结论:PBAT 降解率约 90%,性能良好。"),
        ]
    )
    lead = create_agent(lead_model, [sub.delegate_to_expert])
    out = lead.invoke({"messages": [{"role": "user", "content": "PBAT 地膜降解性能如何?"}]})

    messages = out["messages"]
    # 中间应出现一条"专家回给主"的工具消息,内容是专家的结论
    tool_msgs = [m for m in messages if type(m).__name__ == "ToolMessage"]
    assert any("降解率约 90%" in m.content for m in tool_msgs)
    # 主 agent 的最终答案里带上了汇总
    assert "性能良好" in messages[-1].content


# ── 安全:专家的工具集里【不含】delegate,防止无限递归派活 ──────
def test_expert_cannot_delegate_further():
    """专家不能再派活 —— 这是防递归的关键设计,用测试锁死。"""
    for name, spec in sub._EXPERTS.items():
        tool_names = [t.name for t in spec["tools"]]
        assert "delegate_to_expert" not in tool_names, f"专家「{name}」不该持有 delegate 工具"


# ── 子 agent 可观测性 ────────────────────────────────────────────
def test_subagent_tokens_and_inner_tools_are_collected():
    """子 agent 的消息不进父状态,所以它烧的 token 原本【一分钱不算】,
    内部调了什么工具也完全是黑箱。发生委派时成本统计就偏低,而版本对比正是拿它比的。"""
    from unittest.mock import patch

    from langchain_core.messages import AIMessage

    from ruixue_agent.subagents import collect_subagent_runs, delegate_to_expert

    fake = {
        "messages": [
            AIMessage(
                content="",
                id="s1",
                tool_calls=[{"name": "search_knowledge", "args": {}, "id": "t1"}],
                usage_metadata={"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280},
            ),
            AIMessage(
                content="专家的结论",
                id="s2",
                usage_metadata={"input_tokens": 1500, "output_tokens": 200, "total_tokens": 1700},
            ),
        ]
    }

    class FakeAgent:
        def invoke(self, *a, **k):
            return fake

    with patch("ruixue_agent.subagents._build_expert", return_value=FakeAgent()):
        with collect_subagent_runs() as runs:
            out = delegate_to_expert.invoke({"expert": "文献检索专家", "task": "查国标"})

    assert out == "专家的结论", "主 agent 拿到的仍只是结论,接口没变"
    assert len(runs) == 1
    r = runs[0]
    assert r.sub_run_id.startswith("sa-") and r.expert == "文献检索专家"
    assert r.tools == ["search_knowledge"], "专家内部调的工具必须可见"
    assert (r.input_tokens, r.output_tokens) == (2700, 280), "子 agent 的 token 必须被算进来"


def test_trace_total_tokens_includes_subagents():
    """成本指标必须把子 agent 算进去,否则版本成本对比是错的。"""
    from ruixue_agent.eval.trace import Trace
    from ruixue_agent.subagents import SubAgentRun

    tr = Trace(case_id="c", input_tokens=1000, output_tokens=100)
    assert tr.total_tokens == 1100
    tr.subagent_runs = [
        SubAgentRun("sa-1", "文献检索专家", "t", ["search_knowledge"], 2700, 280, 900, True)
    ]
    assert tr.total_tokens == 4080
    assert tr.all_tool_names == ["search_knowledge"]


def test_failed_delegation_is_also_recorded():
    """专家崩了也要留账 —— 否则排查时只看到主 agent 报错,不知道是哪次委派炸的。"""
    from unittest.mock import patch

    import pytest

    from ruixue_agent.subagents import collect_subagent_runs, delegate_to_expert

    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("专家挂了")

    with patch("ruixue_agent.subagents._build_expert", return_value=Boom()):
        with collect_subagent_runs() as runs:
            with pytest.raises(RuntimeError):
                delegate_to_expert.invoke({"expert": "文献检索专家", "task": "x"})
    assert len(runs) == 1 and runs[0].ok is False and runs[0].error == "RuntimeError"
