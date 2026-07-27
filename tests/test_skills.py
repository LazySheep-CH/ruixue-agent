"""技能(skills)测试:加载 / 场景匹配 / 首轮注入。

技能 = 已验证的作业流程(SOP),回答"该怎么做";工具回答"能做什么"。
做成 Markdown 是为了让领域专家能直接改流程而不必动代码。
"""

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage

from ruixue_agent.agents.middlewares import SkillInjectionMiddleware
from ruixue_agent.skills import load_skills, loader, render_skills, select_skills


# ── 加载 ──────────────────────────────────────────────────────
def test_skills_are_loaded_from_markdown():
    names = [s.name for s in load_skills()]
    assert "地膜选型" in names and "性能预测解读" in names


def test_skill_has_triggers_and_body():
    skill = next(s for s in load_skills() if s.name == "地膜选型")
    assert skill.triggers and "配方" in skill.triggers
    assert "生育期" in skill.body  # 正文是真的 SOP,不是空壳


def test_missing_dir_is_safe(monkeypatch, tmp_path):
    """技能目录不存在时返回空,不报错 —— 缺技能不该让 agent 起不来。"""
    monkeypatch.setattr(loader, "SKILLS_DIR", tmp_path / "nope")
    loader.load_skills.cache_clear()
    assert loader.load_skills() == ()
    loader.load_skills.cache_clear()


def test_file_without_triggers_is_skipped(monkeypatch, tmp_path):
    (tmp_path / "bad.md").write_text("# 没有触发词\n正文", encoding="utf-8")
    monkeypatch.setattr(loader, "SKILLS_DIR", tmp_path)
    loader.load_skills.cache_clear()
    assert loader.load_skills() == ()
    loader.load_skills.cache_clear()


# ── 场景匹配 ──────────────────────────────────────────────────
def test_matches_selection_question():
    hit = select_skills("新疆尉犁选什么地膜配方?")
    assert hit and hit[0].name == "地膜选型"


def test_matches_prediction_question():
    assert any(s.name == "性能预测解读" for s in select_skills("这个膜能降解多少?"))


def test_unrelated_question_injects_nothing():
    """无关问题不注入 —— 技能是按需加载,不能无脑塞满上下文。"""
    assert render_skills("今天天气怎么样") == ""


def test_limit_caps_injected_skills():
    """命中再多也要设上限,否则上下文会被规程塞爆。"""
    assert len(select_skills("配方 降解率 选型 预测", limit=1)) == 1


# ── 中间件:只在首轮注入 ──────────────────────────────────────
class _FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _run(messages):
    agent = create_agent(
        _FakeModel(responses=[AIMessage(content="回答")]),
        [],
        middleware=[SkillInjectionMiddleware()],
    )
    return agent.invoke({"messages": messages})["messages"]


def test_injects_sop_on_first_turn():
    msgs = _run([{"role": "user", "content": "新疆尉犁选什么地膜配方?"}])
    injected = [m for m in msgs if isinstance(m, SystemMessage)]
    assert injected and "作业规程" in injected[0].content


def test_not_injected_again_on_later_turns():
    """同一会话不必反复重申规程 —— 省 token。"""
    msgs = _run(
        [
            {"role": "user", "content": "新疆尉犁选什么地膜配方?"},
            {"role": "assistant", "content": "上一轮的回答"},
            {"role": "user", "content": "那玉米呢?也要选配方"},
        ]
    )
    assert not [m for m in msgs if isinstance(m, SystemMessage)]
