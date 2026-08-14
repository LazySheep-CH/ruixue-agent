"""技能(skills)测试:加载 / 场景匹配 / 首轮注入。

技能 = 已验证的作业流程(SOP),回答"该怎么做";工具回答"能做什么"。
做成 Markdown 是为了让领域专家能直接改流程而不必动代码。
"""

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
    """同一会话不必反复重申规程 —— 省 token。

    ⚠ 这里必须把第一轮【真实产出的消息】接着喂给第二轮,不能手搓一份历史。
    手搓的历史里没有第一轮注入的那条 SystemMessage,判据("这条规程注入过吗")
    自然认为没注入过 —— 测出来的就不是真实行为。
    2026-08-08 改判据时这个测试挂了,一查才发现它原本是靠"首轮"这个旧启发式
    才碰巧通过的,历史本身是假的。
    """
    first = _run([{"role": "user", "content": "新疆尉犁选什么地膜配方?"}])
    assert [m for m in first if isinstance(m, SystemMessage)], "第一轮应当注入"

    # 带着第一轮的全部真实消息(含注入的规程)继续问同类问题
    second = _run([*first, {"role": "user", "content": "那玉米呢?也要选配方"}])
    injected = [m for m in second if isinstance(m, SystemMessage)]
    assert len(injected) == 1, "同一条规程不该被注入第二次"


# ── 注入时机:每条规程只注入一次,但不限定首轮 ──────────────────
#
# 2026-08-08 上课时发现的真 bug:早先判据是"只在首轮注入",于是
#     用户:你好          ← 首轮但没命中关键词 → 不注入
#     用户:帮我选个配方   ← 命中了却不是首轮 → 【永远不会注入】
# 只要开场寒暄一句,整个技能系统就失效。以下四个测试把新判据钉死。


def _inject(messages):
    """跑一次中间件,返回它想追加的文本(没注入则返回 None)。"""
    out = SkillInjectionMiddleware().before_model({"messages": messages}, None)
    return out["messages"][0].content if out else None


def test_injects_on_first_turn_when_matched():
    assert "作业规程" in _inject([HumanMessage("帮我选个配方")])


def test_injects_after_greeting_not_only_first_turn():
    """本次修复的核心:寒暄开场之后再问,照样要注入。"""
    text = _inject(
        [
            HumanMessage("你好"),
            AIMessage("你好,我是瑞雪地膜助手"),
            HumanMessage("帮我选个配方"),
        ]
    )
    assert text and "地膜选型" in text


def test_does_not_reinject_the_same_skill():
    """同一条规程已经在上下文里了,就不再重复塞 —— 省 token 的本意在这里。"""
    first = _inject([HumanMessage("帮我选个配方")])
    again = _inject(
        [
            HumanMessage("帮我选个配方"),
            SystemMessage(first),
            AIMessage("建议 PBAT70/PLA30"),
            HumanMessage("那换个地方呢,配方要变吗"),
        ]
    )
    assert again is None


def test_injects_a_different_skill_later():
    """换了话题命中另一条规程,该注入还得注入(只跳过已注入的那条)。"""
    first = _inject([HumanMessage("帮我选个配方")])
    later = _inject(
        [
            HumanMessage("帮我选个配方"),
            SystemMessage(first),
            AIMessage("建议 PBAT70/PLA30"),
            HumanMessage("那这个膜的降解率大概多少"),
        ]
    )
    assert later and "性能预测解读" in later and "地膜选型" not in later
