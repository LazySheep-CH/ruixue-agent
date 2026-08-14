"""长期记忆的测试。

重点不在"能存能取",而在几条错了会出事的边界:
用户隔离(错了是数据泄露)、幂等(错了会越存越多)、
软删不复活(错了用户会觉得"我删了它还记得")、失败不拖垮主流程。
"""

from __future__ import annotations

import json
import os

import pytest

from ruixue_agent.memory.extract import _strip_fence, extract_facts

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")


def _deps_up() -> bool:
    try:
        from sqlalchemy import text

        from ruixue_agent.persistence.engine import get_engine

        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# ── 抽取:纯函数部分,不需要依赖 ────────────────────────────────


def test_strip_fence_handles_json_code_block():
    """模型爱把 JSON 包在 ```json 里。不剥掉解析就失败,而失败是静默的
    (被 except 吞掉),表现为"记忆功能好像没生效",极难查。"""
    assert _strip_fence('```json\n{"facts": []}\n```') == '{"facts": []}'
    assert _strip_fence('{"facts": []}') == '{"facts": []}'


@pytest.mark.parametrize(
    "raw",
    [
        '{"facts": []}',  # 纯 JSON
        '```json\n{"facts": []}\n```',  # 带语言标记的代码块
        '```\n{"facts": []}\n```',  # 不带语言标记
        '好的,抽取结果如下:\n```json\n{"facts": []}\n```',  # 前面有客套话 ← 旧写法在这挂
        '```json\n{"facts": []}\n```\n以上就是抽取结果。',  # 后面有解释
        '结果:\n```json\n{"facts": []}\n```\n如需调整请告知。',  # 前后都有
        '抽取结果:{"facts": []}',  # 裸 JSON 但有前缀
    ],
)
def test_strip_fence_survives_model_pleasantries(raw):
    """旧写法是 `if s.startswith("```")`,只认整段以代码块开头。

    模型很常见地先客气一句再给代码块,这时判断不成立 → 整段送进 json.loads
    → 报错 → 被 except 吞掉 → 这轮记忆静默丢失。2026-08-08 实测七种形态里
    恰好漏掉"前面有客套话"这一种。
    """
    assert json.loads(_strip_fence(raw)) == {"facts": []}


def test_extract_returns_empty_on_model_failure(monkeypatch):
    """抽取失败必须返回空,而不是抛异常 —— 记忆是锦上添花,
    绝不能因为它让一次已经成功的回答看起来像失败。"""
    import ruixue_agent.memory.extract as ex

    def boom(*a, **k):
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(ex, "create_model", boom)
    assert extract_facts("我在尉犁有50亩地", "好的") == []


def test_extract_empty_question_short_circuits():
    assert extract_facts("", "答案") == []


# ── 存取:需要 PG + Milvus ──────────────────────────────────────

pytestmark_db = pytest.mark.skipif(not _deps_up(), reason="需要 PostgreSQL")


@pytestmark_db
def test_user_isolation_is_enforced_in_pg_too():
    """召回必须双重校验归属:Milvus 侧过滤 + PG 侧再查一次。

    只靠一层的话,任何一层的过滤写错都会召回别人的记忆——
    这不是效果问题,是数据泄露。
    """
    from ruixue_agent.memory import recall, remember
    from ruixue_agent.memory.store import _memory_id

    remember("alice-iso", [("plot", "alice 在尉犁有 50 亩棉花地")])
    # 直接查 PG:即便伪造了 memory_id,user_id 不匹配也拿不到
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from ruixue_agent.persistence.engine import get_engine
    from ruixue_agent.persistence.models import MemoryRow

    mid = _memory_id("alice-iso", "alice 在尉犁有 50 亩棉花地")
    with Session(get_engine()) as s:
        stolen = s.scalar(
            select(MemoryRow).where(MemoryRow.memory_id == mid, MemoryRow.user_id == "bob-iso")
        )
    assert stolen is None, "换个 user_id 就不该查得到"
    assert recall("bob-iso", "我的地在哪") == [], "bob 不该召回 alice 的记忆"


@pytestmark_db
def test_remember_is_idempotent():
    """同一个用户重复说同一件事只存一条 —— 否则用户每次提到地块,
    记忆库就多一条重复的,召回时全是同义句,挤占上下文。"""
    from ruixue_agent.memory import remember

    fact = [("plot", "idem 用户在张掖有 30 亩地")]
    remember("idem-user", fact)
    second = remember("idem-user", fact)
    assert second == 0, "第二次不该新增"


@pytestmark_db
def test_deleted_memory_does_not_come_back():
    """软删 + 内容寻址 = 删过的不会复活。

    如果用硬删:下次用户又说同样的话,会被重新抽出来存回去,
    用户会觉得"我明明删了它还记得" —— 这是隐私功能失效。
    """
    from ruixue_agent.memory import delete_memory, list_memories, remember
    from ruixue_agent.memory.store import _memory_id

    text = "del 用户偏好保墒优先"
    remember("del-user", [("preference", text)])
    mid = _memory_id("del-user", text)
    assert delete_memory("del-user", mid) is True
    assert all(m.memory_id != mid for m in list_memories("del-user"))
    # 再次"抽到"同样的事实
    remember("del-user", [("preference", text)])
    assert all(m.memory_id != mid for m in list_memories("del-user")), "删过的不该复活"


@pytestmark_db
def test_cannot_delete_others_memory():
    from ruixue_agent.memory import delete_memory, remember
    from ruixue_agent.memory.store import _memory_id

    remember("owner-u", [("plot", "owner 的地块信息")])
    mid = _memory_id("owner-u", "owner 的地块信息")
    assert delete_memory("attacker-u", mid) is False, "不能删别人的记忆"


# ── 注入时机与身份隔离(中间件层)──────────────────────────────
#
# 这几条不碰数据库,只测 MemoryRecallMiddleware 的判断逻辑。


class _Rt:
    """最小 runtime:只带 before_model 真正会读的 config.configurable.thread_id。

    注意:这个假 runtime 骗过我们一次,别再单独信它。

    LangGraph 的真 `Runtime` 没有 `.config`(官方文档:"Runtime does not include
    config",要用 `langgraph.config.get_config()`)。这个假类特意带了 `.config`,
    于是下面几条测试全绿,而真实运行里 `_user_id_from` 恒返回空 ——
    长期记忆从上线起一次都没注入过(2026-08-12 实测发现)。

    这些测试仍然有价值:它们验证的是注入时机与身份隔离的判断逻辑,那部分是对的。
    但它们证明不了"接到真框架上能跑"。后者由
    `tests/test_memory_injection_integration.py` 负责 —— 走真 agent、真 runtime,
    只把模型换成假的(不花钱、进 CI)。

    规律:假的输入只能验证你的 if-else,验证不了你对框架的假设。
    """

    def __init__(self, thread_id):
        self.config = {"configurable": {"thread_id": thread_id}}


def _recall_mw(messages, thread_id, fake_rows):
    """跑一次中间件,返回它想追加的文本(没注入则 None)。recall 用假的,不连库。"""
    import ruixue_agent.memory as mem
    from ruixue_agent.agents.middlewares import MemoryRecallMiddleware

    orig = mem.recall
    mem.recall = lambda uid, q: fake_rows
    try:
        out = MemoryRecallMiddleware().before_model({"messages": messages}, _Rt(thread_id))
    finally:
        mem.recall = orig
    return out["messages"][0].content if out else None


class _Row:
    def __init__(self, kind, text):
        self.kind, self.text = kind, text


_ROWS = [_Row("偏好", "在新疆尉犁有 200 亩棉花地")]


def test_memory_injected_after_greeting_not_only_first_turn():
    """记忆最该发挥作用的时刻(用户说"上次那块地"),几乎不可能是第一句话。

    2026-08-08:原判据是"只在首轮",寒暄开场后就永远等不到注入了 ——
    和技能注入是同一个 bug。
    """
    from langchain_core.messages import AIMessage, HumanMessage

    text = _recall_mw(
        [
            HumanMessage("你好"),
            AIMessage("你好,我是瑞雪助手"),
            HumanMessage("还是上次那块地,再帮我看看"),
        ],
        "alice:t1",
        _ROWS,
    )
    assert text and "尉犁" in text


def test_memory_not_injected_twice_in_one_session():
    """已经在上下文里了就不重复塞 —— 省 token 的本意在这里。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    first = _recall_mw([HumanMessage("上次那块地")], "alice:t1", _ROWS)
    again = _recall_mw(
        [
            HumanMessage("上次那块地"),
            SystemMessage(first),
            AIMessage("好的"),
            HumanMessage("那配方呢"),
        ],
        "alice:t1",
        _ROWS,
    )
    assert again is None


def test_no_user_id_means_no_memory_at_all():
    """拿不到身份时一条都不给。

    若图省事写成 user_id or "default",所有匿名请求就共用一份记忆 ——
    张三上午说的话,李四下午就看见了。宁可不给,不可给错人。
    """
    from langchain_core.messages import HumanMessage

    # thread_id 里没有 "用户:" 前缀 → _user_id_from 取不到身份
    assert _recall_mw([HumanMessage("上次那块地")], "no-user-prefix", _ROWS) is None
