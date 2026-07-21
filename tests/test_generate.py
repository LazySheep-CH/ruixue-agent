"""生成的测试。LLM 不真调(慢、要联网),测的是【编排 + grounding 的骨架】。"""

from __future__ import annotations

from ruixue_agent.rag.generate import Generator, _format_context
from ruixue_agent.rag.retriever import Hit


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.last_call = {}

    def search(self, q, k=5, year_min=None, source=None):
        self.last_call = {"q": q, "k": k, "year_min": year_min, "source": source}
        return self.hits


def _hit(cid, text):
    return Hit(cid, text, 0.9, "doc1", ["实验", "1.1 原料"], 1, 1)


def _gen(hits, capture=None):
    g = Generator.__new__(Generator)  # 绕过 __init__(不连真 LLM)
    g.retriever = FakeRetriever(hits)

    class FakeLLM:
        def invoke(self, messages):
            if capture is not None:
                capture.append(messages)

            class R:
                content = "PBAT 牌号为 TH801T [1]。"

            return R()

    g.llm = FakeLLM()
    return g


def test_no_hits_says_so_without_calling_llm():
    """没检索到材料:直接如实说,不调 LLM。

    最危险的情况是让 LLM 在空上下文里凭记忆答 —— 那就是纯幻觉。
    """
    called = []
    g = _gen([], capture=called)
    ans = g.answer("外星人用什么地膜")
    assert ans.used_context is False
    assert "没有" in ans.text
    assert called == []  # LLM 根本没被调


def test_context_is_numbered_for_citation():
    """资料要编号 —— 引用 [1][2] 的前提。"""
    ctx = _format_context([_hit("a", "PBAT牌号TH801T"), _hit("b", "PLA牌号4032D")])
    assert "[1]" in ctx and "[2]" in ctx
    assert "TH801T" in ctx


def test_grounding_instructions_in_prompt():
    """grounding 的规则必须进 prompt —— 这是 RAG 的命根子。"""
    captured = []
    g = _gen([_hit("a", "PBAT牌号TH801T")], capture=captured)
    g.answer("PBAT牌号")
    system_msg = captured[0][0]["content"]
    assert "只依据" in system_msg  # 只用提供的材料
    assert "没有" in system_msg  # 没有就说没有
    assert "引用" in system_msg  # 要标引用


def test_filters_pass_through():
    g = _gen([_hit("a", "x")])
    g.answer("地膜厚度", k=3, year_min=2020, source="标准规范")
    assert g.retriever.last_call == {
        "q": "地膜厚度",
        "k": 3,
        "year_min": 2020,
        "source": "标准规范",
    }


def test_answer_carries_hits_for_verification():
    """答案要带回检索结果 —— 用户要能核实"这话哪儿来的"。"""
    g = _gen([_hit("a", "PBAT牌号TH801T")])
    ans = g.answer("PBAT牌号")
    assert ans.used_context is True
    assert len(ans.hits) == 1
    assert ans.hits[0].document_id == "doc1"
