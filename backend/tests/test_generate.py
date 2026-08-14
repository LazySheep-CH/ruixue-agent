"""生成的测试。LLM 不真调(慢、要联网),测的是编排 + grounding 的骨架。"""

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


# ── 时效性:结构性约束优于提示词约束 ──────────────────────────


def _dated_hit(text="PBAT 价格约 1.2 万元/吨", year=2023, title="生物降解材料产业报告"):
    from ruixue_agent.rag.retriever import Hit

    return Hit(
        chunk_id="c1",
        text=text,
        score=1.0,
        document_id="abc123",
        section_path=["第三章"],
        page_start=1,
        page_end=1,
        title=title,
        year=year,
    )


def test_context_always_carries_the_year():
    """不标年份,2023 年论文里的价格就会被当成"现在的价格"答给用户,
    而用户会拿这个数去采购。评测里真出现过。"""
    from ruixue_agent.rag.generate import _format_context

    ctx = _format_context([_dated_hit()])
    assert "2023 年" in ctx
    assert "生物降解材料产业报告" in ctx


def test_missing_year_says_so_instead_of_omitting():
    """省略年份 = 让模型默认这段资料是当前有效的。必须显式写"年份未知"。"""
    from ruixue_agent.rag.generate import _format_context

    assert "年份未知" in _format_context([_dated_hit(year=None)])


def test_generation_fingerprint_changes_when_rules_change(monkeypatch):
    """缓存键必须覆盖一切影响取值的东西。

    原来的键只有"问题 + 知识库版本"。改了生成规则后,缓存里那些旧格式的答案
    照样命中 —— 改动对老用户静默失效,不报任何错,你以为上线了其实没有。
    """
    import ruixue_agent.rag.generate as g

    before = g.generation_fingerprint()
    g.generation_fingerprint.cache_clear()
    monkeypatch.setattr(g, "_SYSTEM", g._SYSTEM + "\n9. 新规则")
    after = g.generation_fingerprint()
    g.generation_fingerprint.cache_clear()
    assert before != after, "改了生成规则,缓存指纹必须变"


def test_display_title_falls_back_to_filename_when_extraction_failed():
    """1578 篇里 16% 的 title 是抽取失败的产物(封面页眉/学校名),
    而这些文档的文件名恰恰就是真标题。假标题是纯噪声,还占 token。"""
    from ruixue_agent.persistence.repository import display_title

    assert (
        display_title("硕 士 学 位 论 文", "地膜残留对旱地玉米农田土壤理化性状的影响.pdf")
        == "地膜残留对旱地玉米农田土壤理化性状的影响"
    )
    assert display_title("西北农林科技大学", "春小麦地膜覆盖的效应及其范式_1.pdf") == (
        "春小麦地膜覆盖的效应及其范式"
    )
    assert display_title("", "小地老虎发生趋势预报.pdf") == "小地老虎发生趋势预报"
    # 真标题要保留,别被兜底逻辑误伤
    real = "全生物降解地膜对次生盐碱地滴灌春玉米根区水盐分布的影响"
    assert display_title(real, "whatever.pdf") == real
