"""BM25 词法检索的测试。

核心逻辑(挑罕见词)不连库也能测 —— 用假 session 控制词频表返回什么,
检查 _rare_terms_query 的挑词行为。这是整个模块最容易出错、也踩坑最多的地方。
"""

from __future__ import annotations

from ruixue_agent.rag.bm25 import Bm25Search


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalar(self):
        return self._rows


class FakeSession:
    """假 session:按 SQL 里的关键词返回预设结果。

    只需要覆盖 _rare_terms_query 用到的两个查询:
      父块总数(scalar)、词频表(all)。
    """

    def __init__(self, total: int, df: dict[str, int]):
        self.total = total
        self.df = df

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "count(*)" in sql:
            return FakeResult(self.total)
        if "term_df" in sql:
            terms = params["t"]
            return FakeResult([(t, self.df[t]) for t in terms if t in self.df])
        return FakeResult([])


def _query(total, df, question):
    b = Bm25Search(FakeSession(total, df))
    return b._rare_terms_query(question)


def test_keeps_only_rare_terms():
    """★ 核心:常见词被丢掉,只留罕见词。

    这是整个模块的命根子。实测:一个罕见词 + 七个常见词做 OR,
    常见词把候选灌到几万,罕见词的定位能力被淹没,又慢又不准。
    """
    # 地膜 44% 常见,PBAT 罕见(总 1000 块,地膜出现 440,PBAT 出现 5)
    q = _query(1000, {"地膜": 440, "pbat": 5}, "PBAT地膜")
    assert "pbat" in q  # 罕见词留下
    assert "地膜" not in q  # 常见词(44% > 1%)被丢


def test_terms_not_in_df_are_rarest():
    """词频表里【没有】的词 = df 0 = 最罕见,必须保留。

    踩过:早期误判"查询里没有罕见词",因为只看了 df=0 的词却把它们算成"有"。
    索引里没有的词恰恰是最该用的。
    """
    q = _query(1000, {"土壤": 400}, "沙雅县土壤")  # 沙雅县不在词频表
    assert "沙雅县" in q  # df=0,最罕见,留
    assert "土壤" not in q  # 40%,丢


def test_all_common_keeps_some():
    """全是常见词时,退而取最罕见的几个 —— 不能返回空、也不能返回全部。

    踩过:fallback 用全部词,导致候选灌爆、比不 fallback 还慢。
    """
    q = _query(1000, {"土壤": 400, "地膜": 440, "影响": 370}, "地膜对土壤的影响")
    assert q  # 不为空
    terms = q.split(" | ")
    assert len(terms) <= 3  # 只留最罕见的几个,不是全部


def test_stopwords_dropped():
    """纯语法功能词(的/了/是)直接扔,不进查询。"""
    q = _query(1000, {"pbat": 5}, "PBAT的降解是怎么样的")
    assert "'的'" not in q and "'是'" not in q


def test_standard_code_survives():
    """标准号是罕见词的典型,必须整体保留(不被拆、不被当常见词丢)。"""
    q = _query(1000, {}, "符合GB/T 35795的规定")
    assert "gb/t35795" in q


def test_empty_query():
    assert _query(1000, {}, "") == ""
    assert _query(1000, {}, "的了是") == ""  # 全是停用词
