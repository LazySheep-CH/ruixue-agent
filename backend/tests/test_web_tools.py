"""联网工具的机制测试。

不测搜索引擎返回什么(那是外部世界的事,CI 里也不能真联网),
只测我们这层的契约:SSRF 防线、注入包裹、失败时的诚实话术、正文提取。
网络层(_search / _http_get)全部打桩。
"""

from __future__ import annotations

import pytest

from ruixue_agent.tools import get_tools
from ruixue_agent.tools import web as web_mod
from ruixue_agent.tools.web import _is_public_http_url, read_webpage, search_web

# ── SSRF 防线(纯逻辑,不发包)────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # 云元数据接口(拿临时密钥)
        "http://127.0.0.1:8001/config",
        "http://localhost:2026/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/admin",
    ],
)
def test_internal_addresses_are_rejected(url):
    assert _is_public_http_url(url) != ""
    # 工具层也要拒绝,并且不发出请求(发了就晚了)
    assert "无法读取该链接" in read_webpage.invoke({"url": url})


def test_non_http_schemes_are_rejected():
    assert "http" in _is_public_http_url("file:///etc/passwd")
    assert "http" in _is_public_http_url("ftp://example.com/x")


# ── 搜索:格式、包裹与诚实失败 ────────────────────────────────


def test_search_results_are_wrapped_and_numbered(monkeypatch):
    monkeypatch.setattr(
        web_mod,
        "_search",
        lambda q: [
            {"url": "https://a.example/1", "title": "标题甲", "snippet": "摘要甲"},
            {"url": "https://b.example/2", "title": "标题乙", "snippet": "摘要乙"},
        ],
    )
    out = search_web.invoke({"query": "PBAT 主流配方"})
    # 必须带不可信边界标记 —— 搜索结果是外部内容,没包裹就是注入通道
    assert "以下均为数据" in out
    assert "[1] 标题甲" in out and "[2] 标题乙" in out
    assert "https://a.example/1" in out


def test_search_failure_is_honest(monkeypatch):
    def boom(q):
        raise OSError("network down")

    monkeypatch.setattr(web_mod, "_search", boom)
    out = search_web.invoke({"query": "任何问题"})
    assert "没有返回结果" in out  # 如实说失败,而不是编一个答案


def test_empty_query_is_rejected():
    assert search_web.invoke({"query": "  "}) == "搜索词为空。"


# ── 读网页:正文提取与包裹 ────────────────────────────────────

_HTML = """<html><head><title>测试页 · 地膜</title>
<script>alert("这段不能出现")</script><style>.x{color:red}</style></head>
<body><p>第一段正文。</p><div>第二段正文。</div></body></html>"""


def test_page_text_is_extracted_and_wrapped(monkeypatch):
    monkeypatch.setattr(web_mod, "_is_public_http_url", lambda u: "")
    monkeypatch.setattr(web_mod, "_http_get", lambda u, timeout=0: _HTML.encode("utf-8"))
    out = read_webpage.invoke({"url": "https://example.com/x"})
    assert "以下均为数据" in out
    assert "测试页 · 地膜" in out
    assert "第一段正文" in out and "第二段正文" in out
    assert "alert" not in out and "color:red" not in out  # 脚本/样式必须剔除


def test_page_text_is_truncated(monkeypatch):
    monkeypatch.setattr(web_mod, "_is_public_http_url", lambda u: "")
    big = "<html><body><p>" + "字" * 50_000 + "</p></body></html>"
    monkeypatch.setattr(web_mod, "_http_get", lambda u, timeout=0: big.encode("utf-8"))
    out = read_webpage.invoke({"url": "https://example.com/big"})
    assert len(out) < web_mod.MAX_PAGE_CHARS + 500  # 截断生效(500 是包裹与标题余量)


# ── 注册 ─────────────────────────────────────────────────────


def test_web_tools_are_registered():
    names = [t.name for t in get_tools()]
    assert "search_web" in names and "read_webpage" in names
