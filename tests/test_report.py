"""PDF 报告导出的测试。

重点在几条**错了会出事、但不会报错**的地方:

- 中文折行:按空格折等于不折(中文段落没有空格),文字会画到纸外 ——
  PDF 照样生成成功,只是内容看不见。
- 字体子集化:不做的话每份报告 1.6MB,能用但用户会以为出问题了。
- 文件名不拼用户输入:提问里的换行会造成 HTTP 响应头注入。
- 归属校验:猜到别人的 run_id 就能下载别人的报告 —— 那是数据泄露。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ruixue_app.report import ReportData, filename_for, render_pdf


def _data(**kw) -> ReportData:
    base = {
        "run_id": "abcd1234-5678-90ab-cdef-1234567890ab",
        "question": "尉犁县 50 亩棉花地该用什么配方?",
        "answer": "## 推荐配方\n\n- **PBAT 70% / PLA 30%**,厚度 0.012mm\n- 预计 150 天降解率约 62%\n\n出处:\n[1] GB/T 35795-2017",
        "created_at": datetime(2026, 8, 12, 10, 30, tzinfo=UTC),
    }
    base.update(kw)
    return ReportData(**base)


def _text_of(pdf: bytes) -> str:
    import fitz

    doc = fitz.open(stream=pdf, filetype="pdf")
    return "\n".join(p.get_text() for p in doc)


# ── 基本正确性 ────────────────────────────────────────────────


def test_pdf_is_generated_and_text_is_extractable():
    """文字必须是【真文字】不是图片 —— 用户要能复制、能搜。"""
    pdf = render_pdf(_data())
    assert pdf.startswith(b"%PDF")
    text = _text_of(pdf)
    assert "瑞雪地膜智能助手" in text
    assert "尉犁县" in text
    assert "PBAT 70% / PLA 30%" in text


def test_markdown_markers_do_not_leak_onto_the_page():
    """## 和 ** 是给屏幕看的,印在纸上就是噪声。"""
    text = _text_of(render_pdf(_data()))
    assert "##" not in text
    assert "**" not in text
    assert "推荐配方" in text, "标题文字本身要保留,只去掉标记"


def test_citation_is_preserved():
    """出处是这份报告的价值所在 —— 丢了就成了一份没法核实的结论。"""
    assert "GB/T 35795-2017" in _text_of(render_pdf(_data()))


# ── 字体子集化:不做也能跑,只是每份 1.6MB ──────────────────────


def test_pdf_stays_small_thanks_to_font_subsetting():
    """内置中文字体整套嵌入是 1664 KB;子集化后约 13 KB。

    这条测试的意义:子集化那行要是被谁删了,不会有任何报错,
    只有用户下载时觉得"怎么这么大"。用体积卡死它。
    """
    pdf = render_pdf(_data())
    assert len(pdf) < 200 * 1024, f"报告 {len(pdf) / 1024:.0f} KB —— 字体子集化是不是没生效?"


# ── 中文折行:最容易静默出错的一处 ──────────────────────────────


def test_long_chinese_paragraph_wraps_instead_of_running_off_the_page():
    """中文整段没有空格。按空格折行 = 不折行 = 文字画到纸外。

    PDF 会生成成功、不报错,只是右边的字看不见 —— 所以必须用
    "文字有没有超出页宽"来断言,不能只断言"生成成功"。
    """
    import fitz

    long_zh = "地膜覆盖能显著提高地温并减少土壤水分蒸发" * 12
    pdf = render_pdf(_data(answer=long_zh))
    doc = fitz.open(stream=pdf, filetype="pdf")
    page_width = doc[0].rect.width
    for page in doc:
        for block in page.get_text("blocks"):
            # block = (x0, y0, x1, y1, text, ...);x1 是右边界
            assert block[2] <= page_width, "有文字画出了页面右边界 —— 折行没生效"
    assert len(doc) >= 1


def test_content_longer_than_one_page_creates_more_pages():
    """长答案要翻页,不能把后面的内容画在最后一行上面(那会糊成一团)。"""
    import fitz

    pdf = render_pdf(_data(answer="\n".join(f"第 {i} 条结论,内容若干。" for i in range(120))))
    doc = fitz.open(stream=pdf, filetype="pdf")
    assert len(doc) >= 2
    assert "第 119 条结论" in _text_of(pdf), "翻页后内容不能丢"


# ── 失败的运行 ────────────────────────────────────────────────


def test_failed_run_exports_an_explanation_not_a_crash():
    """用户手里有 run_id,给一份"没跑完"的说明,比让下载按钮报错有用。"""
    text = _text_of(render_pdf(_data(status="failed", answer="")))
    assert "未能完成" in text


@pytest.mark.parametrize("answer", ["", "   \n\n  "])
def test_blank_answer_does_not_crash(answer):
    assert render_pdf(_data(answer=answer)).startswith(b"%PDF")


# ── 文件名:不可信输入不进 HTTP 头 ──────────────────────────────


def test_filename_never_contains_user_input():
    """提问里的换行拼进 Content-Disposition 就是响应头注入。

    文件名只用日期 + run_id 前 8 位,天生不含用户输入 —— 这不是"记得转义",
    是**根本不给它机会**。
    """
    name = filename_for(_data(question="换行\n注入: X-Evil: 1\r\n"))
    assert name == "ruixue-report-20260812-abcd1234.pdf"
    assert "\n" not in name and "\r" not in name
    assert "注入" not in name


# ── 表格:agent 的核心输出形态之一 ────────────────────────────
#
# screen_film_recipes 返回的就是配方对比表。原实现按原文逐行画,
# `|---|---|---|` 这类语法记号会原样印在纸上 —— 用户看到的是一堆竖线,不是表。


_TABLE_MD = """按环境推荐如下:

| 配方 | 降解率 | 拉伸强度 | 结论 |
|---|---:|---:|---|
| PBAT70/PLA30 | 29.3% | 19.7 MPa | 推荐 |
| 纯 PLA | 12.6% | 46.0 MPa | 难降解 |

以上。"""


def _rendered_text(md: str) -> str:
    """渲染一段 Markdown 并取回纸面文字。

    ⚠ 别叫 _text_of —— 本文件上面已有同名函数(参数是 PDF 字节)。
      Python 取最后一个定义,重名会让【上面所有测试】把 PDF 字节当 Markdown 传,
      报一个和真实原因毫不相干的 TypeError。踩过一次。
    """
    import fitz

    pdf = render_pdf(ReportData(run_id="r" * 36, question="q", answer=md))
    return fitz.open(stream=pdf, filetype="pdf")[0].get_text()


def test_table_separator_row_never_reaches_the_page():
    """`|---|---:|` 是 Markdown 的语法记号,印出来就是乱码般的一行。"""
    txt = _rendered_text(_TABLE_MD)
    assert "---" not in txt
    assert "|" not in txt, "竖线还在,说明表格没被解析成表"


def test_table_cells_are_all_preserved():
    """去掉竖线不能连内容一起去掉 —— 每个单元格都得在。"""
    txt = _rendered_text(_TABLE_MD)
    for cell in ["配方", "降解率", "PBAT70/PLA30", "29.3%", "19.7 MPa", "推荐", "纯 PLA"]:
        assert cell in txt, f"单元格「{cell}」丢了"


def test_text_around_the_table_is_not_swallowed():
    """表格前后的正文要照常输出 —— 表格块的收尾没做对时最容易吞掉后一段。"""
    txt = _rendered_text(_TABLE_MD)
    assert "按环境推荐如下" in txt
    assert "以上" in txt


def test_ragged_table_does_not_crash():
    """模型偶尔会漏写一个单元格,列数对不齐不能把导出搞崩。"""
    md = "| a | b | c |\n|---|---|---|\n| 1 | 2 |\n| 1 | 2 | 3 | 4 |"
    txt = _rendered_text(md)
    assert "a" in txt and "4" in txt


def test_overlong_cell_is_truncated_not_drawn_off_page():
    """超宽单元格必须截断。画出纸外【不会报错】,只是看不见 —— 最坏的一类失败。"""
    import fitz

    long_cell = "非常长的配方说明" * 40
    md = f"| 名称 | 说明 |\n|---|---|\n| A | {long_cell} |"
    pdf = render_pdf(ReportData(run_id="r" * 36, question="q", answer=md))
    page = fitz.open(stream=pdf, filetype="pdf")[0]
    for block in page.get_text("blocks"):
        # block[2] 是右边界 x。A4 宽 595,右边距 56 → 内容不该越过 540
        assert block[2] <= 545, f"有内容画到了纸外:x2={block[2]}"


# ── 下载接口:归属校验是【安全边界】,不是功能 ──────────────────
#
# 上面全是渲染层的测试(纯函数、不连库)。但这份报告是通过 HTTP 下载的,
# 而**猜到别人的 run_id 就能下载别人的报告 = 数据泄露**。
# 渲染测得再细也覆盖不到这一层 —— 必须走真接口。


_KEY_ALICE = "report-test-alice"
_KEY_BOB = "report-test-bob"


def _deps_up() -> bool:
    try:
        from sqlalchemy import text

        from ruixue_agent.persistence.engine import get_engine

        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _deps_up(), reason="需要 PostgreSQL")


@pytest.fixture
def api(monkeypatch):
    from fastapi.testclient import TestClient

    import ruixue_app.auth as auth
    import ruixue_app.main as main_mod

    monkeypatch.setattr(auth, "API_KEYS", {_KEY_ALICE: "alice-rpt", _KEY_BOB: "bob-rpt"})
    return TestClient(main_mod.app)


@needs_db
def _make_run(user_id: str) -> str:
    """直接落一条 run 记录 —— 不必真跑 agent,这里测的是下载口不是对话。"""
    import uuid

    from sqlalchemy.orm import Session

    from ruixue_agent.persistence.engine import get_engine
    from ruixue_agent.persistence.models import RunRow

    rid = str(uuid.uuid4())
    with Session(get_engine()) as s:
        s.add(
            RunRow(
                run_id=rid,
                user_id=user_id,
                thread_id=f"{user_id}:t1",
                question="尉犁县 50 亩用什么膜?",
                status="succeeded",
                answer="推荐 PBAT70/PLA30,约需 600 公斤。",
            )
        )
        s.commit()
    return rid


@needs_db
def test_owner_can_download_and_gets_a_real_pdf(api):
    rid = _make_run("alice-rpt")
    r = api.get(f"/chat/runs/{rid}/report.pdf", headers={"X-API-Key": _KEY_ALICE})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    assert "attachment" in r.headers["content-disposition"]


@needs_db
def test_another_user_cannot_download_someone_elses_report(api):
    """核心安全断言:换个身份就必须 404。

    返回 404 而不是 403 是刻意的 —— 403 等于告诉对方"这个 run_id 存在",
    是一个可枚举的信息泄露。和 /chat/runs/{id} 查状态保持一致。
    """
    rid = _make_run("alice-rpt")
    r = api.get(f"/chat/runs/{rid}/report.pdf", headers={"X-API-Key": _KEY_BOB})
    assert r.status_code == 404, "别人的报告被下走了 —— 这是数据泄露"


@needs_db
def test_no_credential_is_rejected(api):
    rid = _make_run("alice-rpt")
    assert api.get(f"/chat/runs/{rid}/report.pdf").status_code == 401


@needs_db
def test_unknown_run_id_is_404_not_500(api):
    r = api.get(
        "/chat/runs/00000000-0000-0000-0000-000000000000/report.pdf",
        headers={"X-API-Key": _KEY_ALICE},
    )
    assert r.status_code == 404
