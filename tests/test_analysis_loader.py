"""上传数据契约与 CSV 加载的测试。

这一层的失败方式几乎都是**静默的**,所以测试要盯的是"错了不会报错"的那些:

- 编码判错 → 第一个列名带上 BOM → 归一认不出 → 少认一列,不报错。
- 列名猜错(把 PBAT 认成 PLA)→ 整表预测全错,不报错。
- 数值硬转 float → 一个「约 30」让整行报废,或者被塞成 0。
- 没有实测列 → 要等 agent 分析到一半才发现无从对比。
"""

from __future__ import annotations

import pytest

from ruixue_agent.analysis import DatasetError, load_csv, map_columns, parse_number, summarize

_GOOD = "PLA含量(%),PBAT含量,厚度(um),覆膜天数,降解率%\n30,70,12,120,28.5\n50,50,10,120,41.2\n"


def _csv(text: str, enc: str = "utf-8") -> bytes:
    return text.encode(enc)


# ── 列名归一:错了是静默的 ────────────────────────────────────


def test_chinese_headers_with_units_are_recognized():
    """用户不会写 PLA_pct。认不出这些,整张表就只能退回。"""
    cm = map_columns(["PLA含量(%)", "PBAT 比例", "厚度(um)", "覆膜天数", "降解率"])
    assert cm.features["PLA_pct"] == "PLA含量(%)"
    assert cm.features["PBAT_pct"] == "PBAT 比例"
    assert cm.features["Thickness_um"] == "厚度(um)"
    assert cm.targets["DR"] == "降解率"


def test_pbat_is_never_mistaken_for_pla():
    """把 PBAT 认成 PLA 会让整表预测全错,而且不报错 —— 最坏的一类失败。"""
    cm = map_columns(["PBAT含量", "PLA含量"])
    assert cm.features["PBAT_pct"] == "PBAT含量"
    assert cm.features["PLA_pct"] == "PLA含量"


def test_unknown_columns_are_kept_not_guessed():
    """认不出就是认不出。猜一个映射比留空危险得多。"""
    cm = map_columns(["降解率", "地块编号", "记录人", "备注"])
    assert set(cm.unknown) == {"地块编号", "记录人", "备注"}


def test_target_aliases_cover_common_wordings():
    for header, model in [
        ("失重率", "DR"),
        ("质量损失率", "DR"),
        ("抗拉强度", "TS"),
        ("透湿量", "WVTR"),
    ]:
        assert map_columns([header]).targets.get(model) == header


# ── 编码:BOM 是最容易漏的一个 ──────────────────────────────────


def test_utf8_bom_does_not_break_the_first_header():
    """Excel 导出的 UTF-8 CSV 常带 BOM。不剥掉,第一个列名变成 "﻿PLA含量",
    归一认不出 —— 而且【不报错】,只是少认一列。"""
    cm, rows = load_csv(_csv(_GOOD, "utf-8-sig"))
    assert "PLA_pct" in cm.features
    assert rows[0]["PLA_pct"] == 30.0


def test_gbk_file_is_decoded():
    """国内 Excel 默认导出 GBK。直接按 UTF-8 解会抛 UnicodeDecodeError,
    而 "invalid start byte" 对用户毫无意义。"""
    cm, rows = load_csv(_csv(_GOOD, "gbk"))
    assert cm.targets["DR"] == "降解率%"
    assert len(rows) == 2


def test_undecodable_bytes_give_an_actionable_message():
    with pytest.raises(DatasetError, match="另存为"):
        load_csv(b"\xff\xfe\x00\x00\xff\xff\xfe\xfe")


def test_semicolon_delimiter_is_detected():
    cm, rows = load_csv(_csv("PLA含量;厚度;降解率\n30;12;28.5\n"))
    assert rows[0]["PLA_pct"] == 30.0


# ── 数值解析:宽进严出 ────────────────────────────────────────


@pytest.mark.parametrize(
    "cell,expected",
    [
        ("12.5", 12.5),
        ("12.5%", 12.5),
        ("1,234", 1234.0),
        ("约 30", 30.0),
        (" 8.2 ℃", 8.2),
        ("-3.5", -3.5),
    ],
)
def test_messy_numbers_are_parsed(cell, expected):
    assert parse_number(cell) == expected


@pytest.mark.parametrize("cell", ["", "  ", "—", "N/A", "无", "未测", "高", "偏酸", "见备注"])
def test_non_numeric_becomes_missing_not_zero(cell):
    """描述性文字是【真缺失】,不能硬编成 0 —— 0 会被当成一个真实测量值,
    把均值和预测全部拉偏,而且不报错。"""
    assert parse_number(cell) is None


def test_short_row_does_not_crash():
    """田间表常有末尾缺列。整行报废或索引越界都不该发生。"""
    cm, rows = load_csv(_csv("PLA含量,厚度,降解率\n30,12,28.5\n50\n"))
    assert len(rows) == 2
    assert rows[1]["PLA_pct"] == 50.0
    assert rows[1]["target:DR"] is None


# ── 契约:必填与上限 ──────────────────────────────────────────


def test_missing_target_column_is_rejected_at_upload_time():
    """没有实测列,这张表就无从和预测对比。在上传时拦住,
    而不是等 agent 分析到一半才说"没数据"。"""
    with pytest.raises(DatasetError, match="实测值"):
        load_csv(_csv("PLA含量,厚度\n30,12\n"))


def test_header_only_file_is_rejected():
    with pytest.raises(DatasetError, match="没有数据行"):
        load_csv(_csv("PLA含量,降解率\n"))


def test_empty_file_is_rejected():
    with pytest.raises(DatasetError, match="空"):
        load_csv(b"")


def test_oversized_file_is_rejected_by_bytes():
    """体积闸门要在解析之前 —— 先 decode 再判大小等于已经把它读进内存了。"""
    with pytest.raises(DatasetError, match="超过上限"):
        load_csv(b"x" * (5 * 1024 * 1024))


def test_too_many_rows_is_rejected():
    body = "\n".join("30,12,28.5" for _ in range(6000))
    with pytest.raises(DatasetError, match="超过上限"):
        load_csv(_csv(f"PLA含量,厚度,降解率\n{body}\n"))


# ── 概览:缺失率必须显式报出来 ──────────────────────────────────


def test_summary_reports_missing_rate_per_column():
    """一列 90% 是空的,分析时基本无效。用户该在上传那一刻知道,
    而不是等结论出来才发现是拿默认值算的。"""
    cm, rows = load_csv(_csv("PLA含量,厚度,降解率\n30,,28.5\n50,,41.2\n40,10,33.0\n"))
    s = summarize(cm, rows)
    assert s["n_rows"] == 3
    assert s["stats"]["Thickness_um"]["missing_pct"] == pytest.approx(66.7, abs=0.1)
    assert s["stats"]["PLA_pct"]["missing_pct"] == 0.0
    assert s["stats"]["PLA_pct"]["mean"] == pytest.approx(40.0)


def test_summary_lists_unrecognized_columns():
    """没认出的列要报出来 —— 用户可能只是列名写法不同,看到了才知道能改。"""
    cm, rows = load_csv(_csv("降解率,地块编号\n28.5,A-01\n"))
    assert summarize(cm, rows)["unrecognized_columns"] == ["地块编号"]


# ── 离群检测:z 分数的自指陷阱 ─────────────────────────────────


def _ds(rows_csv: str):
    """构造一个内存里的 Dataset,不连库 —— 这几条测的是统计逻辑。"""
    from ruixue_agent.analysis.store import Dataset

    cm, rows = load_csv(_csv(rows_csv))
    return Dataset(
        dataset_id="d",
        user_id="u",
        filename="t.csv",
        n_rows=len(rows),
        n_cols=len(cm.features) + len(cm.targets),
        columns={"features": cm.features, "targets": cm.targets, "unknown": list(cm.unknown)},
        rows=rows,
        created_at=None,
    )


_MASKING = (
    "PLA含量,厚度,降解率\n"
    "30,12,28.5\n30,12,31.2\n30,10,33.0\n50,12,41.2\n"
    "30,12,22.4\n30,12,35.1\n30,8,29.9\n30,12,88.0\n"
)


def test_lone_extreme_value_is_caught_despite_masking():
    """**这条是核心回归测试。**

    普通 z 分数会漏掉这个 88.0:离群点自己把均值(38.663)和标准差(20.652)
    撑大了,|88−38.663|=49.3 < 2.5×20.652=51.6 —— 最明显的异常反而藏住了。
    中位数 + MAD 没有这个自指问题(一个点动不了中位数)。
    """
    from ruixue_agent.analysis import analyze

    out = analyze.detect_outliers(_ds(_MASKING))
    assert "第8行=88.0" in out, "最明显的离群点被漏掉了 —— 是不是又用回均值+标准差了?"


def test_clean_data_reports_no_outliers():
    """不能见风就是雨:正常波动不该被报成异常。"""
    from ruixue_agent.analysis import analyze

    clean = "PLA含量,厚度,降解率\n" + "".join(f"30,12,{28 + i * 0.5}\n" for i in range(10))
    assert "未发现明显离群值" in analyze.detect_outliers(_ds(clean))


def test_constant_column_is_skipped_not_reported_as_all_outliers():
    """整列填同一个值时 MAD=0,任何偏离都会被判成无穷大离群 ——
    那不是发现,是噪声。必须跳过。"""
    from ruixue_agent.analysis import analyze

    same = "PLA含量,厚度,降解率\n" + "30,12,30.0\n" * 7 + "30,12,31.0\n"
    out = analyze.detect_outliers(_ds(same))
    assert "未发现明显离群值" in out


def test_tiny_sample_makes_no_judgement():
    """5 个点算不出有意义的离散度,硬报异常会把正常波动说成问题。"""
    from ruixue_agent.analysis import analyze

    tiny = "PLA含量,厚度,降解率\n30,12,28.5\n30,12,31.2\n30,12,99.0\n"
    out = analyze.detect_outliers(_ds(tiny))
    assert "未发现明显离群值" in out
    assert "参考价值有限" in out, "小样本必须提示,不能让用户以为这个结论很硬"
