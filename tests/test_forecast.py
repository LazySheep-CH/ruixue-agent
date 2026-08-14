"""天气预报(Open-Meteo)的测试。

**不打真网络** —— 外部服务的可用性不该决定我们 CI 的红绿。这里把取数函数
打桩,测的是我们自己的那几段逻辑:参数夹取、失败降级、来源声明。

重点在两条【错了会出事但不会报错】的地方:

- **失败必须降级不能抛**:外部服务挂掉是常态。抛出去会被中间件降级成
  一句"功能暂时不可用",用户不知道是天气服务的问题还是整个助手坏了。
- **必须声明这是预报**:预报会变、会错。不声明,用户就把概率当成事实 ——
  和"检索资料必须带年份"是同一条原则(见 rag/generate.py)。
"""

from __future__ import annotations

import pytest

from ruixue_agent.predictors import forecast as fc

_FAKE_LOC = {"matched": "尉犁县", "lat": 41.34, "lon": 86.26}


def _fake_payload(n: int) -> dict:
    return {
        "daily": {
            "time": [f"2026-08-{13 + i:02d}" for i in range(n)],
            "temperature_2m_max": [38.1] * n,
            "temperature_2m_min": [23.0] * n,
            "precipitation_sum": [0.0] * n,
            "shortwave_radiation_sum": [23.19] * n,
            "uv_index_max": [7.15] * n,
            "wind_speed_10m_max": [20.8] * n,
        }
    }


@pytest.fixture
def stub(monkeypatch):
    """地点解析和取数都打桩;返回一个记录"实际请求了几天"的列表。"""
    asked: list[int] = []

    def fake_fetch(lat, lon, days):
        asked.append(days)
        return _fake_payload(days)

    monkeypatch.setattr(fc, "resolve_location", lambda p: _FAKE_LOC if p != "火星" else None)
    monkeypatch.setattr(fc, "_fetch", fake_fetch)
    return asked


# ── 参数夹取:越界要在我们这层挡住 ──────────────────────────────


@pytest.mark.parametrize(
    "asked_days,expected",
    # 非正数一律回落到默认 7 —— 0 给 7 天而 -3 给 1 天是不一致的,踩过。
    [(7, 7), (0, 7), (-3, 7), (99, fc.MAX_FORECAST_DAYS), (16, 16), ("abc", 7)],
)
def test_days_are_clamped_before_calling_the_api(stub, asked_days, expected):
    """Open-Meteo 超过 16 天直接报错。在我们这层夹住,用户看到的是正常结果,
    而不是一个 400 —— 外部接口的错误码不该穿透到用户面前。"""
    res = fc.get_forecast("尉犁县", asked_days)
    assert stub == [expected]
    assert res["days"] == expected


def test_days_none_falls_back_to_default(stub):
    assert fc.get_forecast("尉犁县", None)["days"] == 7


# ── 失败降级:必须返回结构化结果,不能抛 ────────────────────────


def test_unknown_place_is_reported_not_raised(stub):
    res = fc.get_forecast("火星", 7)
    assert res["ok"] is False
    assert "火星" in res["reason"]


def test_network_failure_degrades_instead_of_raising(monkeypatch):
    """外部服务挂掉是常态。抛异常会被中间件降级成"功能暂时不可用",
    用户分不清是天气服务的问题还是助手整体坏了。"""
    monkeypatch.setattr(fc, "resolve_location", lambda p: _FAKE_LOC)
    monkeypatch.setattr(fc, "_fetch", lambda *a: None)
    res = fc.get_forecast("尉犁县", 7)
    assert res["ok"] is False
    assert "暂时不可用" in res["reason"]
    assert res["place"] == "尉犁县", "失败时也该告诉用户查的是哪儿"


def test_malformed_payload_does_not_crash(monkeypatch):
    """接口改了字段名 / 返回空对象,不能把整个回答搞崩。"""
    monkeypatch.setattr(fc, "resolve_location", lambda p: _FAKE_LOC)
    monkeypatch.setattr(fc, "_fetch", lambda *a: {"unexpected": 1})
    assert fc.get_forecast("尉犁县", 7)["ok"] is False


# ── 来源声明:这是全模块最重要的一条 ────────────────────────────


def test_output_always_declares_it_is_a_forecast(stub):
    """不声明"这是预报",用户会把一个概率当成事实。

    和"检索资料必须带年份"是同一条原则:**让用户能判断这个数字能不能信**。
    """
    text = fc.format_forecast(fc.get_forecast("尉犁县", 3))
    assert "预报" in text
    assert "非实况" in text or "不是实况" in text
    assert "不确定" in text, "必须提示预报会不准"


def test_output_names_the_data_source_and_issue_date(stub):
    """预报每天都会变。不记发布日期,就没法解释"上次问的怎么不一样"。"""
    text = fc.format_forecast(fc.get_forecast("尉犁县", 3))
    assert "Open-Meteo" in text
    assert "发布于" in text


def test_wind_speed_is_present(stub):
    """风速是这一路数据【独有】的价值 —— 故障诊断专家此前只能说"需用户自查"。
    它要是被谁从 _DAILY 里删了,不会报错,只会让诊断退回猜测。"""
    text = fc.format_forecast(fc.get_forecast("尉犁县", 2))
    assert "风速" in text
    assert "wind_speed_10m_max" in fc._DAILY


def test_failure_text_is_readable_not_a_stack_trace(monkeypatch):
    monkeypatch.setattr(fc, "resolve_location", lambda p: None)
    text = fc.format_forecast(fc.get_forecast("不存在的地方", 7))
    assert text.startswith("取不到天气预报")
