"""环境解析测试:地名匹配 + 土壤查表 + 气候取数降级。

土壤部分离线可测;气候要联网,单独标记(没网自动跳过,不算失败)。
"""

import pytest

from ruixue_agent.predictors.environment import get_environment, get_soil, resolve_location
from ruixue_agent.predictors.schema import UV_PER_UVA_MJ
from ruixue_agent.tools import get_tools
from ruixue_agent.tools.environment import get_climate_info, get_soil_info

_HAS_SOIL = None
try:
    from ruixue_agent.predictors.environment import _county_soil

    _county_soil()
    _HAS_SOIL = True
except Exception:
    _HAS_SOIL = False

pytestmark = pytest.mark.skipif(not _HAS_SOIL, reason="county_soil.csv 未生成")


# ── 地名解析(纯离线)──────────────────────────────────────────
def test_resolve_exact_county():
    r = resolve_location("尉犁县")
    assert r["matched"] == "尉犁县"
    assert 85 < r["lon"] < 88 and 39 < r["lat"] < 42  # 新疆巴州范围


def test_resolve_city_name_maps_to_county():
    """市名(南京)→ 代表性区县。"""
    r = resolve_location("南京")
    assert r is not None
    assert 118 < r["lon"] < 120 and 31 < r["lat"] < 33


def test_resolve_with_province_prefix():
    assert resolve_location("新疆尉犁县")["matched"] == "尉犁县"


def test_resolve_unknown_returns_none():
    assert resolve_location("这个地方不存在啊") is None


# ── 土壤特征:质地三项应≈100%,pH 在合理区间 ────────────────────
def test_soil_features_physically_sane():
    soil = resolve_location("尉犁县")["soil"]
    texture = soil["soil_clay_pct"] + soil["soil_silt_pct"] + soil["soil_sand_pct"]
    assert 95 <= texture <= 105  # 粘+粉+砂 ≈ 100%
    assert 3 <= soil["Soil_pH"] <= 11


# ── 整体:未知地点要优雅失败,不抛异常 ──────────────────────────
def test_unknown_place_returns_not_ok():
    e = get_environment("火星基地", 90)
    assert e["ok"] is False and "未找到" in e["reason"]


# ── 联网:气候取数 + UV 标定换算(没网则跳过)────────────────────
@pytest.mark.parametrize("place,days", [("尉犁县", 90)])
def test_climate_and_uv_calibration(place, days):
    e = get_environment(place, days)
    assert e["ok"]
    if "在线" not in e["sources"]["气候"]:
        pytest.skip("NASA POWER 不可达(离线环境),已降级为默认值兜底")
    f = e["features"]
    assert -30 < f["Temperature_C"] < 60
    assert 0 <= f["Humidity"] <= 100
    # UV = UVA(MJ/m²/d) × 常数 × 天数,应落在训练数据量级内(0~77万)
    assert 0 < f["UV"] < 1.2e6
    assert f["UV"] / (UV_PER_UVA_MJ * days) < 5  # 反推的日 UVA 应是个位数 MJ


# ── 土壤/气候作为【独立工具】暴露 ──────────────────────────────
def test_soil_and_climate_tools_registered():
    names = [t.name for t in get_tools()]
    assert "get_soil_info" in names and "get_climate_info" in names


def test_get_soil_is_offline_and_standalone():
    """土壤查询纯离线:不联网也必须能返回完整结果。"""
    r = get_soil("寿光市")
    assert r["ok"] and "SoilGrids" in r["source"]
    assert "Soil_pH" in r["features"]


def test_soil_tool_output_readable():
    txt = get_soil_info.invoke({"place": "寿光市"})
    assert "pH" in txt and "数据源" in txt


def test_climate_tool_handles_unknown_place():
    """未知地点要给可读提示,不抛异常。"""
    txt = get_climate_info.invoke({"place": "火星基地", "days": 30})
    assert "未找到" in txt
