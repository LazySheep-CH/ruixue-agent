"""环境查询工具:按地点查土壤 / 查气候。

独立成两个工具(而非藏在预测里),因为它们本身就是可独立回答的问题
("这地方土壤怎么样""今年雨水多吗"),且 agent 可自由组合使用。
两者都是确定性能力(查表 / 调 API,不需要推理)→ 是 tool 而非 subagent。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from ruixue_agent.predictors.environment import get_climate, get_soil

# Air_Humidity_pct 与 Humidity 同值(只是不同模型的列名不同),展示时不重复列出。
_HIDDEN = {"Air_Humidity_pct"}


def _fmt(features: dict, labels: dict) -> list[str]:
    return [f"  {labels.get(k, k)}: {v}" for k, v in features.items() if k not in _HIDDEN]


SOIL_LABELS = {
    "Soil_pH": "pH(酸碱度)",
    "soil_clay_pct": "粘粒 %",
    "soil_silt_pct": "粉粒 %",
    "soil_sand_pct": "砂粒 %",
    "Soil_OC_gkg": "有机碳 g/kg",
    "Soil_TN_mgkg": "全氮",
    "soil_bulk_density": "容重",
    "soil_coarse_frag_pct": "粗碎屑",
}
CLIMATE_LABELS = {
    "Temperature_C": "平均气温 ℃",
    "Humidity": "相对湿度 %",
    "solar_rad_MJm2d": "日均太阳辐射 MJ/m²·d",
    "Precipitation_mm": "累计降水 mm",
    "UV": "累计紫外剂量",
    "soil_moisture_pct": "表层土壤含水 %",
}


@tool
def get_soil_info(place: str) -> str:
    """查询某地的土壤情况(pH、有机碳、质地、容重等)。

    数据源:SoilGrids 0-5cm 表层,2898 个县离线表(零网络、秒回)。
    参数 place:县/区级或市级地名,如"尉犁县""新疆尉犁""南京"。
    适用:用户问"这地方土壤怎么样""土壤酸碱度多少""适不适合埋降解膜"。
    """
    r = get_soil(place)
    if not r["ok"]:
        return r["reason"]
    lines = [f"{r['place']}({r['lon']:.2f}, {r['lat']:.2f})土壤情况:"]
    lines += _fmt(r["features"], SOIL_LABELS)
    lines.append(f"数据源:{r['source']}")
    return "\n".join(lines)


@tool
def get_climate_info(place: str, days: int = 90) -> str:
    """查询某地某时段的气候情况(气温、降水、辐射、紫外、湿度)。

    数据源:NASA POWER 日尺度实况(1981–近实时);降水/紫外按天数换算成累计量。
    参数:place 县/区级或市级地名;days 关注的天数(默认 90,如一个种植季)。
    适用:用户问"这地方雨水多吗""日照强不强""紫外线怎么样"。
    """
    r = get_climate(place, days)
    if not r["ok"]:
        return r.get("reason", "气候数据暂不可用")
    lines = [f"{r['place']}({r['lon']:.2f}, {r['lat']:.2f}) {r['period']} 气候:"]
    lines += _fmt(r["features"], CLIMATE_LABELS)
    lines.append(f"数据源:{r['source']}")
    return "\n".join(lines)


@tool
def get_weather_forecast(place: str, days: int = 7) -> str:
    """查询某地未来几天的天气预报(气温、降水、太阳辐射、紫外、风速)。

    数据源:Open-Meteo 预报,最多 16 天。
    参数:place 县/区级或市级地名;days 预报天数(默认 7,上限 16)。

    适用:用户问"下周适合铺膜吗""这几天会不会下雨""要刮大风吗"——
    任何关于未来的天气问题。
    不适用:问"这地方常年气候怎么样""过去一季的降水" —— 那是
    get_climate_info(历史实况),两者数据源和含义都不同,不要混用。

    注意:返回的是预报不是实况,存在不确定性,回答时必须如实说明。
    """
    from ruixue_agent.predictors.forecast import format_forecast, get_forecast

    return format_forecast(get_forecast(place, days))


def get_environment_tools() -> list[BaseTool]:
    return [get_soil_info, get_climate_info, get_weather_forecast]
