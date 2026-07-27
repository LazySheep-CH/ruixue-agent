"""环境解析:地点(+时段)→ 模型需要的土壤/气候特征。

三个来源:
  土壤 → data/predictors/county_soil.csv(离线抽好的 2898 县 SoilGrids,零网络)
  气候 → NASA POWER 在线 API(大陆实测 2s 可用,1981–近实时;失败则跳过,由默认值兜底)
  UV   → 由 NASA 的 UVA 按实证标定常数换算(见 schema.UV_PER_UVA_MJ)

设计:土壤离线、气候在线 + 本地缓存。任一环节失败都【不抛异常】,只是少给几个特征,
由预测层的默认值兜底 —— 保证"没网也能出结果,只是精度下降",并如实标注。
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import pandas as pd

from ruixue_agent.predictors.schema import UV_PER_UVA_MJ

logger = logging.getLogger("ruixue.env")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "predictors"
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
# NASA 参数 → 我们要的量(日尺度)
POWER_PARAMS = "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,ALLSKY_SFC_UVA,GWETTOP"


@cache
def _county_soil() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "county_soil.csv", encoding="utf-8")


# 县级表里只有区/县名(如"江宁区"),没有市名。用户常说市名,故补一张
# 「市名 → 该市一个代表性区县」的映射,覆盖主要农业/地膜使用区。
CITY_TO_COUNTY = {
    "北京": "朝阳区",
    "上海": "浦东新区",
    "天津": "武清区",
    "重庆": "涪陵区",
    "南京": "江宁区",
    "苏州": "吴中区",
    "无锡": "宜兴市",
    "徐州": "铜山区",
    "盐城": "大丰区",
    "南通": "如东县",
    "扬州": "邗江区",
    "连云港": "东海县",
    "济南": "章丘区",
    "青岛": "平度市",
    "潍坊": "寿光市",
    "临沂": "兰陵县",
    "郑州": "中牟县",
    "洛阳": "孟津区",
    "新乡": "原阳县",
    "周口": "扶沟县",
    "石家庄": "藁城区",
    "保定": "定州市",
    "邯郸": "永年区",
    "张家口": "张北县",
    "乌鲁木齐": "米东区",
    "喀什": "疏勒县",
    "阿克苏": "库车市",
    "库尔勒": "尉犁县",
    "石河子": "沙湾市",
    "昌吉": "呼图壁县",
    "伊犁": "伊宁县",
    "哈密": "伊州区",
    "兰州": "榆中县",
    "武威": "凉州区",
    "张掖": "甘州区",
    "酒泉": "肃州区",
    "银川": "贺兰县",
    "西安": "长安区",
    "咸阳": "武功县",
    "呼和浩特": "土默特左旗",
    "包头": "土默特右旗",
    "赤峰": "松山区",
    "通辽": "科尔沁区",
    "成都": "双流区",
    "昆明": "宜良县",
    "太原": "清徐县",
    "沈阳": "辽中区",
    "长春": "农安县",
    "哈尔滨": "双城区",
    "合肥": "肥东县",
    "武汉": "江夏区",
    "长沙": "宁乡市",
    "南昌": "南昌县",
    "杭州": "余杭区",
    "宁波": "余姚市",
    "福州": "闽侯县",
    "广州": "从化区",
    "深圳": "宝安区",
    "南宁": "武鸣区",
    "海口": "琼山区",
}


def resolve_location(place: str) -> dict | None:
    """地名 → 经纬度 + 土壤特征。支持"尉犁县""江苏南京""南京市"等写法。"""
    df = _county_soil()
    p = place.strip().replace("省", "").replace("自治区", "")
    hit = df[df["name"] == p]
    if hit.empty:  # 市名 → 代表性区县
        for city, county in CITY_TO_COUNTY.items():
            if city in p:
                hit = df[df["name"] == county]
                break
    if hit.empty:  # 最后:包含匹配(如"新疆尉犁县"含"尉犁县")
        hit = df[df["name"].apply(lambda n: n in p or p in n)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    soil = {
        k: (None if pd.isna(row[k]) else round(float(row[k]), 3))
        for k in df.columns
        if k not in ("name", "longitude", "latitude")
    }
    return {
        "matched": row["name"],
        "lon": float(row["longitude"]),
        "lat": float(row["latitude"]),
        "soil": {k: v for k, v in soil.items() if v is not None},
    }


@cache
def _fetch_power(lon: float, lat: float, start: str, end: str) -> dict | None:
    """拉 NASA POWER 日数据,返回各参数的期间均值。失败返回 None(不抛)。"""
    url = (
        f"{POWER_API}?parameters={POWER_PARAMS}&community=AG"
        f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)
        out = {}
        for k, series in data["properties"]["parameter"].items():
            vals = [v for v in series.values() if v is not None and v > -900]
            if vals:
                out[k] = sum(vals) / len(vals)
        return out or None
    except Exception as e:  # 网络/接口异常 → 降级,不影响预测流程
        logger.warning("NASA POWER 取数失败(%s),将用默认值兜底", type(e).__name__)
        return None


def get_soil(place: str) -> dict:
    """【土壤】地点 → 土壤特征(纯离线查表,零网络)。

    土壤属性年际变化极慢,本地 SoilGrids 与在线 API 实测同值(pH 7.9 vs 7.8),
    故本地优先:更快、更稳,且在线偶有覆盖空洞(返回 null)。
    """
    loc = resolve_location(place)
    if loc is None:
        return {"ok": False, "reason": f"未找到地点「{place}」(需为县/区级名称)"}
    return {
        "ok": True,
        "place": loc["matched"],
        "lon": loc["lon"],
        "lat": loc["lat"],
        "features": loc["soil"],
        "source": "SoilGrids 0-5cm 离线表(2898 县)",
    }


def get_climate(place: str, days: int = 90, end_date: str | None = None) -> dict:
    """【气候】地点 + 天数 → 气候特征(NASA POWER 在线;失败则降级)。

    days:用于把日均量换算成累计量(降水、UV —— 模型要的是累计)。
    end_date:'YYYYMMDD';默认取【去年同期结束】,保证 NASA 数据已就绪。
    """
    loc = resolve_location(place)
    if loc is None:
        return {"ok": False, "reason": f"未找到地点「{place}」(需为县/区级名称)"}

    end = (
        date.fromisoformat(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}")
        if end_date
        else (date.today().replace(year=date.today().year - 1))
    )
    start = end - timedelta(days=max(days, 1) - 1)
    power = _fetch_power(loc["lon"], loc["lat"], start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))

    if not power:
        return {
            "ok": False,
            "place": loc["matched"],
            "reason": "NASA POWER 取数失败(网络/接口异常),预测时将由默认值兜底",
        }

    f = {
        "Temperature_C": round(power.get("T2M", 0), 2),
        "solar_rad_MJm2d": round(power.get("ALLSKY_SFC_SW_DWN", 0), 3),
        # 累计量:日均 × 天数(模型训练时用的就是累计)
        "Precipitation_mm": round(power.get("PRECTOTCORR", 0) * days, 1),
    }
    rh = round(power.get("RH2M", 0), 2)
    f["Humidity"] = f["Air_Humidity_pct"] = rh
    if "ALLSKY_SFC_UVA" in power:
        f["UV"] = round(power["ALLSKY_SFC_UVA"] * UV_PER_UVA_MJ * days, 0)
    if "GWETTOP" in power:
        f["soil_moisture_pct"] = round(power["GWETTOP"] * 100, 1)

    return {
        "ok": True,
        "place": loc["matched"],
        "lon": loc["lon"],
        "lat": loc["lat"],
        "period": f"{start}~{end}({days}天)",
        "features": f,
        "source": "NASA POWER 在线(日尺度实况)",
    }


def get_environment(place: str, days: int = 90, end_date: str | None = None) -> dict:
    """【组合】土壤 + 气候 → 一份可直接喂模型的环境特征。供按地点预测使用。"""
    soil = get_soil(place)
    if not soil["ok"]:
        return soil
    climate = get_climate(place, days, end_date)

    features = dict(soil["features"])
    sources = {"土壤": soil["source"], "气候": climate.get("source", climate.get("reason", ""))}
    if climate["ok"]:
        features.update(climate["features"])

    return {
        "ok": True,
        "place": soil["place"],
        "lon": soil["lon"],
        "lat": soil["lat"],
        "period": climate.get("period", f"{days}天"),
        "features": features,
        "sources": sources,
    }
