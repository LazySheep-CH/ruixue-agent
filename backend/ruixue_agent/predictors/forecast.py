"""天气预报:Open-Meteo,免 API key。

environment.get_climate 拿的是 NASA POWER 的历史实况(刻意取去年同期,
因为 POWER 数据有延迟),回答不了"下周怎么样"。这个模块补预报能力,
顺带补上了风速 —— 之前完全没有这个量,而风是新疆等地揭膜、撕裂的主要外力。

选 Open-Meteo:不用注册,辐射单位 MJ/m² 和 POWER 一致,可以直接喂预测模型。
不走 MCP:自己写自己用,套协议只多一跳网络。

约束:超时 20s、失败返回 None 不抛(和 _fetch_power 一致);输出必须标明
是预报不是实况,并带发布日期 —— 预报每天在变,不记发布时间就没法解释
"上次问的怎么不一样";结果不进冻结评测基线。
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import date

from ruixue_agent.predictors.environment import resolve_location

logger = logging.getLogger("ruixue.forecast")

FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# 和 _fetch_power 取同一个值:外部服务不可控,必须有上限。
FETCH_TIMEOUT_S = 20

# Open-Meteo 的预报上限是 16 天。超过这个天数它会直接报错,
# 所以在我们这层就夹住 —— 让用户看到"最多 16 天",而不是一个 400。
MAX_FORECAST_DAYS = 16

# 要哪些量:和地膜相关的五项。
#   温度/降水/辐射 —— 驱动降解
#   UV            —— 驱动光老化(新疆等强紫外区的主因)
#   风速          —— 揭膜、撕裂的主要外力,此前完全没有
_DAILY = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "shortwave_radiation_sum,uv_index_max,wind_speed_10m_max"
)

_LABELS = {
    "temperature_2m_max": "最高温(℃)",
    "temperature_2m_min": "最低温(℃)",
    "precipitation_sum": "降水(mm)",
    "shortwave_radiation_sum": "太阳辐射(MJ/m²)",
    "uv_index_max": "紫外指数",
    "wind_speed_10m_max": "最大风速(km/h)",
}


def _fetch(lat: float, lon: float, days: int) -> dict | None:
    """拉预报。失败返回 None(不抛)—— 和 _fetch_power 一致。"""
    qs = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "daily": _DAILY,
            "forecast_days": days,
            "timezone": "Asia/Shanghai",
        }
    )
    try:
        with urllib.request.urlopen(f"{FORECAST_API}?{qs}", timeout=FETCH_TIMEOUT_S) as r:
            return json.load(r)
    except Exception as e:
        logger.warning("Open-Meteo 取数失败(%s),本次无预报", type(e).__name__)
        return None


def get_forecast(place: str, days: int = 7) -> dict:
    """按地点取未来 N 天预报。

    返回 {"ok": bool, ...}。不抛异常 —— 调用方是工具层,
    异常会被中间件降级成一句"暂时不可用",不如在这里给出可读的原因。
    """
    # 非正数、None、传了个奇怪的东西 —— 一律回落到默认天数。
    #
    # 注意:原写法 `max(1, min(int(days or 7), MAX))` 对两种无意义输入的处理不一致:
    #   days=0  → `0 or 7` 短路成 7
    #   days=-3 → 不短路,夹成 1
    # 两者都是"模型给了个没意义的值",却一个给 7 天一个给 1 天。
    # 不一致本身不会报错,只会让行为难以解释 —— 统一成"没意义就用默认"。
    try:
        n = int(days)
    except (TypeError, ValueError):
        n = 0
    days = min(n, MAX_FORECAST_DAYS) if n > 0 else 7

    loc = resolve_location(place)
    if loc is None:
        return {"ok": False, "reason": f"未找到地点「{place}」(需为县/区级名称)"}

    data = _fetch(loc["lat"], loc["lon"], days)
    if not data or "daily" not in data:
        return {
            "ok": False,
            "place": loc["matched"],
            "reason": "天气预报服务暂时不可用",
        }

    daily = data["daily"]
    dates = daily.get("time") or []
    rows = [
        {"date": d, **{k: daily.get(k, [None] * len(dates))[i] for k in _LABELS}}
        for i, d in enumerate(dates)
    ]
    return {
        "ok": True,
        "place": loc["matched"],
        "days": len(rows),
        "rows": rows,
        "source": "Open-Meteo",
        # 取数当天。预报会随时间变,不记这个就没法解释"上次问的怎么不一样"。
        "issued_on": date.today().isoformat(),
    }


def format_forecast(res: dict) -> str:
    """渲染成给模型看的文本。开头就声明这是预报,不是实况。"""
    if not res.get("ok"):
        return f"取不到天气预报:{res.get('reason', '未知原因')}"

    head = (
        f"【天气预报 · 非实况】{res['place']} 未来 {res['days']} 天"
        f"(数据源 {res['source']},发布于 {res['issued_on']})\n"
        f"⚠ 预报存在不确定性,越往后越不准;临近作业请以当天最新预报为准。\n"
    )
    lines = []
    for r in res["rows"]:
        parts = [f"{_LABELS[k]} {r[k]}" for k in _LABELS if r.get(k) is not None]
        lines.append(f"  {r['date']}  " + "  ".join(parts))
    return head + "\n".join(lines)
