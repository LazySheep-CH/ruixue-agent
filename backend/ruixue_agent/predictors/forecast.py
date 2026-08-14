"""天气【预报】:Open-Meteo(免 API key)。

## 为什么补这一路

`environment.get_climate` 拿的是 NASA POWER 的**历史实况**,而且刻意取
【去年同期】—— 因为 POWER 的数据有延迟,近期的取不到。它回答的是
"这地方常年什么气候",**不能回答"下周怎么样"**。

评测集 rf01(「2027年6月15号会下雨吗」)当时的正确行为是拒答,
理由正是"系统没有预报能力"。这个模块补的就是这块。

## 为什么是 Open-Meteo,而且是直连不是 MCP

**免 API key**:不需要用户去注册、不需要在生产环境管一个密钥。
这和 NASA POWER 的性质一样,不引入新的凭据面。

**单位对得上**:`shortwave_radiation_sum` 是 MJ/m²,和 POWER 的
`ALLSKY_SFC_SW_DWN` 同量纲,两路数据可以放在一起比,也能直接喂预测模型。

**它给风速** —— 这是我们此前完全没有的量。故障诊断专家上一轮的原话是
"工具未返回风速数据,大风揭膜无法在此验证,需用户自查";风是新疆等地
**揭膜、撕裂的主要外力**,缺了它诊断只能停在猜测。

**不走 MCP**:MCP 的价值在跨组织复用,而这里我们既是调用方也是实现方,
套一层协议只多一个进程和一次网络跳。判据见 `mcp/client.py` 的说明。

## 三条纪律

1. **超时 + 失败降级**:外部服务不可控,和 `_fetch_power` 同一处理 ——
   失败返回 None、记日志,绝不让一次网络抖动把整个回答搞砸。
2. **必须标明这是预报**:预报会变、会错。答案里不区分"实况"和"预报",
   用户就会把一个概率当成事实 —— 和"资料必须带年份"是同一条原则。
3. **不进冻结评测的主基线**:预报结果每天都不同,进了基线就等于给
   噪声地板引入一个不可控源(我们花了很大力气才把它压到 0%)。
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

    返回 {"ok": bool, ...}。**不抛异常** —— 调用方是工具层,
    异常会被中间件降级成一句"暂时不可用",不如在这里给出可读的原因。
    """
    # 非正数、None、传了个奇怪的东西 —— 一律回落到默认天数。
    #
    # ⚠ 原写法 `max(1, min(int(days or 7), MAX))` 对两种无意义输入的处理不一致:
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
    """渲染成给模型看的文本。**开头就声明这是预报**,不是实况。"""
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
