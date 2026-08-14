"""上传数据集的【数据契约】:列名归一、类型校验、必填判定。

为什么单独有这一层:
用户上传的是自己记的田间数据表。他不会按模型的特征名写列头 ——
模型要的是 `PLA_pct`,用户写的是「PLA含量」「PLA(%)」「pla」。
不做归一,这张表就只能整张退回;做了归一,大部分表当场可用。

和 `tools/calc.py` 的 `_normalize_film_type` 是同一条原则:宽进严出。
入口尽量认,真认不出再报错,并且明确告诉用户缺什么——
"格式不对"这种提示等于让用户自己猜。

必填的只有一样:实测值:
这张表的用途是拿实测去比模型预测。没有实测列,对比就不存在,
这次上传对用户毫无价值。所以它是唯一的硬性要求,而且在加载期就拦 ——
让用户在上传那一刻知道,而不是等 agent 分析到一半说"没数据"。

其余列全部可选:缺的用模型自带的 `feature_medians` 兜底(和
`predictors/predict.py` 走同一套兜底逻辑,不另开一套)。

别名表为什么手写而不是让模型猜:
列名归一如果交给 LLM,同一份表今天认出来明天认不出来,而且错了是静默的
(把「PBAT含量」认成 PLA,预测全错但不报错)。手写别名表是确定性的,
认不出就明说认不出 —— 能用查表解决的,不留给模型自觉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── 三个模型的实测目标列 ────────────────────────────────────────
# 键是模型名(和 predictors/registry.py 一致),值是这个目标的别名。
TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "DR": (
        "target_dr_pct",
        "dr",
        "降解率",
        "失重率",
        "质量损失率",
        "degradation",
        "degradationrate",
    ),
    "TS": ("target_ts_mpa", "ts", "拉伸强度", "抗拉强度", "断裂强度", "tensilestrength"),
    "WVTR": ("target_wvtr", "wvtr", "水蒸气透过率", "透湿量", "透水率", "watervaportransmission"),
}

# ── 特征列别名 ────────────────────────────────────────────────
# 只列【用户真的会填】的那些。24 个特征里像 micro_shannon(微生物多样性指数)
# 这种要专业测序才有的,用户不可能有,不必收 —— 它们走中位数兜底。
FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "PLA_pct": ("pla_pct", "pla", "pla含量", "pla比例", "pla占比", "聚乳酸含量"),
    "PBAT_pct": ("pbat_pct", "pbat", "pbat含量", "pbat比例", "pbat占比"),
    "Thickness_um": ("thickness_um", "thickness", "厚度", "膜厚", "厚度um", "厚度微米"),
    "Time_days": ("time_days", "days", "天数", "埋后天数", "覆膜天数", "铺膜天数", "时间"),
    "Temperature_C": ("temperature_c", "temperature", "温度", "气温", "平均温度", "均温"),
    "Precipitation_mm": ("precipitation_mm", "precipitation", "降水", "降雨", "降水量", "累计降水"),
    "Humidity": ("humidity", "湿度", "相对湿度", "空气湿度"),
    "Air_Humidity_pct": ("air_humidity_pct", "空气湿度百分比"),
    "Soil_pH": ("soil_ph", "ph", "土壤ph", "酸碱度"),
    "soil_moisture_pct": ("soil_moisture_pct", "土壤含水率", "土壤湿度", "含水率"),
    "Soil_OC_gkg": ("soil_oc_gkg", "有机碳", "土壤有机碳", "有机质"),
    "Soil_TN_mgkg": ("soil_tn_mgkg", "全氮", "土壤全氮"),
    "burial_depth_cm": ("burial_depth_cm", "埋深", "埋藏深度"),
    "solar_rad_MJm2d": ("solar_rad_mjm2d", "太阳辐射", "辐射", "日照辐射"),
    "UV": ("uv", "紫外", "紫外线", "紫外剂量"),
    "Color": ("color", "颜色", "膜色"),
    "Additive Type": ("additive type", "additive_type", "添加剂", "添加剂类型"),
}

# 一行一个地块/一次测量。超过这个数基本是传错文件(比如导出了整库),
# 而不是"数据特别多"—— 田间试验很少上千行。拦住它同时也是在保护内存。
MAX_ROWS = 5000
MAX_COLS = 80


def _norm(name: str) -> str:
    """列名归一:去空格/下划线/括号/单位后缀,转小写。

    「PLA 含量(%)」「PLA_含量」「pla含量」归一后都是 `pla含量`。
    """
    s = str(name).strip().lower()
    s = re.sub(r"[（(].*?[)）]", "", s)  # 去掉括号里的单位说明
    s = re.sub(r"[\s_\-/·]+", "", s)
    s = s.replace("%", "").replace("℃", "").replace("°c", "")
    return s


# 归一后的别名 → 标准名。构建一次,查表 O(1)。
_LOOKUP: dict[str, str] = {}
for std, aliases in FEATURE_ALIASES.items():
    _LOOKUP[_norm(std)] = std
    for a in aliases:
        _LOOKUP[_norm(a)] = std
_TARGET_LOOKUP: dict[str, str] = {}
for model, aliases in TARGET_ALIASES.items():
    for a in aliases:
        _TARGET_LOOKUP[_norm(a)] = model


@dataclass(frozen=True)
class ColumnMap:
    """一张表的列映射结果 —— 认出了什么、没认出什么,都要能说清楚。"""

    features: dict[str, str]  # 标准特征名 -> 原始列名
    targets: dict[str, str]  # 模型名(DR/TS/WVTR) -> 原始列名
    unknown: tuple[str, ...]  # 没认出来的列(保留原名,不丢弃也不猜)

    @property
    def has_target(self) -> bool:
        return bool(self.targets)


def map_columns(headers: list[str]) -> ColumnMap:
    """把用户的列头映射到标准名。认不出的原样留在 unknown 里。

    不猜:认不出就是认不出。把「PBAT含量」猜成 PLA 会让整张表的预测
    全错,而且不报错 —— 这类静默错误比直接拒绝坏得多。
    """
    features: dict[str, str] = {}
    targets: dict[str, str] = {}
    unknown: list[str] = []
    for h in headers:
        key = _norm(h)
        if model := _TARGET_LOOKUP.get(key):
            targets.setdefault(model, h)  # 同一目标多列时取第一列
        elif std := _LOOKUP.get(key):
            features.setdefault(std, h)
        else:
            unknown.append(h)
    return ColumnMap(features=features, targets=targets, unknown=tuple(unknown))


class DatasetError(ValueError):
    """数据不合契约。message 要直接可展示给用户 —— 说清缺什么、怎么改。"""


def validate(headers: list[str], n_rows: int) -> ColumnMap:
    """校验表头与规模;不合格抛 DatasetError(消息面向用户,不是面向开发者)。"""
    if not headers:
        raise DatasetError("这张表没有列头。请第一行写列名,例如:PLA含量、厚度、天数、降解率。")
    if len(headers) > MAX_COLS:
        raise DatasetError(f"列数 {len(headers)} 超过上限 {MAX_COLS},请只保留分析需要的列。")
    if n_rows == 0:
        raise DatasetError("这张表只有列头没有数据行。")
    if n_rows > MAX_ROWS:
        raise DatasetError(
            f"数据 {n_rows} 行超过上限 {MAX_ROWS} 行。田间试验通常几十到几百行,"
            "行数这么多通常是导出了整个数据库 —— 请先筛选出要分析的部分。"
        )

    cm = map_columns(headers)
    if not cm.has_target:
        # 唯一的硬性要求。在这里拦住,用户上传当场就知道;
        # 否则要等 agent 分析到一半才发现"没有可对比的实测值"。
        raise DatasetError(
            "没有找到实测值列。这张表要能和模型预测做对比,至少需要一列实测结果:\n"
            "  · 降解率(也可写「失重率」「质量损失率」)\n"
            "  · 拉伸强度(也可写「抗拉强度」)\n"
            "  · 水蒸气透过率(也可写「透湿量」「透水率」)\n"
            "请把对应列的表头改成上述名称之一后重新上传。"
        )
    return cm
