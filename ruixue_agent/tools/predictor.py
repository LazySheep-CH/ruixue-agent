"""地膜性能预测工具:把三个机器学习模型包成 agent 能调的工具。

每个工具接收一个"已知参数字典"(用户/agent 从对话里能提取到的配方+环境),
未提供的参数由预测层用领域默认值兜底,返回带免责说明的预测结果。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from ruixue_agent.predictors.predict import predict_text


@tool
def predict_degradation(inputs: dict) -> str:
    """预测生物降解地膜的【降解率】(%,越高降解越多)。

    参数 inputs:已知参数的字典,键用特征名。常用:
      PLA_pct, PBAT_pct(配方比例%)、Thickness_um(厚度µm)、Time_days(埋后天数)、
      Temperature_C(环境温度)、Precipitation_mm(累计降水)、burial_depth_cm(埋深)、
      Soil_pH、soil_moisture_pct 等土壤参数。
    未提供的参数会用训练数据中位数兜底,结果会标注哪些用了默认值。
    示例:{"PLA_pct": 100, "Thickness_um": 10, "Time_days": 90, "Temperature_C": 25}
    """
    return predict_text("DR", inputs)


@tool
def predict_water_vapor_rate(inputs: dict) -> str:
    """预测地膜的【水蒸气透过率 WVTR】(g/m²·d,越低保墒越好)。

    参数 inputs:已知参数字典。常用:PLA_pct, PBAT_pct, Thickness_um(厚度)、
      Temperature_C、Humidity(湿度%)、Precipitation_mm、UV、Additive Type、Color 等。
    未提供的参数用默认值兜底,结果标注哪些用了默认。
    """
    return predict_text("WVTR", inputs)


@tool
def predict_tensile_strength(inputs: dict) -> str:
    """预测地膜的【拉伸强度】(MPa,越高越结实)。

    参数 inputs:已知参数字典。常用:PLA_pct, PBAT_pct(配方)、Thickness_um、
      Additive Type、Time_days、Temperature_C 等。
    未提供的参数用默认值兜底,结果标注哪些用了默认。
    """
    return predict_text("TS", inputs)


@tool
def predict_by_location(place: str, days: int, recipe: dict | None = None) -> str:
    """【推荐】按【地点】预测地膜三大性能:自动查该地土壤+气候,不用用户填环境参数。

    用户只需说地点、用多少天、什么配方,本工具自动补齐环境:
      土壤(pH/有机质/质地等)← 该县 SoilGrids 数据
      气候(温度/降水/辐射/UV/湿度)← NASA POWER 该地该时段实况
    参数:
      place:  县/区级或市级地名,如"尉犁县""新疆尉犁""南京"
      days:   地膜使用/埋设天数,如 90
      recipe: 配方等已知参数,如 {"PLA_pct": 30, "PBAT_pct": 70, "Thickness_um": 10}
    返回三个性能的预测 + 环境数据来源 + 哪些参数仍用了默认值。
    """
    from ruixue_agent.predictors.environment import get_environment

    env = get_environment(place, days)
    if not env["ok"]:
        return (
            f"{env['reason']}。请提供县/区级地名(如「尉犁县」),"
            "或直接用单项预测工具并自行给出环境参数。"
        )

    inputs = {**env["features"], "Time_days": days, **(recipe or {})}
    lines = [
        f"地点:{env['place']}({env['lon']:.2f}, {env['lat']:.2f}) · 时段 {env['period']}",
        f"环境数据来源:{env['sources']}",
        "",
    ]
    for name in ["DR", "WVTR", "TS"]:
        lines.append(predict_text(name, inputs))
        lines.append("")
    lines.append("注:环境为该地典型值,微生物等无地理数据源的参数用训练中位数估计,结果供参考。")
    return "\n".join(lines)


def get_predictor_tools() -> list[BaseTool]:
    """性能预测工具:三个单项 + 一个按地点自动补齐环境的综合工具。"""
    return [
        predict_by_location,
        predict_degradation,
        predict_water_vapor_rate,
        predict_tensile_strength,
    ]
