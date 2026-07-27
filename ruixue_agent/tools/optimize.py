"""配方试算工具:给地点+天数,一次算完多个候选配方的三大性能对比表。

确定性能力(网格 + 模型预测,无需推理)→ 是 tool。
"权衡哪个配方更合适"需要结合作物/场景判断 → 交给「配方优化专家」子 agent。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from ruixue_agent.predictors.environment import get_environment
from ruixue_agent.predictors.optimize import (
    DEFAULT_PBAT_OPTIONS,
    DEFAULT_THICKNESS_UM,
    format_table,
    screen_recipes,
)


@tool
def screen_film_recipes(place: str, days: int, thickness_um: float = DEFAULT_THICKNESS_UM) -> str:
    """批量试算不同 PBAT/PLA 配方在某地的表现,返回三大性能对比表(用于选配方)。

    自动查该地土壤+气候,再对一组配方比例逐个预测降解率、水蒸气透过率、拉伸强度。
    参数:
      place: 县/区级或市级地名,如"尉犁县""新疆尉犁"
      days:  地膜覆盖天数(取作物生育期,如棉花约 150 天、玉米约 120 天)
      thickness_um: 地膜厚度µm,默认 10(国标常用);注意模型对 20µm 以下厚度
                    无分辨力,故厚度不作为优化维度,详见返回文本中的局限说明。
    适用:用户问"该选什么配方""哪种膜适合这里"时,先用本工具拿到对比数据。
    """
    env = get_environment(place, days)
    if not env["ok"]:
        return env["reason"]
    rows = screen_recipes(env["features"], days, thickness_um=thickness_um)
    note = (
        f"局限:原始数据混入了非薄膜样本(ISO527 拉伸样条等,已在训练时按厚度筛除),"
        f"但薄膜样本仍稀疏 —— 模型对厚度的响应不单调、在 8~15µm 区间几乎无变化。"
        f"故本表固定厚度 {thickness_um:.0f}µm,【只比较配方差异,不要据此调厚度】。"
        f"厚度对性能的真实影响请参考国标与文献(数据层面:膜越厚越保墒,符合 Fick 定律)。"
    )
    return (
        f"{env['place']} · {env['period']} · 候选 PBAT {list(DEFAULT_PBAT_OPTIONS)}%"
        f"(PLA 为其余)· 厚度固定 {thickness_um:.0f}µm\n"
        f"环境来源:{env['sources']}\n\n"
        f"{format_table(rows)}\n\n"
        f"{note}\n"
        "另:未给出的参数(添加剂、颜色、微生物等)按训练数据中位数估计,结果供横向对比。"
    )


def get_optimize_tools() -> list[BaseTool]:
    return [screen_film_recipes]
