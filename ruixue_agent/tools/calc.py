"""计算类工具:用量、成本等确定性计算。

用量的主要驱动是厚度(近似线性),其次是材料类型(密度不同),
因此用公式而非查表,任意厚度都能算:
    每亩用量(kg) = 系数[类型] × 厚度(mm)
    总用量(kg)   = 面积(亩) × 每亩用量

系数由权威用量范围反推标定(详见 docs/数据来源.md):
    PE:       0.01mm ≈ 9 kg/亩  → 系数 900(0.008→7.2、0.015→13.5,
              落在国标 6-8 / 12-15 kg/亩区间内)
    生物降解: 0.01mm ≈ 10 kg/亩 → 系数 1000(PBAT/PLA 共混密度约 1.25,
              高于 PE 的 0.92,同厚度更重)
数据来源:GB13735-2017《聚乙烯吹塑农用地面覆盖薄膜》、
农业农村部生物降解地膜技术模式。
"""

from langchain_core.tools import tool

# 每亩用量系数:每亩用量(kg) = 系数 × 厚度(mm)
COVERAGE_COEFFICIENT: dict[str, float] = {
    "PE": 900.0,  # 普通聚乙烯地膜
    "生物降解": 1000.0,  # 全生物降解地膜(PBAT/PLA 等共混)
}

DEFAULT_THICKNESS_MM = 0.010  # 国标最小标称厚度,经济型常用值


@tool
def estimate_film_usage(
    area_mu: float, film_type: str = "PE", thickness_mm: float = DEFAULT_THICKNESS_MM
) -> str:
    """估算指定面积、类型、厚度的地膜用量(kg)。

    参数:
        area_mu: 覆膜面积,单位【亩】。
        film_type: 地膜类型,"PE"(普通聚乙烯) 或 "生物降解"(全生物降解),默认 PE。
        thickness_mm: 地膜厚度,单位【毫米】,默认 0.010(国标最小标称厚度)。
    返回:
        一段说明用量的中文文字,含面积、类型、厚度、总用量。
    """
    film_type = film_type.upper()  # 兼容 "pe" 等小写输入;中文 key 不受影响
    if film_type not in COVERAGE_COEFFICIENT.keys():
        raise ValueError("暂不支持该地膜用量的计算")
    if area_mu <= 0.0 or thickness_mm <= 0.0:
        raise ValueError("请输入正确的地膜面积和厚度")
    film_usage = COVERAGE_COEFFICIENT[film_type] * thickness_mm * area_mu
    return (
        f"面积为{area_mu}亩的种植区域下，{thickness_mm}毫米的{film_type}地膜，"
        f"总用量需要{film_usage:.1f}公斤"
    )


if __name__ == "__main__":
    print(estimate_film_usage.invoke({"area_mu": 10, "film_type": "PE"}))
    print(estimate_film_usage.invoke({"area_mu": 10, "film_type": "pe"}))
