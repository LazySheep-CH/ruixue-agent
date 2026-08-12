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


def _normalize_film_type(raw: str) -> str | None:
    """把各种说法归一到系数表的 key;认不出返回 None。

    ## 为什么需要这层归一

    系数表的 key 是 "PE" / "生物降解",但用户和模型不会照着念。实测:
        "全生物降解" ✗   "生物降解膜" ✗   "PBAT" ✗
    ——而"全生物降解"恰恰是最自然的说法(docstring 里都写着这四个字)。
    评测 mt02 问的就是"全生物降解膜",它过了只是因为模型碰巧归一成了
    "生物降解" —— 靠运气,不靠设计。

    和 memory 的 _strip_fence 同一个原则:**宽进严出**。入口尽量认,
    真认不出再报错(交给 ToolErrorHandlingMiddleware 降级)。

    ⚠ 顺序不能反:必须先判 PE 类,再判"降解" ——
      否则"不可降解地膜"会因为含"降解"二字被错判成生物降解膜,
      系数差 11%,而且错得毫无征兆。
    """
    s = (raw or "").strip().upper()
    if s in COVERAGE_COEFFICIENT:
        return s
    if any(k in s for k in ("PE", "聚乙烯", "不可降解", "非降解", "普通")):
        return "PE"
    if any(k in s for k in ("降解", "PBAT", "PLA")):
        return "生物降解"
    return None


@tool
def estimate_film_usage(
    area_mu: float, film_type: str = "PE", thickness_mm: float = DEFAULT_THICKNESS_MM
) -> str:
    """估算指定面积、类型、厚度的地膜用量(kg)。

    参数:
        area_mu: 覆膜面积,单位【亩】。
        film_type: 地膜类型,"PE"(普通聚乙烯) 或 "生物降解"(全生物降解),默认 PE。
                   也接受"全生物降解""PBAT"等常见说法。
        thickness_mm: 地膜厚度,单位【毫米】,默认 0.010(国标最小标称厚度)。
    返回:
        一段说明用量的中文文字,含面积、类型、厚度、总用量。
    """
    # 归一化各种说法(见 _normalize_film_type);认不出才报错
    kind = _normalize_film_type(film_type)
    if kind is None:
        raise ValueError(f"暂不支持的地膜类型「{film_type}」,可用:PE / 生物降解")
    if area_mu <= 0.0 or thickness_mm <= 0.0:
        raise ValueError("请输入正确的地膜面积和厚度")
    film_type = kind
    film_usage = COVERAGE_COEFFICIENT[film_type] * thickness_mm * area_mu
    return (
        f"面积为{area_mu}亩的种植区域下，{thickness_mm}毫米的{film_type}地膜，"
        f"总用量需要{film_usage:.1f}公斤"
    )


if __name__ == "__main__":
    print(estimate_film_usage.invoke({"area_mu": 10, "film_type": "PE"}))
    print(estimate_film_usage.invoke({"area_mu": 10, "film_type": "pe"}))
