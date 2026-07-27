"""配方批量试算(确定性):一次算完多个候选配方的三大性能,供权衡对比。

为什么做成"批量":让 subagent 逐个配方去调预测,十几个候选就要转十几圈 LLM 循环,
又慢又贵。这里一次把网格算完,返回一张对比表 —— subagent 只需看表做判断。
原则:确定性的部分榨干在工具里,只把【权衡取舍】留给 LLM。

⚠️ 已知局限:【厚度不能作为决策维度】(实测结论,必须如实披露)

  根因:原始数据混入了非薄膜样本 —— TS 里大量 1/2/4/5/6mm 的值是 ISO 527
  标准拉伸样条,不是膜;DR/WVTR 也有 1mm+ 的厚片。训练时已按 schema 的
  max_thickness_um 筛掉(WVTR ≤200µm、TS ≤500µm),整体 R² 因此提升
  (WVTR 0.849→0.878、TS 0.847→0.852)—— 但这【没有】让厚度变成可用维度:

  · 数据层面物理正确:WVTR 中位数随厚度单调下降(623→459→176→56→52),
    符合 Fick 定律(膜越厚越保墒)。
  · 但模型层面不可靠:固定其他条件、只变厚度时,预测【非单调】
    (10→413.7、50→444.7↑、120→324.6↓),且在实际地膜 8~15µm 区间内
    几乎无变化(DR/TS 差异 0.05,属噪声)。原因是薄膜样本稀疏 +
    厚度与其他特征在树模型里交互纠缠。

  → 故本模块把厚度作为【固定输入】,优化维度只用【配方比例】——
    该维度覆盖良好且物理正确(PBAT↑ → 降解↑、拉伸↓,与材料学一致,
    并有 tests/test_optimize.py 的物理闸门守着)。
"""

from __future__ import annotations

from ruixue_agent.predictors.predict import predict

# 全生物降解地膜以 PBAT 为主、PLA 增强;两端(纯PLA/纯PBAT)一并给出以看清取舍趋势
DEFAULT_PBAT_OPTIONS = (0, 40, 60, 70, 80, 90, 100)  # %;PLA = 100 - PBAT
DEFAULT_THICKNESS_UM = 10.0  # 国标常用标称厚度;见上方局限说明,不作为优化维度

# 模型对厚度有分辨力的下界(实测:低于此值预测不随厚度变化)
THICKNESS_RESOLUTION_UM = 20.0


def screen_recipes(
    env_features: dict,
    days: int,
    pbat_options: tuple = DEFAULT_PBAT_OPTIONS,
    thickness_um: float = DEFAULT_THICKNESS_UM,
) -> list[dict]:
    """对配方比例逐个预测三大性能,返回可直接对比的结果列表。

    env_features:该地环境特征(来自 environment.get_environment)。
    days:地膜使用天数(生育期长度)。thickness_um:固定厚度(非优化维度,见模块说明)。
    """
    rows = []
    for pbat in pbat_options:
        inputs = {
            **env_features,
            "Time_days": days,
            "PBAT_pct": pbat,
            "PLA_pct": 100 - pbat,
            "Thickness_um": thickness_um,
        }
        row = {"PBAT_pct": pbat, "PLA_pct": 100 - pbat}
        for name, key in [("DR", "降解率%"), ("WVTR", "透过率"), ("TS", "拉伸MPa")]:
            row[key] = predict(name, inputs)["prediction"]
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    """把试算结果排成一张易读的对比表(给 LLM 看,也给人看)。"""
    header = f"{'PBAT%':>7}{'PLA%':>7}{'降解率%':>11}{'透过率':>11}{'拉伸MPa':>11}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['PBAT_pct']:>7}{r['PLA_pct']:>7}"
            f"{r['降解率%']:>11}{r['透过率']:>11}{r['拉伸MPa']:>11}"
        )
    # 只标各指标的极值,不替用户判断"哪个最好"—— 那取决于作物与场景
    if rows:
        lines.append("")
        for key, desc, best_is_max in [
            ("降解率%", "降解最快", True),
            ("拉伸MPa", "最结实", True),
            ("透过率", "最保墒(透过率最低)", False),
        ]:
            best = (max if best_is_max else min)(rows, key=lambda r: r[key])
            lines.append(f"  {desc}: PBAT {best['PBAT_pct']}% → {best[key]}")
    return "\n".join(lines)
