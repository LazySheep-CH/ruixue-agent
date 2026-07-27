"""配方批量试算(确定性):一次算完多个候选配方的三大性能,供权衡对比。

为什么做成"批量":让 subagent 逐个配方去调预测,十几个候选就要转十几圈 LLM 循环,
又慢又贵。这里一次把网格算完,返回一张对比表 —— subagent 只需看表做判断。
原则:确定性的部分榨干在工具里,只把【权衡取舍】留给 LLM。

⚠️ 已知局限(实测,必须如实披露给用户):
  训练数据的厚度中位数为 DR 167µm / WVTR 57µm / TS 616µm,而【实际农用地膜是
  8~15µm】(GB 0.008~0.015mm),三份数据里 <15µm 的样本仅占 1%/16%/5%。
  实测:5µm 与 12µm 的预测【完全相同】,要到 20µm 以上模型才有分辨力。
  → 厚度【不能】作为可优化维度(会给出"改厚度没用"的误导结论),故本模块
    只把厚度作为固定输入,优化维度聚焦【配方比例】(该维度覆盖良好且物理正确:
    PBAT↑ → 降解↑、拉伸↓,与材料学常识一致)。
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
