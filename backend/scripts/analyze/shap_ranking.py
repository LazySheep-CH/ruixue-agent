"""SHAP 因子排名:对每个模型算 SHAP 特征重要度,和论文的主因子对照。

    uv run python scripts/analyze/shap_ranking.py

用论文同款模型(XGBoost)在填充后的数据上训练,再用 SHAP(TreeExplainer)算每个
因素对预测的平均贡献(mean|SHAP|),排序。对照论文已报的主因子看是否一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor

from ruixue_agent.predictors.schema import MODELS

ROOT = Path(__file__).resolve().parents[2]

# 论文已报的主因子(用于对照;TS 无论文)
PAPER_TOP = {
    "DR": ["Temperature_C", "Time_days", "PLA_pct", "PBAT_pct", "Precipitation_mm"],
    "WVTR": ["Precipitation_mm", "UV", "Thickness_um", "Area_mm2"],
    "TS": None,
}


def prep(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """按各模型策略加载→数值化→Fertilizer 归一→(TS)删不可能→目标填充,返回 X,y。

    优先用 TabPFN 填充好的数据(<name>_tabpfn_filled.csv,最强填充器);没有则回退。
    """
    cfg = MODELS[name]
    filled = ROOT / "data" / "predictors" / f"{name}_tabpfn_filled.csv"
    if filled.exists():
        d = pd.read_csv(filled)
        return d[cfg["features"]], d[cfg["target"]]

    df = pd.read_csv(ROOT / "data" / "predictors" / cfg["data"], encoding="utf-8-sig")
    df = df[cfg["features"] + [cfg["target"]]].apply(pd.to_numeric, errors="coerce")
    if cfg["fertilizer_map"]:
        df["Fertilizer"] = df["Fertilizer"].map(
            lambda v: cfg["fertilizer_map"].get(int(v)) if pd.notna(v) else np.nan
        )
    # TS 删物理不可能(pH),DR/WVTR 裁剪
    for col, (lo, hi) in cfg["ranges"].items():
        if col not in df.columns:
            continue
        if cfg["drop_policy"] == "drop":
            df = df[(df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi))]
        else:
            df[col] = df[col].clip(lo, hi)
    if cfg.get("impute_target"):  # 整表 KNN 填(含目标),用全部行
        df = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df), columns=df.columns)
    else:
        df = df[df[cfg["target"]].notna()]
        df[cfg["features"]] = KNNImputer(n_neighbors=5).fit_transform(df[cfg["features"]])
    return df[cfg["features"]], df[cfg["target"]]


def main() -> None:
    for name in ["DR", "WVTR", "TS"]:
        X, y = prep(name)
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        sv = shap.TreeExplainer(model).shap_values(X)
        importance = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(
            ascending=False
        )

        print(f"\n═══ {name}  SHAP 因子排名(前 8)═══")
        for i, (feat, val) in enumerate(importance.head(8).items(), 1):
            print(f"  {i}. {feat:<22} {val:.3f}")
        paper = PAPER_TOP[name]
        if paper:
            top5 = list(importance.head(5).index)
            hit = [f for f in paper if f in top5]
            print(f"  论文主因子: {paper}")
            print(f"  我们 top5 命中: {hit}  ({len(hit)}/{len(paper)})")


if __name__ == "__main__":
    main()
