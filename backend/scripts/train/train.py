"""通用训练脚本:按 schema.MODELS 配置训练 DR / WVTR / TS。

    uv run python scripts/train/train.py DR
    uv run python scripts/train/train.py WVTR
    uv run python scripts/train/train.py TS

企业级:固定种子、K 折 CV、数据血缘(sha256)、模型卡、训练/服务分离。
按数据策略:DR/WVTR 不删行(越界裁剪)、TS 可删行;所有模型不删列;
高缺失(WVTR/TS)用 MissForest(IterativeImputer+ExtraTrees)在管道内插补(无泄露)。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ruixue_agent.predictors.schema import MODELS

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N_SPLITS = 5
TARGET_UNITS = {"DR": "%", "WVTR": "g/m²·d", "TS": "MPa"}  # 各模型目标的单位


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def candidates() -> dict:
    """5 个强回归器(都固定种子);内部 n_jobs=-1,外层 CV 用 n_jobs=1 避免过度并行。"""
    return {
        "XGBoost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(random_state=SEED),
    }


def make_imputer() -> IterativeImputer:
    """MissForest 式插补:用 ExtraTrees 迭代预测缺失值;add_indicator 附"是否缺失"标志位。"""
    return IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=20, random_state=SEED, n_jobs=-1),
        max_iter=5,
        add_indicator=True,
        random_state=SEED,
    )


# 插补很贵;5 个候选算法共用同一个插补器 → 用 joblib 缓存,让插补每折只 fit 一次,
# 之后 4 个算法直接复用缓存结果(约 5 倍加速),TS 这种大数据才跑得动。
_MEMORY = joblib.Memory(location=str(ROOT / "models" / ".impute_cache"), verbose=0)


def build(model, impute: bool):
    if not impute:
        return model
    return Pipeline([("impute", make_imputer()), ("model", model)], memory=_MEMORY)


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """按配置:Fertilizer 编码归一 → 丢无标签行 → 越界处理(裁剪或删行)。不删列。"""
    report = {"rows_in": len(df)}
    df = df.copy()

    # 0. 统一转数值:非数值单元格(如 TS 的 Roughness 里混入的空格 ' ')→ NaN,
    #    否则整列变字符串型、插补器报错。转成 NaN 后由插补统一处理。
    df = df.apply(pd.to_numeric, errors="coerce")

    # 1. Fertilizer 编码归一到 0=未使用/1=使用
    fmap = cfg["fertilizer_map"]
    if fmap and "Fertilizer" in df.columns:
        df["Fertilizer"] = df["Fertilizer"].map(
            lambda v: fmap.get(int(v)) if pd.notna(v) else np.nan
        )

    # 2. 丢弃目标缺失的行 —— 无标签无法监督训练(这不是"删好数据",是没法学)。
    #    但 impute_target 模式(论文法)保留这些行,稍后连目标一起填(见 main)。
    if not cfg.get("impute_target"):
        n = len(df)
        df = df[df[cfg["target"]].notna()]
        report["dropped_no_label"] = n - len(df)

    # 3. 越界处理:winsorize(DR/WVTR 裁剪不删行)或 drop(TS 删物理不可能行)
    ranges, policy = cfg["ranges"], cfg["drop_policy"]
    if policy == "winsorize":
        clipped = {}
        for col, (lo, hi) in ranges.items():
            if col not in df.columns:
                continue
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.any():
                clipped[col] = int(bad.sum())
                df[col] = df[col].clip(lo, hi)
        report["policy"], report["clipped"] = "winsorize", clipped
    else:
        n = len(df)
        mask = pd.Series(True, index=df.index)
        dropped = {}
        for col, (lo, hi) in ranges.items():
            if col not in df.columns:
                continue
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.any():
                dropped[col] = int(bad.sum())
                mask &= ~bad
        df = df[mask]
        report["policy"], report["dropped_impossible"], report["dropped_by_col"] = (
            "drop",
            n - len(df),
            dropped,
        )

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def main(name: str) -> None:
    cfg = MODELS[name]
    out_dir = ROOT / "models" / "predictors" / name
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    disclosure = None
    tabpfn_filled = ROOT / "data" / "predictors" / f"{name}_tabpfn_filled.csv"

    if tabpfn_filled.exists():
        # 首选:TabPFN 填充数据(实测最强填充器,RMSE 比 KNN 低约40%)。已清洗+填满。
        print(f"═══ 训练 {name}  (TabPFN 填充数据 {tabpfn_filled.name}) ═══")
        data_file = tabpfn_filled  # 修复:两个分支都要定义,否则写模型卡时 data_file 未定义
        df = pd.read_csv(tabpfn_filled)
        report = {"source": tabpfn_filled.name, "rows": len(df), "fill": "TabPFN"}
        disclosure = {
            "fill_method": "TabPFN(IterativeImputer);实测 RMSE 比 KNN 低约40%、比 MissForest 低27%",
            "note": (
                "WVTR/TS 目标也由 TabPFN 填(含填充标签);真实标签严格口径约 "
                "WVTR 0.80 / TS 0.73(见 docs 与 SHAP 报告)。DR 零缺失、无填充。"
            ),
        }
        impute_now = False
    else:
        # 回退:原始数据 → 清洗 →(WVTR/TS)KNN 目标填充(论文法,双口径披露)。
        data_file = ROOT / "data" / "predictors" / cfg["data"]
        print(f"═══ 训练 {name}  ({data_file.name}) ═══")
        df = pd.read_csv(data_file, encoding="utf-8-sig")[cfg["features"] + [cfg["target"]]]
        df, report = clean(df, cfg)
        if cfg.get("impute_target") and df[cfg["target"]].isna().any():
            real = df[df[cfg["target"]].notna()]
            real_r2 = cross_val_score(
                build(ExtraTreesRegressor(n_estimators=400, random_state=SEED), True),
                real[cfg["features"]],
                real[cfg["target"]],
                cv=cv,
                scoring="r2",
            ).mean()
            filled = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df), columns=df.columns)
            disclosure = {
                "target_imputed_rows": int(len(df) - len(real)),
                "real_label_rows": int(len(real)),
                "real_label_only_cv_r2": round(float(real_r2), 4),
                "target_fill_method": "KNNImputer(k=5),同论文",
                "note": "主 CV R² 含填充目标(论文法),偏乐观;真实标签口径见 real_label_only_cv_r2",
            }
            df = filled
        impute_now = cfg["impute"] and disclosure is None
    # 厚度筛选:原始数据混入了厚片/ISO527 拉伸样条(1~6mm),与地膜不是一类样本。
    # 按模型配置的上限剔除(实证:WVTR 筛后 R² 与薄膜分辨力双升;DR 无效故不筛)。
    max_t = cfg.get("max_thickness_um")
    if max_t and "Thickness_um" in df.columns:
        n0 = len(df)
        df = df[df["Thickness_um"] <= max_t].reset_index(drop=True)
        report["thickness_filter"] = f"≤{max_t:.0f}µm,{n0}→{len(df)} 行"

    X, y = df[cfg["features"]], df[cfg["target"]]
    miss_pct = 100 * X.isna().sum().sum() / (X.shape[0] * X.shape[1])
    print(f"清洗: {report}")
    if disclosure and "target_imputed_rows" in disclosure:
        print(
            f"目标填充(论文法): 全部 {len(X)} 行(填了 {disclosure['target_imputed_rows']} 行目标);"
            f"真实标签严格口径 R²={disclosure['real_label_only_cv_r2']}"
        )
    elif disclosure:
        print(f"填充: {disclosure['fill_method']}")
    print(
        f"训练集: {len(X)} 行 × {len(cfg['features'])} 特征;缺失率 {miss_pct:.1f}%"
        f"{' → 管道内 MissForest 插补' if impute_now else '(已填满/无缺失)'}"
    )

    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    print(f"\n{'算法':<16}{'CV R²':<10}{'CV RMSE':<10}{'CV MAE':<10}")
    print("─" * 46)
    results = {}
    for cn, model in candidates().items():
        cvr = cross_validate(build(model, impute_now), X, y, cv=cv, scoring=scoring, n_jobs=1)
        r2, rmse, mae = cvr["test_r2"].mean(), -cvr["test_rmse"].mean(), -cvr["test_mae"].mean()
        results[cn] = {"cv_r2": float(r2), "cv_rmse": float(rmse), "cv_mae": float(mae)}
        print(f"{cn:<16}{r2:<10.4f}{rmse:<10.3f}{mae:<10.3f}")

    best_name = max(results, key=lambda k: results[k]["cv_r2"])
    print(
        f"\n最优: {best_name}  CV R²={results[best_name]['cv_r2']:.4f}"
        + (f"  (论文单split R²={cfg['paper_r2']})" if cfg["paper_r2"] else "")
    )

    final = build(candidates()[best_name], impute_now).fit(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_dir / f"{name}_model.joblib")
    card = {
        "schema_version": 1,
        "name": name,
        "target": cfg["target"],
        "algorithm": best_name,
        "impute": cfg["impute"],
        "feature_order": cfg["features"],
        "target_unit": TARGET_UNITS[name],
        # 各特征默认值(训练数据中位数)—— 服务端用户没给的环境/土壤参数用它兜底。
        "feature_medians": {k: round(float(v), 4) for k, v in X.median().items()},
        "metrics": {**results[best_name], "cv_folds": N_SPLITS},
        "disclosure": disclosure,  # 目标填充披露(透明);无则 None
        "benchmark": results,
        "paper_r2_single_split": cfg["paper_r2"],
        "data": {
            "file": data_file.name,
            "sha256": sha256(data_file),
            "n_rows": len(X),
            "clean_report": report,
        },
        "env": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
    }
    (out_dir / f"{name}_model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存: {out_dir}/{name}_model.joblib  +  模型卡\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        sys.exit(f"用法: python scripts/train/train.py [{'/'.join(MODELS)}]")
    main(sys.argv[1])
