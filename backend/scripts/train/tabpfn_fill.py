"""用 TabPFN 填充缺失值(整表含目标)—— 实测最强填充器(RMSE 比 MissForest 低 27%)。

    TABPFN_TOKEN=... HF_TOKEN=... uv run python scripts/train/tabpfn_fill.py TS

产出:data/predictors/<name>_tabpfn_filled.csv(离线一次性;慢,几分钟~十几分钟)。
需环境变量:TABPFN_TOKEN(PriorLabs)+ HF_TOKEN(HuggingFace 门禁权重)。
DR 零缺失,无需填充。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from ruixue_agent.predictors.schema import CATEGORICAL_DOMAINS, MODELS

ROOT = Path(__file__).resolve().parents[2]


def main(name: str) -> None:
    from tabpfn import TabPFNRegressor  # 延迟导入(要先设好 token 环境变量)

    cfg = MODELS[name]
    cols = cfg["features"] + [cfg["target"]]
    df = pd.read_csv(ROOT / "data" / "predictors" / cfg["data"], encoding="utf-8-sig")
    df = df[cols].apply(pd.to_numeric, errors="coerce")

    # Fertilizer 编码归一
    if cfg["fertilizer_map"]:
        df["Fertilizer"] = df["Fertilizer"].map(
            lambda v: cfg["fertilizer_map"].get(int(v)) if pd.notna(v) else np.nan
        )

    # 越界处理:TS 删物理不可能行;DR/WVTR 裁剪
    for col, (lo, hi) in cfg["ranges"].items():
        if col not in df.columns:
            continue
        if cfg["drop_policy"] == "drop":
            df = df[(df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi))]
        else:
            df[col] = df[col].clip(lo, hi)
    df = df.reset_index(drop=True)

    print(f"{name}: {len(df)} 行,TabPFN 填整表(含目标)...")
    t = time.time()
    filled = pd.DataFrame(
        IterativeImputer(estimator=TabPFNRegressor(), max_iter=3, random_state=42).fit_transform(
            df
        ),
        columns=cols,
    )
    # 1) 连续变量修正:插补器不认物理边界,会填出 PBAT=-45% 这种不可能值。
    #    按变量字典的合理范围裁剪回边界。
    clipped = {}
    for col, (lo, hi) in cfg["ranges"].items():
        if col in filled.columns:
            bad = int(((filled[col] < lo) | (filled[col] > hi)).sum())
            if bad:
                clipped[col] = bad
                filled[col] = filled[col].clip(lo, hi)

    # 2) 分类变量修正:填充器当连续量填出了小数(如 Color=2.4),四舍五入回整数 + 裁剪合法域。
    for col in cfg["categorical"]:
        if col in filled.columns and col in CATEGORICAL_DOMAINS:
            lo, hi = CATEGORICAL_DOMAINS[col]
            filled[col] = filled[col].round().clip(lo, hi).astype(int)

    out = ROOT / "data" / "predictors" / f"{name}_tabpfn_filled.csv"
    filled.to_csv(out, index=False)
    print(f"  完成 {time.time() - t:.0f}s → {out.name}")
    print(f"  连续变量越界裁剪: {clipped or '无'};分类列已修正为整数")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        sys.exit(f"用法: python scripts/train/tabpfn_fill.py [{'/'.join(MODELS)}]")
    main(sys.argv[1])
