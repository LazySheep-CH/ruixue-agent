"""离线抽取:2898 个县的土壤属性(SoilGrids 栅格)→ 一张静态表。

    uv run python scripts/env/extract_county_soil.py

为什么离线做:读 .tif 要 rasterio(重依赖)。抽成 CSV 后,服务端只需查表,
不装任何地理库、不联网 —— 大陆最稳。土壤属性年际变化极慢,一次抽取长期可用。

单位换算(由 DR 训练数据分布反推,见 docs 方案 §12):
    pH / 粘 / 粉 / 砂 / 有机碳  → 原始值 ÷ 10
    全氮 / 容重 / 粗碎屑        → 原始值(DR 论文未换算)
校验:粘+粉+砂(÷10)≈ 100%。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
TIF_DIR = Path(r"D:\soilgrids-app\data\china")
COUNTY = ROOT / "data" / "predictors" / "county_latlon.csv"
OUT = ROOT / "data" / "predictors" / "county_soil.csv"
DEPTH = "0-5cm"  # 表层:地膜降解主要发生在表层

# SoilGrids 属性 → (模型特征名, 除数)
PROPS = {
    "phh2o": ("Soil_pH", 10),
    "clay": ("soil_clay_pct", 10),
    "silt": ("soil_silt_pct", 10),
    "sand": ("soil_sand_pct", 10),
    "soc": ("Soil_OC_gkg", 10),
    "nitrogen": ("Soil_TN_mgkg", 1),
    "bdod": ("soil_bulk_density", 1),
    "cfvo": ("soil_coarse_frag_pct", 1),
}


def main() -> None:
    counties = pd.read_csv(COUNTY, encoding="utf-8-sig")
    pts = list(zip(counties["longitude"], counties["latitude"], strict=True))
    print(f"县级点位: {len(pts)}")

    out = counties[["name", "longitude", "latitude"]].copy()
    for prop, (feature, div) in PROPS.items():
        path = TIF_DIR / f"{prop}_{DEPTH}_mean.tif"
        with rasterio.open(path) as src:
            vals = np.array([v[0] for v in src.sample(pts)], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
        vals[vals <= 0] = np.nan  # 海域/无数据
        out[feature] = vals / div
        ok = np.isfinite(out[feature]).sum()
        print(
            f"  {prop:10} → {feature:22} 有效 {ok}/{len(pts)}  中位={np.nanmedian(out[feature]):.1f}"
        )

    # 质量校验:粘+粉+砂 应≈100%
    tex = out[["soil_clay_pct", "soil_silt_pct", "soil_sand_pct"]].sum(axis=1)
    print(f"\n校验 粘+粉+砂: 中位={tex.median():.1f}% (应≈100)")

    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"已保存: {OUT.name}  ({len(out)} 行)")


if __name__ == "__main__":
    main()
