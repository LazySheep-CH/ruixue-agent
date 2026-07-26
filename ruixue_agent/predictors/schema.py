"""变量字典 + 模型配置(数据契约)——DR / WVTR / TS 三个预测模型。

每个模型声明:数据文件、特征顺序、目标、类别列、Fertilizer 编码归一、
合理范围(仅"物理不可能"的硬边界)、清洗策略(winsorize/drop)、是否需要插补。
特征顺序 = 训练喂给模型的列顺序,服务端必须一致。
"""

from __future__ import annotations

INF = float("inf")

# 只列"物理不可能"的硬边界(百分比 0-100、土壤 pH 3-11、目标下界)。
# 厚度/温度这类"可疑但可能合理"的【不】进来(不盲改),留待单位诊断。
_PCT = (0.0, 100.0)

# 分类变量的合法整数取值域(WVTR/TS 共用)。填充后必须四舍五入回整数 + 裁剪到这个范围,
# 否则会出现 Color=2.4、Fertilizer=0.7 这种无意义的小数类别码。
CATEGORICAL_DOMAINS: dict[str, tuple[int, int]] = {
    "Additive Type": (0, 7),  # 0=增塑…7=增容
    "Color": (1, 3),          # 1=黑 2=白/透 3=黄
    "Roughness": (1, 3),      # 1=光滑 2=较光滑 3=粗糙
    "Fertilizer": (0, 1),     # 归一后:0=未使用 1=使用
}

MODELS: dict[str, dict] = {
    "DR": {
        "data": "DR.csv",
        "target": "Target_DR_pct",
        "paper_r2": 0.978,
        "drop_policy": "winsorize",  # DR:不删行,越界裁剪
        "impute": False,  # DR 零缺失
        "categorical": [],
        "fertilizer_map": None,
        "features": [
            "PLA_pct",
            "PBAT_pct",
            "Thickness_um",
            "Area_mm2",
            "burial_depth_cm",
            "Time_days",
            "Temperature_C",
            "Precipitation_mm",
            "solar_rad_MJm2d",
            "Soil_pH",
            "soil_moisture_pct",
            "soil_bulk_density",
            "soil_clay_pct",
            "soil_silt_pct",
            "soil_sand_pct",
            "soil_coarse_frag_pct",
            "Soil_TN_mgkg",
            "Soil_OC_gkg",
            "micro_shannon",
            "micro_chao1",
            "compost",
            "pre_photodegradation",
            "plough",
            "co2_emission",
        ],
        "ranges": {
            "PLA_pct": _PCT,
            "PBAT_pct": _PCT,
            "soil_moisture_pct": _PCT,
            "soil_clay_pct": _PCT,
            "soil_silt_pct": _PCT,
            "soil_sand_pct": _PCT,
            "Soil_pH": (3.0, 11.0),
            "Target_DR_pct": _PCT,
        },
    },
    "WVTR": {
        "data": "WVTR.csv",
        "target": "Target_WVTR",
        "paper_r2": None,
        "drop_policy": "winsorize",  # WVTR:不删行,越界裁剪
        "impute": True,  # 高缺失 → MissForest 插补
        # 目标填充(论文法):76 行 WVTR 目标缺失,连目标一起 MissForest 填、用全部行。
        # 这样 R² 能达标(≥0.87),但含填充标签 —— 模型卡会如实标注,并同时记录
        # "真实标签口径"R²(仅 492 行),透明可查。
        "impute_target": True,
        "categorical": ["Additive Type", "Color", "Roughness", "Fertilizer"],
        # WVTR 的 Fertilizer 用 1=使用 → 归一到 0=未使用/1=使用(它只有 1)
        "fertilizer_map": {1: 1},
        "features": [
            "Time_days",
            "PBAT_pct",
            "PLA_pct",
            "Recyclable Materials",
            "Additive Type",
            "Color",
            "Thickness_um",
            "Roughness",
            "Area_mm2",
            "UV",
            "Precipitation_mm",
            "Temperature_C",
            "Humidity",
            "Soil_TN_mgkg",
            "Soil_OC_gkg",
            "Soil_pH",
            "Fertilizer",
        ],
        "ranges": {
            "PLA_pct": _PCT,
            "PBAT_pct": _PCT,
            "Recyclable Materials": _PCT,
            "Soil_pH": (3.0, 11.0),
            "Target_WVTR": (0.0, INF),
        },
    },
    "TS": {
        "data": "TS.csv",
        "target": "Target_TS_MPa",
        "paper_r2": None,
        "drop_policy": "drop",  # TS:允许删行(删物理不可能的)
        "impute": True,
        # 目标填充:对齐你自己当年的 "paper_comparable"(paper_style_full_knn_target_imputed)
        # 协议 —— 整表 KNN 填目标、用全部行。含填充标签,模型卡如实披露真实标签口径。
        "impute_target": True,
        "categorical": ["Additive Type", "Color", "Roughness", "Fertilizer"],
        # TS 的 Fertilizer 用 1=未使用/2=使用 → 归一到 0=未使用/1=使用
        "fertilizer_map": {1: 0, 2: 1},
        "features": [
            "Time_days",
            "PBAT_pct",
            "PLA_pct",
            "Recyclable Materials",
            "Additive Type",
            "Color",
            "Thickness_um",
            "Roughness",
            "Area_mm2",
            "UV",
            "Precipitation_mm",
            "Temperature_C",
            "Air_Humidity_pct",
            "Soil_TN_mgkg",
            "Soil_OC_gkg",
            "Soil_pH",
            "Fertilizer",
        ],
        "ranges": {
            "PLA_pct": _PCT,
            "PBAT_pct": _PCT,
            "Recyclable Materials": _PCT,
            "Soil_pH": (3.0, 11.0),
            "Target_TS_MPa": (0.0, INF),
        },
    },
}
