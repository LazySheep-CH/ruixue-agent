"""预测子系统 · 加载层:加载"模型 + 模型卡",懒加载单例,加载时校验 schema。

服务端只【加载】训练产物,不训练(训练/服务分离)。lru_cache 保证每个模型只加载一次
(像 checkpointer),之后所有预测复用。加载时校验模型期望特征数 == 模型卡 feature_order,
不一致就 fail fast(根治"模型和卡对不上"的隐患)。
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import joblib

MODEL_STORE = Path(__file__).resolve().parents[2] / "models" / "predictors"


@cache
def load_predictor(name: str) -> tuple:
    """加载 <name> 的模型 + 模型卡(懒加载单例);校验 schema 一致,返回 (model, card)。"""
    d = MODEL_STORE / name
    model_path = d / f"{name}_model.joblib"
    card_path = d / f"{name}_model_card.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"预测模型 {name} 未找到:{model_path}。请先训练:"
            f"uv run python scripts/train/train.py {name}"
        )
    model = joblib.load(model_path)
    card = json.loads(card_path.read_text(encoding="utf-8"))

    # schema 校验:模型内部期望的特征数,必须等于模型卡里声明的特征数。
    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is not None and n_expected != len(card["feature_order"]):
        raise ValueError(
            f"{name} schema 不一致:模型需要 {n_expected} 个特征,"
            f"但模型卡 feature_order 有 {len(card['feature_order'])} 个。"
        )
    return model, card
