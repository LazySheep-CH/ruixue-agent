"""预测子系统测试:加载/schema校验/默认值兜底/工具注册。

模型是训练产物(models/predictors/,gitignore),没训练就跳过(CI 里跳过,不失败)。
"""

import pytest

from ruixue_agent.predictors.predict import predict, predict_text
from ruixue_agent.predictors.registry import MODEL_STORE, load_predictor
from ruixue_agent.tools import get_tools

_HAS_MODELS = (MODEL_STORE / "DR" / "DR_model.joblib").exists()
pytestmark = pytest.mark.skipif(
    not _HAS_MODELS, reason="预测模型未训练(models/predictors/ 为空);CI 跳过"
)


def test_load_and_schema_consistent():
    """加载即校验:模型期望特征数 == 模型卡 feature_order 长度。"""
    model, card = load_predictor("DR")
    assert model.n_features_in_ == len(card["feature_order"]) == 24
    assert card["target_unit"] == "%"
    assert len(card["feature_medians"]) == 24


def test_predict_full_inputs_in_valid_range():
    r = predict("DR", {"PLA_pct": 100, "Thickness_um": 10, "Time_days": 90, "Temperature_C": 25})
    assert 0 <= r["prediction"] <= 100  # 降解率是 %
    assert r["unit"] == "%"


def test_partial_inputs_fall_back_to_defaults():
    """只给部分参数,其余用默认;provided/defaulted 数量对得上。"""
    r = predict("WVTR", {"PLA_pct": 30, "PBAT_pct": 70})
    assert len(r["provided"]) == 2
    assert len(r["defaulted"]) == 17 - 2  # WVTR 共 17 特征


def test_case_insensitive_key_matching():
    """键容错:'pla pct' 应匹配到标准名 PLA_pct。"""
    r = predict("TS", {"pla pct": 50})
    assert "PLA_pct" in r["provided"]


def test_predict_text_has_disclosure():
    """文本输出必须带免责说明(哪些用了默认、结果仅供参考)。"""
    txt = predict_text("DR", {"PLA_pct": 100})
    assert "默认" in txt
    assert "仅供参考" in txt


def test_predictor_tools_registered():
    names = [t.name for t in get_tools()]
    for t in ["predict_degradation", "predict_water_vapor_rate", "predict_tensile_strength"]:
        assert t in names
