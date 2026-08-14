"""配方试算 + 配方优化专家 测试。

试算依赖训练好的模型(models/predictors/),没训练则跳过。
"""

import pytest

import ruixue_agent.subagents as sub
from ruixue_agent.predictors.optimize import (
    DEFAULT_PBAT_OPTIONS,
    format_table,
    screen_recipes,
)
from ruixue_agent.predictors.registry import MODEL_STORE
from ruixue_agent.tools import get_tools

_HAS_MODELS = (MODEL_STORE / "DR" / "DR_model.joblib").exists()


# ── 专家注册(不依赖模型)────────────────────────────────────
def test_recipe_expert_registered_with_batch_tool():
    """配方优化专家应拿到批量试算工具,而不是逐个预测工具 —— 少转 LLM 循环。"""
    names = [t.name for t in sub._EXPERTS["配方优化专家"]["tools"]]
    assert "screen_film_recipes" in names
    assert "predict_degradation" not in names  # 给单项工具会导致逐个试、转很多圈


def test_expert_prompt_requires_honesty():
    """专家提示必须要求披露不确定性 —— 预测有误差,不能说得像定论。"""
    prompt = sub._EXPERTS["配方优化专家"]["prompt"]
    assert "不确定性" in prompt or "参考" in prompt


def test_screen_tool_registered_for_lead_agent():
    assert "screen_film_recipes" in [t.name for t in get_tools()]


# ── 批量试算(需模型)────────────────────────────────────────
@pytest.mark.skipif(not _HAS_MODELS, reason="预测模型未训练")
def test_screen_recipes_covers_all_options():
    rows = screen_recipes({}, days=150)
    assert len(rows) == len(DEFAULT_PBAT_OPTIONS)
    for r in rows:
        assert r["PBAT_pct"] + r["PLA_pct"] == 100  # 配方比例互补
        assert 0 <= r["降解率%"] <= 100
        assert r["拉伸MPa"] > 0


@pytest.mark.skipif(not _HAS_MODELS, reason="预测模型未训练")
def test_pbat_tradeoff_is_physically_correct():
    """物理正确性:PBAT 越高 → 越易降解、越不结实(材料学常识)。

    这是给整套预测系统的一道"物理合理性"闸门 —— 模型若学反了,这里立刻报警。
    """
    rows = {r["PBAT_pct"]: r for r in screen_recipes({}, days=150, pbat_options=(0, 100))}
    assert rows[100]["降解率%"] > rows[0]["降解率%"]  # 纯 PBAT 比纯 PLA 更易降解
    assert rows[100]["拉伸MPa"] < rows[0]["拉伸MPa"]  # 纯 PBAT 更柔软、强度更低


@pytest.mark.skipif(not _HAS_MODELS, reason="预测模型未训练")
def test_format_table_readable_and_flags_extremes():
    txt = format_table(screen_recipes({}, days=120))
    assert "PBAT%" in txt and "降解率%" in txt
    assert "降解最快" in txt and "最结实" in txt
