"""预测子系统 · 预测层:用户给已知参数,其余用领域默认值兜底,返回带免责的结果。

用户不可能填全 17~24 个特征(尤其环境/土壤)。这里:
  1. 把用户给的参数按特征名对上(容错大小写/下划线/空格);
  2. 没给的用模型卡里的 feature_medians(训练数据中位数)兜底;
  3. 按 feature_order 组装成带列名的 DataFrame,预测;
  4. 返回预测值 + 哪些用了默认(诚实免责:默认值可能与实际不符)。
"""

from __future__ import annotations

import pandas as pd

from ruixue_agent.predictors.registry import load_predictor

# 各模型的中文名(给 agent 回复用)
NAMES = {"DR": "降解率", "WVTR": "水蒸气透过率", "TS": "拉伸强度"}


def _norm(key: str) -> str:
    """归一化特征名用于容错匹配:小写、去空格/下划线。"""
    return key.lower().replace(" ", "").replace("_", "")


def predict(name: str, inputs: dict | None) -> dict:
    """预测 <name>;inputs 是 {特征名: 值} 的已知参数,其余用默认值。返回结果字典。"""
    model, card = load_predictor(name)
    features = card["feature_order"]
    defaults = card["feature_medians"]

    # 把用户输入的键容错匹配到标准特征名
    norm_to_feat = {_norm(f): f for f in features}
    given = {}
    for k, v in (inputs or {}).items():
        feat = norm_to_feat.get(_norm(str(k)))
        if feat is not None and v is not None:
            given[feat] = v

    # 组装一行:给了就用,没给用默认
    row, provided, defaulted = {}, [], []
    for f in features:
        if f in given:
            row[f] = given[f]
            provided.append(f)
        else:
            row[f] = defaults[f]
            defaulted.append(f)

    x = pd.DataFrame([row])[features]  # 按训练时的列顺序
    pred = float(model.predict(x)[0])
    return {
        "model": name,
        "target": NAMES.get(name, name),
        "prediction": round(pred, 2),
        "unit": card.get("target_unit", ""),
        "provided": provided,
        "defaulted": defaulted,
    }


def predict_text(name: str, inputs: dict | None) -> str:
    """预测并格式化成给 agent 用的中文文本(含免责说明)。"""
    r = predict(name, inputs)
    lines = [f"预测{r['target']}:{r['prediction']} {r['unit']}"]
    if r["provided"]:
        lines.append(f"你提供的参数({len(r['provided'])}):{', '.join(r['provided'])}")
    if r["defaulted"]:
        lines.append(
            f"用默认值兜底({len(r['defaulted'])}):{', '.join(r['defaulted'])}"
            " —— 这些参数你没给,用了训练数据中位数,结果为估计、仅供参考。"
        )
    return "\n".join(lines)
