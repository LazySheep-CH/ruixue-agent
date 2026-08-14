"""对已上传数据集的分析:概览、与模型预测对比、异常检测、国标符合性。

为什么是【预置工具】而不是【给 agent 代码执行】:
"让 agent 自己写 pandas 跑"能力更强,但那需要一个真沙箱 ——
没有沙箱就等于让模型在服务器上执行任意代码,提示注入直接升级成远程代码执行。
沙箱(容器/gVisor)、文件生命周期、资源限额、超时、结果回传,是一个月的工程。

预置工具的能力有边界,但边界内覆盖了绝大多数真实需求,而且工具仍然只读:
它们只读 PG 里那份数据,不写文件、不执行用户提供的表达式。

agent 依然是"自主规划"的 —— 它决定先看什么、再看什么、怎么解读,
只是不写代码而是调这几个工具。能用受限接口做到的,不开放通用能力。

统计口径的两条纪律:
1) 缺失不当零。 缺失值参与均值会把结论拉偏,而且不报错。
所有统计都只在非缺失样本上算,并同时报出样本量 ——
"3 行数据算出来的均值"和"300 行算出来的"分量完全不同,不报 n 等于隐瞒。

2) 对比要报方向,不只报大小。 「偏差 8%」没有信息量;
「实测比预测低 8%」才能指向原因(降解慢了 → 可能是配方/温度/微生物)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ruixue_agent.analysis.store import Dataset

# 目标列在行里的键前缀(见 loader.load_csv)
_T = "target:"

# 各目标的中文名与单位,报告里要带上 —— 光给数字用户不知道是什么。
TARGET_INFO = {
    "DR": ("降解率", "%"),
    "TS": ("拉伸强度", "MPa"),
    "WVTR": ("水蒸气透过率", "g/m²·d"),
}


def _vals(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def _stats(vals: list[float]) -> dict:
    """基础统计。样本量必须一起返回 —— 见模块说明的纪律 1)。"""
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = sum(vals) / n
    # 样本标准差(n-1);n=1 时无从谈离散,记 0 而不是崩
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    ordered = sorted(vals)
    return {
        "n": n,
        "mean": round(mean, 3),
        "std": round(math.sqrt(var), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "median": round(
            ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2, 3
        ),
    }


# ── 1) 概览 ────────────────────────────────────────────────────


def describe(ds: Dataset) -> str:
    """这张表里有什么、数据质量如何。分析的第一步永远是先看数据本身。"""
    lines = [f"数据集《{ds.filename}》:{ds.n_rows} 行"]

    if ds.targets:
        lines.append("\n实测指标:")
        for model, orig in sorted(ds.targets.items()):
            name, unit = TARGET_INFO.get(model, (model, ""))
            s = _stats(_vals(ds.rows, f"{_T}{model}"))
            if s["n"] == 0:
                lines.append(f"  · {name}(原列名「{orig}」):整列都是空的,无法分析")
                continue
            miss = ds.n_rows - s["n"]
            lines.append(
                f"  · {name} {unit}:n={s['n']}"
                + (f"(缺 {miss} 行)" if miss else "")
                + f"  均值 {s['mean']}  中位 {s['median']}"
                + f"  范围 {s['min']}~{s['max']}  标准差 {s['std']}"
            )

    if ds.features:
        lines.append("\n可用于预测的条件列:")
        for std, orig in sorted(ds.features.items()):
            s = _stats(_vals(ds.rows, std))
            miss_pct = round(100 * (1 - s["n"] / ds.n_rows), 1) if ds.n_rows else 100.0
            flag = "  ⚠ 缺失过半,该列基本无效" if miss_pct > 50 else ""
            lines.append(
                f"  · {std}(原列名「{orig}」):缺失 {miss_pct}%"
                + (f"  范围 {s['min']}~{s['max']}" if s["n"] else "")
                + flag
            )

    unknown = ds.columns.get("unknown") or []
    if unknown:
        lines.append(
            "\n未识别的列(不参与分析):" + "、".join(unknown) + "\n"
            "  如果其中有需要用到的指标,请把列名改成标准写法后重新上传。"
        )
    return "\n".join(lines)


# ── 2) 与模型预测对比 ──────────────────────────────────────────


@dataclass
class RowDiff:
    idx: int
    actual: float
    predicted: float

    @property
    def diff(self) -> float:
        return self.actual - self.predicted

    @property
    def rel_pct(self) -> float | None:
        """相对偏差 %。预测值为 0 时无意义,返回 None —— 不返回 inf。"""
        return round(100 * self.diff / self.predicted, 1) if self.predicted else None


def compare_with_model(ds: Dataset, model: str) -> str:
    """逐行拿实测比模型预测,给出偏差方向与幅度。

    这是整个上传功能的核心价值:别人做不了这件事,因为别人没有这三个
    自训预测模型。用户传一张田间记录,就能知道"我这块地和模型的认知差多少"。
    """
    from ruixue_agent.predictors.predict import predict

    name, unit = TARGET_INFO.get(model, (model, ""))
    tkey = f"{_T}{model}"
    usable = [(i, r) for i, r in enumerate(ds.rows, 1) if r.get(tkey) is not None]
    if not usable:
        return f"这张表里没有可用的{name}实测值,无法对比。"

    diffs: list[RowDiff] = []
    failed = 0
    for i, r in usable:
        feats = {k: v for k, v in r.items() if not k.startswith(_T) and v is not None}
        try:
            pred = predict(model, feats)["prediction"]
        except Exception:
            failed += 1
            continue
        diffs.append(RowDiff(idx=i, actual=r[tkey], predicted=float(pred)))

    if not diffs:
        return f"{name}预测全部失败,无法对比(可能是条件列缺失过多)。"

    rel = [d.rel_pct for d in diffs if d.rel_pct is not None]
    mae = sum(abs(d.diff) for d in diffs) / len(diffs)
    # 带符号的平均偏差:方向比幅度更有指向性(见模块说明纪律 2))
    bias = sum(d.diff for d in diffs) / len(diffs)
    direction = "高于" if bias > 0 else "低于"

    lines = [
        f"{name}实测 vs 模型预测(n={len(diffs)}{f',{failed} 行预测失败' if failed else ''}):",
        f"  平均绝对偏差 MAE = {mae:.2f} {unit}",
        f"  平均偏差 = {bias:+.2f} {unit} —— 实测整体**{direction}**模型预测",
    ]
    if rel:
        lines.append(f"  平均相对偏差 = {sum(rel) / len(rel):+.1f}%")

    worst = sorted(diffs, key=lambda d: -abs(d.diff))[:5]
    lines.append("  偏差最大的几行:")
    for d in worst:
        pct = f"({d.rel_pct:+.1f}%)" if d.rel_pct is not None else ""
        lines.append(
            f"    第 {d.idx} 行:实测 {d.actual} / 预测 {d.predicted:.2f} → {d.diff:+.2f} {pct}"
        )

    lines.append(
        "\n解读提示:模型是在公开文献数据上训练的,系统性偏差通常来自"
        "【当地条件与训练数据分布不同】(如灌溉方式、覆膜工艺),不一定是测量错误。"
        "单行的大偏差则要先排查记录是否有误。"
    )
    return "\n".join(lines)


# ── 3) 异常值 ──────────────────────────────────────────────────


# 修正 z 分数的判定阈值。3.5 是 Iglewicz & Hoaglin (1993) 的推荐值。
OUTLIER_Z = 3.5
# 正态分布下 MAD ≈ 0.6745σ,乘这个常数把 MAD 折算回"相当于几个标准差",
# 这样阈值的量纲和普通 z 分数一致,好解释。
_MAD_TO_SIGMA = 0.6745


def _median(vals: list[float]) -> float:
    o = sorted(vals)
    n = len(o)
    return o[n // 2] if n % 2 else (o[n // 2 - 1] + o[n // 2]) / 2


def detect_outliers(ds: Dataset, z: float = OUTLIER_Z) -> str:
    """找离群行。先看数据本身有没有问题,再谈结论。

    为什么用中位数 + MAD,而不是均值 + 标准差:
    普通 z 分数有个致命的自指问题:**离群点会把均值和标准差一起撑大,
    于是把自己藏起来**(统计上叫 masking)。实测踩过:

        8 行降解率 22.4~41.2,混进一个 88.0
        均值 38.663、标准差 20.652 → |88−38.663| = 49.3 < 2.5×20.652 = 51.6
        → 最明显的那个异常值反而没被报出来

    换成中位数和 MAD(绝对偏差的中位数)就没这个问题:一个点动不了中位数。
    同样这组数据,修正 z 分数 = 0.6745×55.9/3.3 ≈ 11.4,远超阈值,当场命中。

    样本越小这个差别越致命,而田间数据恰恰都是小样本 —— 所以这不是
    "更严谨一点",是在我们的场景下前者根本不能用。

    注意:样本量小于 5 时不做判断:5 个点算不出有意义的离散度,
      硬报"异常"会把正常波动说成问题。
    """
    lines: list[str] = []
    for key, label in [
        *((f"{_T}{m}", TARGET_INFO.get(m, (m, ""))[0]) for m in ds.targets),
        *((k, k) for k in ds.features),
    ]:
        vals = _vals(ds.rows, key)
        if len(vals) < 5:
            continue
        med = round(_median(vals), 4)
        mad = round(_median([abs(v - med) for v in vals]), 4)
        if mad == 0:
            # 过半的值完全相同(常见于"整列填了同一个值")。此时任何偏离都会
            # 被判成无穷大的离群 —— 那不是发现,是噪声。直接跳过。
            continue
        hits = [
            (i, r[key], _MAD_TO_SIGMA * abs(r[key] - med) / mad)
            for i, r in enumerate(ds.rows, 1)
            if r.get(key) is not None and _MAD_TO_SIGMA * abs(r[key] - med) / mad > z
        ]
        if hits:
            detail = "、".join(f"第{i}行={v}(偏离 {s:.1f})" for i, v, s in hits[:5])
            lines.append(f"  · {label}:{len(hits)} 个离群点(中位数 {med},MAD {mad})→ {detail}")

    if not lines:
        n = ds.n_rows
        return f"未发现明显离群值(修正 z 分数阈值 {z})。" + (
            f"注意样本仅 {n} 行,小样本下这个结论参考价值有限。" if n < 10 else ""
        )
    return (
        f"离群检测(中位数 + MAD,修正 z 分数阈值 {z}):\n"
        + "\n".join(lines)
        + (
            "\n\n离群不等于错误:可能是记录笔误,也可能是真实的极端地块。"
            "建议先核对原始记录,再决定是否剔除。"
        )
    )


# ── 4) 国标符合性 ──────────────────────────────────────────────

# 只放【确定的、写在国标里】的硬指标。拿不准的宁可不判 ——
# 判错合规会让用户做出错误的商业决策。
_LIMITS = {
    "Thickness_um": (
        10.0,
        None,
        "GB 13735-2017 规定聚乙烯农用地膜标称厚度不低于 0.010mm(10µm)",
    ),
}


def check_standards(ds: Dataset) -> str:
    """对能明确判定的指标做国标符合性检查。

    只检查有明确国标条文的项。降解率、透过率这些的合格线随产品类型、
    作物、地区而变,没有一刀切的阈值 —— 硬给一个会误导用户。
    那类问题应该走 search_knowledge 查条文,而不是在这里写死。
    """
    lines: list[str] = []
    for key, (lo, hi, basis) in _LIMITS.items():
        vals = [(i, r[key]) for i, r in enumerate(ds.rows, 1) if r.get(key) is not None]
        if not vals:
            continue
        bad = [
            (i, v) for i, v in vals if (lo is not None and v < lo) or (hi is not None and v > hi)
        ]
        head = f"  · {key}:{len(vals)} 行有数据,"
        if bad:
            lines.append(
                head
                + f"**{len(bad)} 行不达标** → "
                + "、".join(f"第{i}行={v}" for i, v in bad[:5])
                + f"\n    依据:{basis}"
            )
        else:
            lines.append(head + f"全部达标(依据:{basis})")

    if not lines:
        return (
            "这张表里没有可做国标判定的列(目前支持:厚度)。\n"
            "降解率、透过率等指标的合格线随产品类型与作物而变,没有统一阈值 —— "
            "这类问题请直接问知识库查具体标准条文。"
        )
    return "国标符合性检查:\n" + "\n".join(lines)
