"""汇总、版本对比、显著性判断。

这个文件真正要解决的问题:凭什么说"新版本更好":
最常见也最错的做法:跑一遍旧版得 78%,跑一遍新版得 81%,宣布提升 3 个点。

问题是大模型有温度,同一个版本连跑两次本来就能差好几个点。3 个点可能
全是噪声。基于噪声做的优化决策,方向完全是随机的。

这里用两件事把它管住:

1) 噪声地板(noise floor) —— 同一版本重复跑 N 次,看指标自己抖多少。
   这是"什么都没改"时的差异下限。新旧差异没超过它 = 什么都没证明。

2) 配对符号检验 —— 两个版本跑的是【同一套题】,所以不该比两个平均分,
   而该逐题配对看:有多少题从错变对(b),多少题从对变错(c)。
   若真的没差别,一道题往哪个方向翻应该是五五开;于是 b、c 服从
   二项分布 B(b+c, 0.5),算个精确双侧 p 值即可。
   这就是 McNemar 检验的思路,好处是只看翻转的题,那些两版都对/都错的题
   本来就不携带区分信息,平均分却会被它们稀释。

不用 scipy:math.comb 就能精确算,少一个依赖。
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ruixue_agent.eval.schema import CATEGORY_ORDER
from ruixue_agent.eval.scoring import CaseScore


@dataclass
class Report:
    """一次完整评测的汇总。"""

    n: int = 0
    passed: int = 0
    by_category: dict[str, tuple[int, int]] = field(default_factory=dict)  # 类别 -> (过, 总)
    # None 表示【不适用】(这批题里没有需要调工具的),不是 0 分。
    # 用 0.0 表示不适用会在报告里显示成 "precision 0.000",看着像糟透了 ——
    # 把"没这回事"和"做得很差"显示成同一个样子,是最容易误导人的一类 bug。
    tool_precision: float | None = None
    tool_recall: float | None = None
    keypoint_recall: float | None = None
    # 成本三件套:钱、时间、工具调用次数。正确率提升但成本翻倍,不一定是好交易。
    avg_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tool_calls: float = 0.0
    errors: int = 0
    per_case: dict[str, bool] = field(default_factory=dict)  # 配对比较要用
    # 题面指纹:题号 → 问题文本的哈希。
    #
    # 为什么必须存:配对比较是按【题号】配的。如果哪天改了某道题的题面却沿用
    # 原题号,对比就会把两道不同的题当成同一道 —— 静默给出错误结论,而且几乎
    # 查不出来。实测就干过一次(rf04 换了题面沿用原号)。存指纹后能直接拦下。
    case_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0


def question_hash(text: str) -> str:
    """题面指纹。取前 10 位够用 —— 它只用来发现"题变了",不是密码学用途。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def aggregate(scores: list[CaseScore], traces=None, cases=None) -> Report:
    """把逐题结果汇总成一份报告。traces 传入时才有成本指标;cases 传入才有题面指纹。"""
    rep = Report(n=len(scores), passed=sum(s.passed for s in scores))
    rep.per_case = {s.case_id: s.passed for s in scores}
    rep.errors = sum(1 for s in scores if s.reason.startswith("运行异常"))
    if cases:
        rep.case_hashes = {c.id: question_hash(c.question) for c in cases}

    for s in scores:
        ok, tot = rep.by_category.get(s.category, (0, 0))
        rep.by_category[s.category] = (ok + int(s.passed), tot + 1)

    # 工具指标只在【确实期望调工具】的题上平均。
    # 把 no_tool 题的 recall(恒为 1)混进来会把整体拉高,看着好看但没意义。
    routed = [s for s in scores if s.category in ("tool_route", "multi_tool", "knowledge")]
    if routed:
        rep.tool_precision = statistics.fmean(s.tool_precision for s in routed)
        rep.tool_recall = statistics.fmean(s.tool_recall for s in routed)
    kp = [s for s in scores if s.keypoint_recall is not None and s.category != "refuse"]
    if kp:
        rep.keypoint_recall = statistics.fmean(s.keypoint_recall for s in kp)

    if traces:
        rep.avg_tokens = statistics.fmean(t.total_tokens for t in traces)
        rep.avg_latency_ms = statistics.fmean(t.latency_ms for t in traces)
        rep.avg_tool_calls = statistics.fmean(len(t.tool_calls) for t in traces)
    return rep


# ── 噪声地板 ──────────────────────────────────────────────────
def noise_floor(reports: list[Report]) -> dict:
    """同一版本重复跑 N 次的抖动幅度。

    返回 pass_rate 的极差和标准差。极差比标准差更该看 ——
    N 只有 3~5 次时标准差本身就很不稳,而"最好一次和最差一次差多少"
    是直接能理解、也直接能用来卡阈值的。
    """
    rates = [r.pass_rate for r in reports]
    out = {
        "runs": len(rates),
        "rates": rates,
        "spread": (max(rates) - min(rates)) if len(rates) > 1 else 0.0,
        "stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        **per_case_stability(reports),
    }
    return out


def per_case_stability(reports: list[Report]) -> dict:
    """逐题稳定性:哪几道题在重复跑之间会翻来翻去。

    为什么总通过率的极差不够:
    极差只告诉你"整体抖 6%",但 6% 可能是【同两道题反复翻】,也可能是
    【每次都是不同的题在翻】。这两种情况的处理方式完全不同:

        固定几道题不稳 → 那几道题本身写得含糊,或者 agent 在那类问题上确实摇摆
                         → 该修用例或修 agent
        到处随机翻     → 温度带来的普遍抖动
                         → 只能靠多跑几轮取平均,修单道题没意义

    更要紧的一点:一道只失败过一次的题,可能根本不是缺陷,只是抖了一下。
    在花时间去"修"它之前,先看它是不是每次都错 —— 否则修的是噪声。

    返回:
        flaky   在多轮之间结果不一致的题 → 结论不可信,别拿它当依据
        always_fail  每轮都失败 → 这才是真缺陷,值得修
        always_pass  每轮都通过
    """
    if len(reports) < 2:
        return {"flaky": (), "always_fail": (), "always_pass": ()}
    ids = set(reports[0].per_case)
    for r in reports[1:]:
        ids &= set(r.per_case)
    flaky, fail, ok = [], [], []
    for i in sorted(ids):
        vals = [r.per_case[i] for r in reports]
        (ok if all(vals) else fail if not any(vals) else flaky).append(i)
    return {"flaky": tuple(flaky), "always_fail": tuple(fail), "always_pass": tuple(ok)}


def consensus(reports: list[Report], categories: dict[str, str] | None = None) -> Report:
    """把多轮结果合成一份【共识报告】:每道题按多数票定输赢。

    categories:题号 → 类别。给了才能重算分类别统计(Report 里不含逐题类别)。

    为什么对比要用共识而不是某一轮:
    单轮自己就抖(实测温度 0 下极差仍有 6.1%)。拿单轮当基线做对比,等于
    用一把会晃的尺子去量另一把会晃的尺子 —— 而且很容易不自觉地挑一轮好看的
    当基线,那就成了自欺。

    多数票把单轮噪声压下去:一道题要被判定为"错",得三轮里错两轮。
    代价是要跑 N 倍的时间和钱,但这是"能下结论"的入场费。

    成本指标取各轮平均 —— 成本本身也在抖,单轮数字没有代表性。
    """
    if not reports:
        return Report()
    if len(reports) == 1:
        return reports[0]
    ids = set(reports[0].per_case)
    for r in reports[1:]:
        ids &= set(r.per_case)
    per = {i: sum(r.per_case[i] for r in reports) * 2 > len(reports) for i in sorted(ids)}

    out = Report(n=len(per), passed=sum(per.values()), per_case=per)
    out.errors = round(statistics.fmean(r.errors for r in reports))
    for key in ("tool_precision", "tool_recall", "keypoint_recall"):
        vals = [getattr(r, key) for r in reports if getattr(r, key) is not None]
        setattr(out, key, statistics.fmean(vals) if vals else None)
    for key in ("avg_tokens", "avg_latency_ms", "avg_tool_calls"):
        setattr(out, key, statistics.fmean(getattr(r, key) for r in reports))

    # 分类别按共识逐题重算,【不能】把各轮的分子分母相加 ——
    # 那样 3 轮会变成 "9/12",看着像 12 道题,实际只有 4 道,读者会误判样本量。
    if categories:
        for cid, ok in per.items():
            cat = categories.get(cid)
            if cat is None:
                continue
            hit, tot = out.by_category.get(cat, (0, 0))
            out.by_category[cat] = (hit + int(ok), tot + 1)
    return out


# ── 配对比较 ──────────────────────────────────────────────────
def _binom_two_sided(b: int, c: int) -> float:
    """b 次"变好"、c 次"变差"时,若真无差别,出现这么极端结果的概率。

    零假设:每次翻转往两边等概率。于是 b ~ B(n=b+c, p=0.5)。
    双侧 p = 2 × P(X ≤ min(b,c)),并截断到 1。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _rate(rep: Report, ids: set[str]) -> float:
    return (sum(rep.per_case[i] for i in ids) / len(ids)) if ids else 0.0


@dataclass
class Comparison:
    improved: tuple[str, ...]  # 从错变对的题
    regressed: tuple[str, ...]  # 从对变错的题
    delta_pass_rate: float
    p_value: float
    verdict: str
    changed: tuple[str, ...] = ()  # 题面改过、已排除在对比之外的题


def compare(base: Report, cur: Report, floor: float = 0.0, alpha: float = 0.05) -> Comparison:
    """逐题配对比较两个版本。

    floor:噪声地板(来自 noise_floor 的 spread)。差异没超过它就别下结论,
           哪怕 p 值好看 —— p 值只说"翻转不像随机",不管翻转是不是温度导致的。
    """
    common = set(base.per_case) & set(cur.per_case)
    # 题号相同但【题面变了】的,不能拿来配对 —— 那是两道不同的题。
    # 静默比下去会给出看似合理、实则毫无意义的结论。
    changed = tuple(
        sorted(
            i
            for i in common
            if base.case_hashes.get(i)
            and cur.case_hashes.get(i)
            and base.case_hashes[i] != cur.case_hashes[i]
        )
    )
    common -= set(changed)
    # 基线是加指纹功能【之前】存的 → 无从判断题面变没变。
    # 这时候必须明说"没法验证",而不是默认可比 —— 默认可比正是这个守卫要防的事。
    # 实测就吃过一次:rf04 换了题面,对比却把它算成"变好",差点当成改动生效。
    unverifiable = not base.case_hashes and bool(cur.case_hashes)
    improved = tuple(sorted(i for i in common if not base.per_case[i] and cur.per_case[i]))
    regressed = tuple(sorted(i for i in common if base.per_case[i] and not cur.per_case[i]))
    # 通过率也只在可比的题上算,否则分母里混着不可比的题
    delta = _rate(cur, common) - _rate(base, common)
    p = _binom_two_sided(len(improved), len(regressed))

    if not common:
        verdict = "无法比较:两次运行没有共同题目(评测集变了?)"
    elif abs(delta) <= floor:
        verdict = f"差异 {delta:+.1%} 未超过噪声地板 {floor:.1%} —— 不能说明任何问题"
    elif p > alpha:
        verdict = f"翻转数 {len(improved)}↑/{len(regressed)}↓,p={p:.3f} > {alpha},差异不显著"
    else:
        d = "提升" if delta > 0 else "下降"
        verdict = f"{d}显著(p={p:.3f}),{len(improved)}↑/{len(regressed)}↓"
    if changed:
        verdict += f"(已排除题面改过的 {len(changed)} 题:{', '.join(changed)})"
    if unverifiable:
        verdict += "  ⚠ 基线不含题面指纹,无法确认两边是同一套题 —— 结论仅供参考"
    return Comparison(improved, regressed, delta, p, verdict, changed)


# ── 输出 ──────────────────────────────────────────────────────
def render(rep: Report, title: str = "Agent 评测") -> str:
    lines = [f"── {title} ──  {rep.passed}/{rep.n} 通过  ({rep.pass_rate:.1%})"]
    if rep.errors:
        lines.append(f"  ⚠ 其中 {rep.errors} 题是运行异常,不是答错 —— 先修环境再看分数")
    lines.append("  分类别:")
    for cat in CATEGORY_ORDER:
        if cat not in rep.by_category:
            continue
        ok, tot = rep.by_category[cat]
        bar = "█" * round(ok / tot * 10) + "·" * (10 - round(ok / tot * 10))
        lines.append(f"    {cat:<11} {bar} {ok}/{tot}")

    def _n(v):
        return "  n/a" if v is None else f"{v:.3f}"

    lines.append(
        f"  工具选择  precision {_n(rep.tool_precision)}  recall {_n(rep.tool_recall)}"
        f"   |  关键点召回 {_n(rep.keypoint_recall)}"
    )
    lines.append(
        f"  成本      {rep.avg_tokens:.0f} tokens/题   {rep.avg_latency_ms / 1000:.1f}s/题"
        f"   {rep.avg_tool_calls:.1f} 次工具/题"
    )
    return "\n".join(lines)


def save(rep: Report, scores: list[CaseScore], traces, path: Path) -> None:
    """存盘,供以后当基线对比。

    连轨迹一起存:分数低的时候必须能翻出"这道题当时到底调了什么、答了什么",
    只存一个数字是没法改进的。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "report": asdict(rep),
                "scores": [asdict(s) for s in scores],
                "traces": [t.to_dict() for t in (traces or [])],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load(path: Path) -> Report:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["report"]
    rep = Report(**{k: v for k, v in raw.items() if k in Report.__dataclass_fields__})
    # JSON 会把 tuple 变成 list,by_category 的值要还原成元组以便比较
    rep.by_category = {k: tuple(v) for k, v in rep.by_category.items()}
    return rep
