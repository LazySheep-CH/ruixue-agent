"""Agent 级评测入口 —— 回答"这个 agent 到底能不能把活干成"。

和 scripts/run_eval.py 的分工:
    run_eval.py        只测检索层:有没有把对的资料捞上来(Recall@k / MRR)
    run_agent_eval.py  测整个 agent:工具选得对不对、该拒答时拒没拒、
                       注入挡没挡住、花了多少钱

用法:
    uv run python scripts/run_agent_eval.py                    # 跑一轮,存基线
    uv run python scripts/run_agent_eval.py --repeat 3         # 跑 3 轮量噪声地板
    uv run python scripts/run_agent_eval.py --baseline runs/xx.json   # 和基线比
    uv run python scripts/run_agent_eval.py --only injection   # 只跑某一类(快速验证)
    uv run python scripts/run_agent_eval.py --dry-run          # 只校验评测集,不调模型

⚠ 这个脚本【会真的调大模型花钱】。33 题一轮大约几分钟。
   要在 CI 里跑的确定性部分在 tests/test_agent_eval.py,那部分不花钱。

## 怎么用它做版本对比(这才是评测的意义)

    1. 改动前:--repeat 3 跑三轮,记下噪声地板(同版本自己抖多少)
    2. 存一份基线:结果自动落在 runs/agent_eval_<时间戳>.json
    3. 改动后:--baseline <那份基线> 再跑
    4. 看输出的判定 —— 差异没超过噪声地板,就【不要】声称有提升

第 1 步最容易被省掉,但省掉它,后面所有对比都没有意义。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from ruixue_agent.eval import report as rp
from ruixue_agent.eval.runner import build_eval_agent, run_all
from ruixue_agent.eval.schema import load_cases

EVAL = Path("data/eval/agent_evalset.jsonl")
RUNS = Path("runs")


def _known_tools() -> set[str]:
    """真实工具名集合 —— 用来校验评测集里的工具名没写错。"""
    from ruixue_agent.subagents import delegate_to_expert
    from ruixue_agent.tools import get_tools

    return {t.name for t in get_tools()} | {delegate_to_expert.name}


def _progress(i, n, case, score):
    mark = "✓" if score.passed else "✗"
    tail = "" if score.passed else f"  ← {score.reason}"
    print(f"  [{i:>2}/{n}] {mark} {case.id} ({case.category}){tail}", flush=True)


def _rescore(cases, path: Path) -> int:
    """用【当前的判分逻辑】重判一份已存的轨迹,不调模型。

    为什么这是个一等操作:判分规则改了之后,历史分数就不可比了 ——
    但历史【轨迹】还是有效的(agent 当时确实那么做了)。重判而不是重跑,
    既省钱又能得到真正可比的数字。

    实测价值:首轮报 78.8%,修掉判分 bug 后同一批轨迹重判是 97.0% ——
    差的 18 个点全是判分错误,agent 一行没改。没有这个能力,就只能再花一次钱
    重跑,而且还分不清"分数变了"是因为改了判分还是因为模型本来就飘。
    """
    import json

    from ruixue_agent.agents.prompt import SYSTEM_PROMPT
    from ruixue_agent.eval.scoring import score_case
    from ruixue_agent.eval.trace import ToolCall, Trace

    raw = json.loads(path.read_text(encoding="utf-8"))
    traces = {
        t["case_id"]: Trace(
            case_id=t["case_id"],
            answer=t["answer"],
            tool_calls=[ToolCall(**c) for c in t["tool_calls"]],
            input_tokens=t["input_tokens"],
            output_tokens=t["output_tokens"],
            latency_ms=t["latency_ms"],
            interrupted=t.get("interrupted", False),
            error=t.get("error", ""),
        )
        for t in raw["traces"]
    }
    picked = [c for c in cases if c.id in traces]
    if not picked:
        print(f"{path} 里没有与当前评测集匹配的轨迹(评测集换了?)")
        return 2

    scores = [score_case(c, traces[c.id], system_prompt=SYSTEM_PROMPT) for c in picked]
    rep = rp.aggregate(scores, [traces[c.id] for c in picked])
    print(rp.render(rep, f"离线重判 {path.name}"))
    for s in scores:
        if not s.passed:
            print(f"    ✗ {s.case_id} ({s.category}): {s.reason}")

    out = path.with_name(path.stem + "_rescored.json")
    rp.save(rep, scores, [traces[c.id] for c in picked], out)
    print(f"\n重判结果已存:{out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1, help="重复跑几轮(≥2 才能量噪声地板)")
    ap.add_argument("--baseline", type=Path, help="对比用的基线结果文件")
    ap.add_argument("--only", help="只跑某个类别,如 injection")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--dry-run", action="store_true", help="只校验评测集,不调模型")
    ap.add_argument("--rescore", type=Path, help="改了判分后,离线重判某次已存的轨迹(不花钱)")
    args = ap.parse_args()

    cases = load_cases(EVAL, known_tools=_known_tools())
    if args.rescore:
        return _rescore(cases, args.rescore)
    if args.only:
        cases = [c for c in cases if c.category == args.only]
        if not cases:
            print(f"没有类别为 {args.only} 的用例")
            return 2
    print(f"评测集:{len(cases)} 题  模型:{args.model}")

    if args.dry_run:
        # 评测集本身就该能被单独校验:工具名、正则、类别自洽性全在加载期查过了。
        print("✓ 评测集校验通过(--dry-run 不调用模型)")
        return 0

    from ruixue_agent.agents.prompt import SYSTEM_PROMPT

    agent = build_eval_agent(args.model)

    # 存盘:分数低的时候要能翻出"哪道题、调了什么、答了什么"
    RUNS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports, last, saved = [], None, []
    for r in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n── 第 {r}/{args.repeat} 轮 ──")
        traces, scores = run_all(agent, cases, on_case=_progress, system_prompt=SYSTEM_PROMPT)
        rep = rp.aggregate(scores, traces)
        reports.append(rep)
        print()
        print(rp.render(rep, f"第 {r} 轮" if args.repeat > 1 else "Agent 评测"))
        last = (rep, scores, traces)

        # 【每轮都存】,不是只存最后一轮。
        #
        # 多轮跑的目的就是找摇摆的题,而一道题为什么摇摆,只能靠【对照它在
        # 各轮的轨迹】看出来 —— 这次调了工具下次没调?检索回来的片段不一样?
        # 只留最后一轮,等于把诊断所需的对照组丢了,还得再花一次钱重跑。
        #
        # 另一个好处:跑到一半崩了,前面几轮的结果不会一起没。
        out = RUNS / (
            f"agent_eval_{stamp}_r{r}.json" if args.repeat > 1 else f"agent_eval_{stamp}.json"
        )
        rp.save(rep, scores, traces, out)
        saved.append(out)
    print("\n结果已存:" + ", ".join(str(p) for p in saved))

    floor = 0.0
    if args.repeat > 1:
        nf = rp.noise_floor(reports)
        floor = nf["spread"]
        print(
            f"\n── 噪声地板({nf['runs']} 轮同版本重复)──\n"
            f"  通过率:{['%.1f%%' % (x * 100) for x in nf['rates']]}\n"
            f"  极差 {nf['spread']:.1%}  标准差 {nf['stdev']:.1%}\n"
            f"  → 以后版本对比时,差异不超过 {nf['spread']:.1%} 就不能声称有变化。"
        )
        # 逐题稳定性比总极差更可操作:它区分"该修的真缺陷"和"修了也白修的噪声"。
        if nf["always_fail"]:
            print(f"  ✗ 每轮都失败(真缺陷,值得修):{', '.join(nf['always_fail'])}")
        if nf["flaky"]:
            print(
                f"  ~ 时对时错(结论不可信,别拿它当依据):{', '.join(nf['flaky'])}\n"
                f"    这些题要么本身写得含糊,要么 agent 在这类问题上确实摇摆。"
            )
        if not nf["flaky"]:
            print("  ✓ 没有摇摆的题 —— 单次运行的结论就可以直接采信")

    if args.baseline:
        base = rp.load(args.baseline)
        cmp = rp.compare(base, last[0], floor=floor)
        print(f"\n── 与基线对比({args.baseline.name})──")
        print(
            f"  通过率 {base.pass_rate:.1%} → {last[0].pass_rate:.1%}  ({cmp.delta_pass_rate:+.1%})"
        )
        if cmp.improved:
            print(f"  变好:{', '.join(cmp.improved)}")
        if cmp.regressed:
            print(f"  变差:{', '.join(cmp.regressed)}")
        print(f"  判定:{cmp.verdict}")

    # 有运行异常时用非零退出码 —— 这样接 CI 或脚本时不会把"环境挂了"当成"跑完了"。
    return 1 if last[0].errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
