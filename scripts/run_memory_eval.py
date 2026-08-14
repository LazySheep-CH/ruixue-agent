"""长期记忆收益实验入口:量"有记忆"比"没记忆"好多少。

    uv run python scripts/run_memory_eval.py                 # 跑全部 12 题
    uv run python scripts/run_memory_eval.py --cases m01,m06 # 只跑指定几题
    uv run python scripts/run_memory_eval.py --out data/eval/baselines/memory_v1.json

和 run_agent_eval.py 的区别:那个测"agent 能力有多强",这个测
"某一个模块贡献了多少" —— 是**对照实验**,不是能力评分。
所以它的输出是一个【差值】,不是一个分数。

前置:PG + Milvus 起着(要真写、真召回记忆)。跑一轮约 24 次对话 + 24 次抽取。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ruixue_agent.eval.memory_bench import (  # noqa: E402
    load_memory_cases,
    run_bench,
)
from ruixue_agent.eval.runner import build_eval_agent  # noqa: E402

EVALSET = "data/eval/memory_evalset.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evalset", default=EVALSET)
    ap.add_argument("--cases", default="", help="只跑这些 id,逗号分隔")
    ap.add_argument("--out", default="", help="结果写到这个 json")
    args = ap.parse_args()

    cases = load_memory_cases(args.evalset)
    if args.cases:
        want = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.id in want]
        if not cases:
            print(f"没有匹配 {want} 的题")
            return 1

    print(f"记忆收益实验:{len(cases)} 题 × 2 arm(off / on)")
    print("=" * 74)

    # 和生产同一套装配,只换 checkpointer 和温度(温度 0 是为了能测量,见 runner.py)
    agent = build_eval_agent()

    def show(i, n, c, off, on):
        mark = {
            (False, True): "  ↑ 记忆救回",
            (True, False): "  ↓ 记忆帮倒忙",
        }.get((off.passed, on.passed), "")
        flags = ""
        if not on.stored:
            flags += "  ⚠ 没抽到事实"
        if on.stored and not on.injected:
            # 存了却没进上下文 —— 这才是"记忆没生效",和"抽取失败"是两个病
            flags += "  ⚠ 存了但没注入"
        print(
            f"[{i:>2}/{n}] {c.id}  off={'✔' if off.passed else '✘'} "
            f"on={'✔' if on.passed else '✘'}  "
            f"注入{'是' if on.injected else '否'}/存{on.stored}条{mark}{flags}"
        )
        if on.asked_back:
            print("         on 组仍在反问 —— 记忆没被用上")
        elif on.missed:
            print(f"         on 组漏掉:{list(on.missed)}")

    rep = run_bench(agent, cases, on_case=show)
    s = rep.summary()

    print("=" * 74)
    print(f"无记忆(off):{s['off_rate']:.1%}")
    print(f"有记忆(on) :{s['on_rate']:.1%}")
    print(f"净收益      :{s['delta']:+.1%}  ({s['n']} 题)")
    if s["errors"]:
        print(f"⚠ 运行异常 {s['errors']} 次(不计入正确率)")
    # 分层诊断:三种失败方式的解法完全不同,混成一个数就没法定位
    print(f"记忆真正进上下文:{s['injected']}/{s['n']} 题")
    print(f"仍在反问(诊断,不进总分):off {s['off_asked_back']} / on {s['on_asked_back']}")
    if s["no_fact_stored"]:
        print(f"⚠ {s['no_fact_stored']} 题没抽到任何事实 —— 瓶颈在【抽取层】")
    if s["stored_not_injected"]:
        print(f"⚠ {s['stored_not_injected']} 题存了却没注入 —— 瓶颈在【召回/注入层】")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "summary": s,
                    "off": [asdict(r) for r in rep.off],
                    "on": [asdict(r) for r in rep.on],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"已写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
