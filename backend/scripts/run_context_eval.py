"""上下文治理量化实验:摘要压缩到底什么时候触发、压掉多少。

    uv run python scripts/run_context_eval.py                # 默认 30 轮
    uv run python scripts/run_context_eval.py --turns 40
    uv run python scripts/run_context_eval.py --out runs/context_v1.json

为什么必须量这个:
对外材料和文档上写着「超 5 万 token 触发摘要压缩、保留最近 20 条」。
但这句话从来没有被验证过 —— 评审问一句「你们大概几轮会触发?」,
现在答不上来。更糟的可能是:阈值定得太高,真实对话根本到不了,
那这个模块等于写在对外材料上但从不生效(记忆那次就是这么发现的)。

粗估:实测单轮 input 均值 6147 token、output 均值 635,扣掉每轮重发的
固定开销(系统提示 + 14 个工具的 schema)后每轮净增约 2300 token
→ 约第 22 轮触发。但估算不能当数字用,所以真跑一遍。

实验设计:
同一个 thread_id 连续问 N 轮(和记忆实验刻意相反 —— 那边要跨 thread
才测得到长期记忆,这边要同 thread 才让上下文累积)。

每轮记录:进模型前的消息条数、token 数、是否触发了压缩、压缩前后的差值。
触发那一轮的前后对比就是压缩率。

问题刻意选知识类:它们会带回长工具结果,上下文涨得快,
能在可接受的轮数内摸到阈值。这是为了让实验跑得完,不是为了让数字好看——
报告里会同时给出"每轮净增"这个与问题类型无关的量。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 连续追问,每轮都往上下文里加东西。刻意混入知识类问题(工具结果长)。
TURNS = [
    "地膜覆盖对土壤温度有什么影响?",
    "那对土壤水分呢?",
    "PBAT 和 PLA 在降解膜里各起什么作用?",
    "全生物降解地膜的国家标准对断裂伸长率有什么要求?",
    "厚度对地膜性能的影响是怎样的?",
    "地膜残留对土壤有什么危害?",
    "生物降解地膜在田间靠什么降解?",
    "紫外线对地膜老化的影响机理是什么?",
    "覆膜对杂草有抑制作用吗?原理是什么?",
    "地膜回收目前有哪些技术路线?",
    "棉花覆膜的技术规程大概是什么?",
    "玉米覆膜和棉花有什么不同?",
    "水稻能用地膜吗?",
    "地膜的透光率对作物有什么影响?",
    "黑色地膜和透明地膜怎么选?",
    "降解膜的诱导期是什么意思?",
    "土壤微生物怎么影响降解速率?",
    "土壤 pH 对降解有影响吗?",
    "地膜厚度和用量是什么关系?",
    "国标对地膜的拉伸强度怎么规定?",
    "地膜的水蒸气透过率意味着什么?",
    "干旱地区选膜要注意什么?",
    "风大的地方呢?",
    "覆膜时间对产量的影响有研究吗?",
    "揭膜时机怎么定?",
    "地膜和滴灌怎么配合?",
    "残膜回收机械有哪些类型?",
    "地膜污染的治理政策有哪些?",
    "降解膜的成本比 PE 高多少?",
    "总结一下我们刚才聊过的要点。",
]


@dataclass
class TurnRecord:
    turn: int
    question: str
    n_messages_before: int  # 进模型前的消息条数
    tokens_before: int  # 进模型前的 token 数
    summarized: bool  # 这一轮是否触发了压缩
    n_messages_after: int = 0  # 压缩后的消息条数
    tokens_after: int = 0
    error: str = ""

    @property
    def compression_pct(self) -> float:
        if not self.summarized or not self.tokens_before:
            return 0.0
        return round(100 * (1 - self.tokens_after / self.tokens_before), 2)


@dataclass
class Report:
    turns: list[TurnRecord] = field(default_factory=list)

    def summary(self) -> dict:
        ok = [t for t in self.turns if not t.error]
        fired = [t for t in ok if t.summarized]
        toks = [t.tokens_before for t in ok]
        # 每轮净增:用相邻两轮的差值中位数,比"总量/轮数"稳健
        # (触发压缩那一轮会出现负增长,均值会被它拉偏)
        deltas = sorted(b - a for a, b in zip(toks, toks[1:], strict=False))
        median_delta = deltas[len(deltas) // 2] if deltas else 0
        return {
            "turns_run": len(ok),
            "first_trigger_turn": fired[0].turn if fired else None,
            "trigger_count": len(fired),
            "max_tokens": max(toks) if toks else 0,
            "median_growth_per_turn": median_delta,
            "avg_prompt_tokens": round(sum(toks) / len(toks)) if toks else 0,
            "compressions": [
                {
                    "turn": t.turn,
                    "tokens": f"{t.tokens_before} → {t.tokens_after}",
                    "messages": f"{t.n_messages_before} → {t.n_messages_after}",
                    "compression_pct": t.compression_pct,
                }
                for t in fired
            ],
            "avg_compression_pct": (
                round(sum(t.compression_pct for t in fired) / len(fired), 2) if fired else 0.0
            ),
            "max_compression_pct": (max(t.compression_pct for t in fired) if fired else 0.0),
            "errors": sum(1 for t in self.turns if t.error),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=len(TURNS))
    ap.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="临时改压缩阈值(仅本次实验,不改生产配置)。"
        "用途:生产阈值 5 万在 30 轮内根本不触发,要量压缩率就得先让它触发。",
    )
    ap.add_argument(
        "--keep",
        type=int,
        default=0,
        help="临时改保留的最近消息条数(生产是 20)。"
        "我们每轮约 4 条消息,20 条≈最近 5 轮 —— 可压的部分本来就少,"
        "实测压缩率仅 1.33%,其中一次是负的(摘要比原文还长)。",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from langchain.agents.middleware import SummarizationMiddleware

    from ruixue_agent.agents.builder import KEEP_RECENT_MESSAGES, SUMMARIZE_AT_TOKENS
    from ruixue_agent.eval.runner import build_eval_agent

    rec = Report()
    current: dict = {}

    # 只改【本次实验】用的那个中间件实例的阈值,不动 builder 里的常量 ——
    # 生产配置该由数据来改,不该被一个实验脚本顺手改掉。
    threshold = args.threshold or SUMMARIZE_AT_TOKENS

    # 插桩:包住真实的 before_model,记录压缩前后的状态。
    # 不改生产代码,只在实验期间替换方法 —— 测的必须是线上那套逻辑。
    orig = SummarizationMiddleware.before_model

    def spy(self, state, runtime):
        # 改阈值只能在这里做:编译后的 agent 不暴露 .middleware,
        # 但插桩函数拿得到 self(就是那个中间件实例)。
        # _trigger_conditions 是 [(kind, value)] 列表,见 _should_summarize。
        if args.threshold:
            self._trigger_conditions = [("tokens", args.threshold)]
        if args.keep:
            self.keep = ("messages", args.keep)
            if hasattr(self, "_keep_condition"):
                self._keep_condition = ("messages", args.keep)
        msgs = state["messages"]
        current["n_before"] = len(msgs)
        current["tok_before"] = self.token_counter(msgs)
        out = orig(self, state, runtime)
        if out and "messages" in out:
            # 压缩后的消息里,RemoveMessage 之类的控制消息不算内容
            new_msgs = [m for m in out["messages"] if type(m).__name__ != "RemoveMessage"]
            current["fired"] = True
            current["n_after"] = len(new_msgs)
            current["tok_after"] = self.token_counter(new_msgs)
        return out

    SummarizationMiddleware.before_model = spy
    try:
        agent = build_eval_agent()
        if args.threshold:
            # 遍历已编译 agent 的中间件,把 SummarizationMiddleware 的阈值改掉。
            for mw in getattr(agent, "middleware", []) or []:
                if isinstance(mw, SummarizationMiddleware):
                    mw.trigger = ("tokens", args.threshold)
                    if hasattr(mw, "max_tokens_before_summary"):
                        mw.max_tokens_before_summary = args.threshold
        tid = f"ctxeval:{uuid.uuid4().hex[:8]}"  # 同一个 thread —— 上下文必须累积
        print(f"上下文实验:同一会话连问 {args.turns} 轮")
        # 注意:表头必须反映【实际生效】的参数,不能打印常量 ——
        #   否则用 --keep 6 跑出来的报告上写着"保留 20 条",
        #   过两周回看这份结果会完全误读。踩过一次。
        keep = args.keep or KEEP_RECENT_MESSAGES
        tnote = "" if not args.threshold else f"(实验值,生产 {SUMMARIZE_AT_TOKENS})"
        knote = "" if not args.keep else f"(实验值,生产 {KEEP_RECENT_MESSAGES})"
        print(f"触发阈值 {threshold} tokens{tnote},保留最近 {keep} 条{knote}")
        print("=" * 72)

        for i, q in enumerate(TURNS[: args.turns], 1):
            current.clear()
            err = ""
            try:
                agent.invoke(
                    {"messages": [{"role": "user", "content": q}]},
                    {"configurable": {"thread_id": tid}, "recursion_limit": 30},
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:120]

            t = TurnRecord(
                turn=i,
                question=q,
                n_messages_before=current.get("n_before", 0),
                tokens_before=current.get("tok_before", 0),
                summarized=bool(current.get("fired")),
                n_messages_after=current.get("n_after", 0),
                tokens_after=current.get("tok_after", 0),
                error=err,
            )
            rec.turns.append(t)
            mark = (
                f"  ★ 触发压缩 {t.tokens_before}→{t.tokens_after}({t.compression_pct}%)"
                if t.summarized
                else ""
            )
            print(
                f"[{i:>2}/{args.turns}] {t.n_messages_before:>3} 条 / {t.tokens_before:>6} tok{mark}"
            )
            if err:
                print(f"       ⚠ {err}")
    finally:
        SummarizationMiddleware.before_model = orig

    s = rec.summary()
    print("=" * 72)
    if s["first_trigger_turn"]:
        print(f"首次触发:第 {s['first_trigger_turn']} 轮;共触发 {s['trigger_count']} 次")
        print(f"平均压缩率 {s['avg_compression_pct']}%,最高 {s['max_compression_pct']}%")
    else:
        # 这本身就是结论:阈值定得太高,真实对话根本到不了。
        print(f"⚠ {s['turns_run']} 轮内【从未触发】压缩,峰值仅 {s['max_tokens']} tokens")
        print(f"  阈值 {threshold} 定得偏高 —— 这个模块在真实对话里等于不生效。")
    print(f"每轮上下文净增(中位数):{s['median_growth_per_turn']} tokens")
    print(f"平均 prompt 长度:{s['avg_prompt_tokens']} tokens;峰值 {s['max_tokens']}")
    if s["errors"]:
        print(f"⚠ {s['errors']} 轮运行异常")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"summary": s, "turns": [asdict(t) for t in rec.turns]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"已写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
