"""跑评测:把一道题喂给 agent,收轨迹,判分。

两个刻意的设计:
1) 每道题一个全新会话。 用内存 checkpointer + 独立 thread_id,题与题之间
零串扰。共用会话的话,上一题的上下文会帮到下一题(或者干扰它),分数就不是
"这道题的能力",而是"题目顺序"的函数 —— 换个顺序分数就变,没法复现。

2) 异常不中断整轮。 一道题炸了就记成 error 继续跑下一道。跑一轮要几十分钟
甚至花不少钱,不能因为第 7 题超时就前功尽弃。但 error 会单独统计、
不混进正确率 —— 环境问题和能力问题必须分开看。
"""

from __future__ import annotations

import logging
import uuid

from ruixue_agent.eval.schema import EvalCase
from ruixue_agent.eval.scoring import CaseScore, score_case
from ruixue_agent.eval.trace import Timer, Trace, extract
from ruixue_agent.subagents import collect_subagent_runs

logger = logging.getLogger("ruixue.eval")


# 评测用的温度。0 不是"更聪明",是为了能测量。
#
# 实测:不设温度(用服务端默认 1.0)时,同一版本连跑三轮通过率
# 84.8% / 93.9% / 97.0%,极差 12.1% ≈ 4 道题 —— 这把尺子测不出任何
# 小于 4 道题的改进,拿它做 A/B 对比等于掷骰子。
#
# 注意:温度 0 不等于完全确定(服务端批处理、MoE 路由仍有残余抖动),
#   它只是去掉了最大的那个噪声源。噪声地板仍然要实测,不能假设它是 0。
EVAL_TEMPERATURE = 0.0


def build_eval_agent(model_name: str = "deepseek-v4-pro", temperature: float = EVAL_TEMPERATURE):
    """装配一个用于评测的 agent —— 和生产同一套装配代码,只换 checkpointer 和温度。

    必须复用生产的装配函数,不能在这里另拼一个:另拼的那个迟早和生产跑偏,
    于是评测测的是一个线上并不存在的 agent,分数再好看也没用。
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from ruixue_agent.agents.builder import create_ruixue_agent

    return create_ruixue_agent(
        model_name=model_name, checkpointer=InMemorySaver(), temperature=temperature
    )


def run_case(
    agent, case: EvalCase, recursion_limit: int = 40, system_prompt: str = ""
) -> tuple[Trace, CaseScore]:
    """跑一道题,返回 (轨迹, 分数)。"""
    cfg = {
        # 每题独立线程 —— 题与题之间不许串味
        "configurable": {"thread_id": f"eval-{case.id}-{uuid.uuid4().hex[:8]}"},
        # 图层面的硬刹车。中间件里的 ModelCallLimit 是"优雅收尾",
        # 这个是"再不停就抛异常",防止评测被一道死循环的题挂住。
        "recursion_limit": recursion_limit,
    }
    try:
        # 收集本次运行的子 agent 委派 —— 否则它们烧的 token 和调的工具全是黑箱
        with collect_subagent_runs() as subs, Timer() as t:
            state = agent.invoke({"messages": [{"role": "user", "content": case.question}]}, cfg)
        tr = extract(case.id, state, t.ms)
        tr.subagent_runs = list(subs)
    except Exception as e:
        # 记类型和消息就够;完整堆栈进日志。评测报告里塞堆栈会淹没真正的信号。
        logger.exception("用例 %s 运行失败", case.id)
        tr = Trace(case_id=case.id, error=f"{type(e).__name__}: {e}"[:200], latency_ms=0)
    return tr, score_case(case, tr, system_prompt=system_prompt)


def run_all(
    agent, cases: list[EvalCase], on_case=None, system_prompt: str = ""
) -> tuple[list[Trace], list[CaseScore]]:
    """顺序跑完整套题。

    为什么不并发:并发会同时打模型的限流,而且相互竞争会让延迟指标失真 ——
    延迟本身是我们要测的东西,不能被评测方式污染。评测慢一点没关系,准更重要。
    """
    traces, scores = [], []
    for i, c in enumerate(cases, 1):
        tr, sc = run_case(agent, c, system_prompt=system_prompt)
        traces.append(tr)
        scores.append(sc)
        if on_case:
            on_case(i, len(cases), c, sc)
    return traces, scores
