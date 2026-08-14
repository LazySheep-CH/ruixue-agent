"""从一次 agent 运行里抽出可判分的【轨迹】。

为什么单独抽一层,而不是在判分时直接翻 messages:
判分要回答的是"工具选对了吗、答案里有没有这几个要点";它不该关心
LangChain 的消息对象长什么样。中间隔一层 Trace,好处是:

    · 判分函数变成纯函数(输入 Trace,输出分数),可以单测,不用起 agent;
    · 换框架/升版本导致消息结构变了,只改这一个文件;
    · 记录下来的 Trace 可以存盘复盘 —— 分数低的时候要能回看它到底调了什么。

最后一点很实际:评测报告只给一个数字是没法改进的,必须能翻出"哪道题、
走了什么路径、错在哪一步"。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    # 工具返回内容截断存 —— 全存会让轨迹文件大到没法看,但完全不存又没法复盘。
    result_preview: str = ""


@dataclass
class Trace:
    """一次运行的全部可观测事实。判分只看这个对象。"""

    case_id: str
    answer: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    # 中断(HITL)或异常都算"没跑完",要和"跑完了但答错"区分开 ——
    # 前者是系统问题,后者是能力问题,混在一起会误导优化方向。
    interrupted: bool = False
    error: str = ""
    # 本次运行里发生的【子 agent 委派】账单。
    #
    # 为什么必须单独收:子 agent 的消息不进父状态,所以它烧的 token
    # 完全不在下面 input_tokens/output_tokens 里 —— 只要发生委派,
    # 成本统计就是偏低的,而我们还拿它做版本对比。
    # 见 subagents.collect_subagent_runs。
    subagent_runs: list = field(default_factory=list)
    # 执行失败的工具名。工具挂了(Milvus 断连等)会被中间件降级成一条提示,
    # agent 于是老实回"该功能暂时不可用"—— 判分若按"缺要点"算,就把
    # 【环境问题】记成了【能力问题】,你会去改提示词,而实际上要修的是 Milvus。
    failed_tools: list[str] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        """按调用顺序的工具名(可能重复:同一工具调多次)。"""
        return [c.name for c in self.tool_calls]

    @property
    def tool_set(self) -> set[str]:
        return {c.name for c in self.tool_calls}

    @property
    def total_tokens(self) -> int:
        """总 token = 父 agent + 【所有子 agent】。

        漏掉子 agent 的那部分是真发生过的 bug:成本指标偏低,
        而版本对比正是拿它比的。
        """
        sub = sum(r.input_tokens + r.output_tokens for r in self.subagent_runs)
        return self.input_tokens + self.output_tokens + sub

    @property
    def all_tool_names(self) -> list[str]:
        """父 agent + 子 agent 内部调过的所有工具。排查"到底干了什么"时看这个。"""
        inner = [t for r in self.subagent_runs for t in r.tools]
        return self.tool_names + inner

    def to_dict(self) -> dict:
        return asdict(self)


def _text(content: Any) -> str:
    """把消息内容规整成纯文本。

    content 可能是字符串,也可能是分块列表(多模态/带思考块的模型会这样返回)。
    不处理的话,答案会变成 "[{'type': 'text', ...}]" 这种字符串,
    关键点一个都匹配不上 —— 判分全错,却看不出原因。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        return "".join(parts)
    return str(content)


def extract(case_id: str, state: dict, latency_ms: int) -> Trace:
    """从 agent 返回的 state 里抽轨迹。

    state["messages"] 的形状(ReAct 循环的产物):
        HumanMessage        用户问题
        AIMessage           带 tool_calls → 说明它决定调工具
        ToolMessage         工具返回
        ...(可能循环多轮)
        AIMessage           不带 tool_calls → 这就是最终答案
    """
    tr = Trace(case_id=case_id, latency_ms=latency_ms)
    msgs = state.get("messages", []) or []

    # 工具返回按 tool_call_id 索引,好和发起它的调用配对。
    results: dict[str, str] = {}
    for m in msgs:
        if getattr(m, "type", "") == "tool":
            body = _text(getattr(m, "content", ""))
            results[getattr(m, "tool_call_id", "")] = body[:300]

    last_ai_text = ""
    for m in msgs:
        if getattr(m, "type", "") != "ai":
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            tr.tool_calls.append(
                ToolCall(
                    name=tc.get("name", ""),
                    args=tc.get("args", {}) or {},
                    result_preview=results.get(tc.get("id", ""), ""),
                )
            )
        usage = getattr(m, "usage_metadata", None) or {}
        tr.input_tokens += usage.get("input_tokens", 0)
        tr.output_tokens += usage.get("output_tokens", 0)
        # 最终答案 = 最后一条【不带工具调用】的 AI 消息。
        # 不能简单取 messages[-1]:带 HITL 时最后一条可能是中断标记。
        if not (getattr(m, "tool_calls", None) or []):
            txt = _text(getattr(m, "content", ""))
            if txt.strip():
                last_ai_text = txt

    # 工具执行失败的标记由 ToolErrorHandlingMiddleware 写入(共用同一个常量,
    # 不在这里硬编码措辞 —— 那样改一个字评测就悄悄失效了)。
    from ruixue_agent.agents.middlewares import TOOL_FAILURE_MARKER

    tr.failed_tools = [c.name for c in tr.tool_calls if TOOL_FAILURE_MARKER in c.result_preview]

    tr.answer = last_ai_text
    # LangGraph 用 __interrupt__ 表示"停下来等人确认"。这不是失败,但也不是完成。
    if state.get("__interrupt__"):
        tr.interrupted = True
    return tr


class Timer:
    """测一次运行的墙上时间。延迟也是评测指标 —— 慢到用户等不了,答对也没用。"""

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.perf_counter() - self._t) * 1000)
        return False
