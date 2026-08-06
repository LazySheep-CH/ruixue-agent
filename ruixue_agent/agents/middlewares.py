"""Agent 中间件:横切关注点(计时/日志/错误处理…),套在模型/工具调用外,不动业务逻辑。

create_agent(middleware=[...]) 接收它们。参考架构 的中间件链也是这么叠的。
钩子(和你 mini-harness 写过的一样):
    before_model(state, runtime)      模型调用前
    after_model(state, runtime)       模型调用后
    wrap_tool_call(request, handler)  包裹工具执行(handler(request) 才是真正执行)
"""

import logging
import time

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage

from ruixue_agent.guardrails.injection import REINFORCE_NOTICE, detect_injection
from ruixue_agent.skills import render_skills

logger = logging.getLogger("ruixue.agent")


class TimingLoggingMiddleware(AgentMiddleware):
    """记录每次工具调用的名称与耗时 —— 可观测性的起点。

    有了它,一条请求里 agent 调了哪些工具、各花多久,日志里一目了然,
    排查"为什么这次慢""RAG 是不是超时"就有据可依。
    """

    def wrap_tool_call(self, request, handler):
        # ============================================================
        #   (你写函数体):
        #   1. 记开始时间:      t0 = time.perf_counter()
        #   2. 取工具名:        name = request.tool_call["name"]
        #   3. 执行工具:        result = handler(request)   ← 这一句才真正执行工具
        #   4. 算耗时并记日志:   ms = (time.perf_counter() - t0) * 1000
        #                       logger.info("工具 %s 耗时 %.0fms", name, ms)
        #   5. return result
        # ============================================================
        t0 = time.perf_counter()
        name = request.tool_call["name"]
        result = handler(request)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("工具 %s 耗时 %.0fms", name, ms)
        return result


class PromptInjectionGuardMiddleware(AgentMiddleware):
    """提示注入防御:检测用户消息里的注入企图,记日志并重申系统规则。

    为什么【不直接拒绝】:正则判定必然有误差,把真实用户挡在门外的代价,
    远大于让一次注入企图通过(何况本 agent 的工具全是只读的,破坏上限有限)。
    故采取"检测 → 告警留痕 → 在上下文里重申身份与边界"的策略 ——
    既不误伤,又让模型带着防御提示去回答。

    留痕很重要:安全事件要能在日志里查得到(配合 request_id 可定位到具体用户/请求)。
    """

    def before_model(self, state, runtime):
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        # 只检最后一条【用户】消息:历史消息已检过,工具返回的内容由 RAG 侧边界隔离
        if type(last).__name__ != "HumanMessage":
            return None
        threats = detect_injection(str(last.content))
        if not threats:
            return None
        logger.warning("检测到疑似提示注入: %s", threats)
        # 追加一条系统消息重申边界(不改用户原文,便于排查时看到真实输入)
        return {"messages": [SystemMessage(content=REINFORCE_NOTICE)]}


class SkillInjectionMiddleware(AgentMiddleware):
    """按场景注入【作业规程(技能)】:告诉模型"这类问题该怎么做"。

    工具解决"能做什么",技能解决"该怎么做"——领域经验(先看生育期、三项指标
    如何权衡、有哪些坑)写死在系统提示会越堆越长、稀释注意力;写进代码更糟,
    因为它要由领域专家反复迭代。故做成 skills/*.md,按关键词命中才注入。

    只在【首轮】注入:同一会话里规程不必反复重申,省 token。
    """

    def before_model(self, state, runtime):
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if type(last).__name__ != "HumanMessage":
            return None
        # 首轮判定:此前没有任何模型回复 → 这是本会话第一个问题
        if any(type(m).__name__ == "AIMessage" for m in messages[:-1]):
            return None
        text = render_skills(str(last.content))
        if not text:
            return None
        logger.info("注入作业规程(技能)")
        return {"messages": [SystemMessage(content=text)]}


# 工具失败消息的机器可读标记。
#
# 为什么要单独抽一个常量:评测需要区分【工具挂了】和【模型没选对工具】——
# 前者是环境问题(修 Milvus),后者是能力问题(改提示词/工具描述)。两者混在
# 一起会把优化方向带偏。靠在评测里硬编码这句话的措辞太脆(改个字就失效),
# 所以在产生它的地方定义常量,消费方(eval/trace.py)引用同一个。
TOOL_FAILURE_MARKER = "[TOOL_FAILED]"


class ToolErrorHandlingMiddleware(AgentMiddleware):
    """工具执行失败时优雅降级:不让单个工具异常拖垮整个对话。

    工具会失败(RAG 断连、坏输入抛异常…)。若异常一路往上冒,整个请求崩、
    用户得到 500。这里在 wrap_tool_call 里兜住异常,记 ERROR 日志,并返回一条
    "工具失败"的 ToolMessage —— agent 收到它,能回一句"知识库暂时不可用",
    而不是整个崩掉。这就是"故障隔离 + 优雅降级"。
    """

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            name = request.tool_call["name"]
            # 完整详情(含堆栈)只进【服务端日志】,给你排查用。
            # logger.exception 比 logger.error 多带堆栈,能看到错在哪一行。
            logger.exception("工具 %s 失败", name)
            # 【错误脱敏】只把异常【类型名】给模型,不给 str(e) 的具体内容 ——
            # 原始文字可能含内网 IP、数据库地址等,进了模型上下文可能被复述给用户。
            # 给类型是为了让模型能贴切回应(连不上 vs 参数错),又不泄露任何细节。
            safe_reason = type(e).__name__
            return ToolMessage(
                content=(
                    f"{TOOL_FAILURE_MARKER} 工具 {name} 执行失败({safe_reason}),"
                    f"请告知用户该功能暂时不可用,不要编造结果。"
                ),
                tool_call_id=request.tool_call["id"],
            )
