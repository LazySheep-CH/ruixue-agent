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
from langchain_core.messages import ToolMessage

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


class ToolErrorHandlingMiddleware(AgentMiddleware):
    """工具执行失败时优雅降级:不让单个工具异常拖垮整个对话。

    工具会失败(RAG 断连、坏输入抛异常…)。若异常一路往上冒,整个请求崩、
    用户得到 500。这里在 wrap_tool_call 里兜住异常,记 ERROR 日志,并返回一条
    "工具失败"的 ToolMessage —— agent 收到它,能回一句"知识库暂时不可用",
    而不是整个崩掉。这就是"故障隔离 + 优雅降级"。
    """

    def wrap_tool_call(self, request, handler):
        # ============================================================
        #   (你写函数体):
        #   1. try: 正常执行并返回
        #          return handler(request)
        #   2. except Exception as e:  兜住工具抛出的任何异常
        #        a. 记 ERROR 日志(注意用 logger.error,级别比 info 高):
        #           logger.error("工具 %s 失败: %s", request.tool_call["name"], e)
        #        b. 返回一条错误 ToolMessage,让 agent 知道工具挂了、能优雅回应。
        #           ToolMessage 必须带 tool_call_id(对应哪次调用),从 request 取:
        #           return ToolMessage(
        #               content=f"工具执行失败: {e}",
        #               tool_call_id=request.tool_call["id"],
        #           )
        # ============================================================
        try:
            return handler(request)
        except Exception as e:
            logger.error("工具 %s 失败: %s", request.tool_call["name"], e)
            return ToolMessage(
                content=f"工具执行失败: {e}",
                tool_call_id=request.tool_call["id"],
            )