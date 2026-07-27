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
                    f"工具 {name} 执行失败({safe_reason}),请告知用户该功能暂时不可用,不要编造结果。"
                ),
                tool_call_id=request.tool_call["id"],
            )
