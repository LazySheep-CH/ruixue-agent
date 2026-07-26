"""可观测性:请求追踪(request_id)+ 结构化日志。

一句话:给每个请求发一个唯一编号(像快递单号),让它出现在【这个请求产生的每一条
日志】里,并回写到响应头。这样线上排查时,凭一个编号就能捞出某次请求的完整日志。

三个零件:
  1. request_id_var —— 一个 ContextVar(每请求独立的"餐盘",并发不串)
  2. RequestIdLogFilter —— 给每条日志盖上当前请求的编号
  3. RequestIdMiddleware —— 每个请求:生成编号 -> 存进餐盘 -> 回写响应头
"""

import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# 每个请求独立的一份值(并发下互不覆盖)。没有请求时(比如启动日志)默认 "-"。
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdLogFilter(logging.Filter):
    """在每条日志记录上盖一个 request_id 字段,供格式串 %(request_id)s 使用。

    logging.Filter 本职是"决定这条日志放不放行",但也常被借用来【给日志加字段】——
    我们就是借它,在每条日志被输出前,把当前请求的编号塞进去。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # ===== (你写一行)=====
        # 把当前请求的编号,盖到这条日志记录上(取当前"餐盘"里的值):
        #   record.request_id = request_id_var.get()
        record.request_id = request_id_var.get()
        return True  # True = 放行这条日志(我们只借它加字段,不丢弃任何日志)


def configure_logging(level: int = logging.INFO) -> None:
    """配置全局日志:输出格式带上 [req=...],并挂上上面的 Filter。

    替代原来的 logging.basicConfig —— 那个格式里没有 request_id。
    """
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdLogFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [req=%(request_id)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()  # 清掉可能已存在的 handler,避免日志打两遍
    root.addHandler(handler)
    root.setLevel(level)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """每个 HTTP 请求进来时:确定一个 request_id,全程带着,并回写响应头。

    中间件(middleware)在这里 = "每个请求进出都要穿过的关卡"。
    和 agent 的中间件是同一个思想,只是这层套在【HTTP 请求】外面。
    """

    async def dispatch(self, request: Request, call_next):
        # ===== (你写一行)=====
        # 优先用调用方传来的 X-Request-ID(便于跨服务把同一次调用串起来),
        # 没有就自己生成一个短随机串。uuid4().hex 是 32 位十六进制,取前 12 位够用:
        #   rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)  # 存进"餐盘";token 用于稍后精确还原
        try:
            response = await call_next(request)  # 放行去后续处理(端点在这中间执行)
            response.headers["X-Request-ID"] = rid  # 回写响应头,调用方也能看到编号
            return response
        finally:
            # 请求结束,把餐盘还原(reset)。避免这份值"泄漏"到复用该线程的下个任务。
            request_id_var.reset(token)
