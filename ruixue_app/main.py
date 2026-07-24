"""FastAPI 服务:把 agent 包成 HTTP 接口(POST /chat)。

起服务:uv run uvicorn ruixue_app.main:app --reload
测试:  浏览器开 http://127.0.0.1:8000/docs —— 点右上角 Authorize 填 API Key 再测。
前置:  docker 起着(agent 要调 RAG)。
"""

import json
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ruixue_agent.agents import create_ruixue_agent
from ruixue_app.auth import get_current_user

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="瑞雪地膜智能助手")

# 启动时建一次 agent,所有请求复用。建 agent 含加载模型,贵 —— 别每次请求都建。
_agent = create_ruixue_agent()


# ── 限流(Rate Limiting)────────────────────────────────────────
def _rate_key(request: Request) -> str:
    """限流维度:按【谁】来限流(每个 key 一个独立的额度池)。"""
    # ===== (你写一行)=====
    # 优先按 API Key(≈按用户,每个用户各自额度),取不到就按来源 IP:
    #   return request.headers.get("X-API-Key") or get_remote_address(request)
    return request.headers.get("X-API-Key") or get_remote_address(request)


limiter = Limiter(key_func=_rate_key)  # 限流器,按 _rate_key 区分不同调用方
app.state.limiter = limiter  # 挂到 app 上(slowapi 要求)
# 超过限额时,自动返回 429 Too Many Requests(不用我们自己写)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── 全局异常兜底(错误脱敏)────────────────────────────────────
logger = logging.getLogger("ruixue.app")


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """未预料的异常统一兜底:任何没被单独处理的异常,最后都掉进这里。

    原则:详情记【服务端日志】(供你排查),给用户返回【脱敏的通用消息】——
    DB 连接串、文件路径、堆栈,绝不能出现在返回给用户的响应里(泄露内部结构=送攻击者情报)。
    注:401/422/429 这些是"有意的"错误,由各自的处理器返回正确状态码,不会走到这里。
    """
    # ===== (你写两行)=====
    # 1. 把异常详情(含堆栈)记到服务端日志——logger.exception 会自动带上堆栈:
    #      logger.exception("未处理异常: %s", exc)
    # 2. 给用户返回脱敏的通用错误(注意:content 里【绝不能】放 exc 的内容!):
    #      return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试"})
    logger.exception("未处理异常: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"}
    )


# ── 请求 / 响应模型 ────────────────────────────────────────────
class ChatRequest(BaseModel):
    """客户端 POST 过来的 JSON。"""
    # ===== (你写)=====
    # 给两个字段加【长度上限】(Field 声明"这个字段最长多少",防超长输入烧 token)。
    # 把下面两行改成:
    #   thread_id: str = Field(..., max_length=64)      # 会话ID,短; ... 表示必填
    #   message: str = Field(..., max_length=2000)      # 用户消息,最长 2000 字
    thread_id: str = Field(..., max_length=64)
    message: str = Field(..., max_length=2000)


class ChatResponse(BaseModel):
    """返回给客户端的 JSON。"""
    answer: str


# ── 端点 ──────────────────────────────────────────────────────
@app.post("/chat")
@limiter.limit("20/minute")  # 每个 key(用户)每分钟最多 20 次,超了自动 429。可按需调。
def chat(
    request: Request,  # slowapi 靠它拿限流 key,必须有这个参数
    req: ChatRequest,
    user_id: str = Depends(get_current_user),  # 先认证,拿到当前用户
) -> ChatResponse:
    thread_id = f"{user_id}:{req.thread_id}"  # 命名空间隔离:用户只能碰自己的对话
    config = {"configurable": {"thread_id": thread_id}}
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]}, config=config
    )
    return ChatResponse(answer=result["messages"][-1].content)


@app.post("/chat/stream")
@limiter.limit("20/minute")
def chat_stream(
    request: Request,  # slowapi 要求
    req: ChatRequest,
    user_id: str = Depends(get_current_user),
):
    """流式端点:边生成边返回(SSE)。返回持续字节流,不是一次性 JSON。"""
    thread_id = f"{user_id}:{req.thread_id}"
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        for chunk, meta in _agent.stream(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            stream_mode="messages",
        ):
            reasoning = chunk.additional_kwargs.get("reasoning_content")
            if reasoning:
                yield f"data: {json.dumps({'type': 'thinking', 'text': reasoning}, ensure_ascii=False)}\n\n"
            if chunk.content:
                yield f"data: {json.dumps({'type': 'answer', 'text': chunk.content}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
