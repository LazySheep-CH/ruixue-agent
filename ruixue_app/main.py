"""FastAPI 服务:把 agent 包成 HTTP 接口(POST /chat)。

起服务:uv run uvicorn ruixue_app.main:app --reload
测试:  浏览器开 http://127.0.0.1:8000/docs —— 点右上角 Authorize 填 API Key 再测。
前置:  docker 起着(agent 要调 RAG)。
"""

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ruixue_agent.agents import create_ruixue_agent
from ruixue_app.auth import get_current_user
from ruixue_app.observability import RequestIdMiddleware, configure_logging
from ruixue_app.routes import auth as auth_routes

# 用带 request_id 的结构化日志格式,替代默认的 logging.basicConfig。
configure_logging()

# agent 先占位;真正在服务【启动】时才建(见下面的 lifespan)。
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务的生命周期钩子:yield 之前 = 启动时跑一次;yield 之后 = 关闭时跑一次。

    为什么要它:之前 `_agent = create_ruixue_agent()` 写在模块顶层,
    意味着【任何人 import 这个文件】都会立刻建 agent(连库、加载模型)。
    跑测试、被别的脚本导入,都会莫名其妙连库。放进 lifespan 后,
    只有真正【起服务】时才建,一次,之后所有请求复用同一个 agent。
    """
    global _agent  # 要在函数里【改】模块全局变量,必须先声明 global
    # ===== (你写一行)=====
    # 服务启动:建一次 agent,存进 _agent 给所有请求复用:
    #   _agent = create_ruixue_agent()
    _agent = create_ruixue_agent()
    yield
    # ↑ yield 前是"启动";yield 后是"关闭"。关闭时的清理写在这下面
    #   (目前没有要清理的资源,先留空)。


app = FastAPI(title="瑞雪地膜智能助手", lifespan=lifespan)

# 请求追踪:每个请求发一个 request_id,贯穿日志、回写响应头。
# 这是【HTTP 层】的中间件(套在整个请求外),和 agent 里的中间件是同一思想、不同层。
app.add_middleware(RequestIdMiddleware)

# 跨域(CORS):前端是独立工程(frontend/),开发期由 Next 代理到这里,属同源;
# 但前端一旦独立部署到别的域名,浏览器就会拦跨域请求 —— 故显式放行。
# 【安全】只放行白名单来源,不用 "*"(带自定义头 X-API-Key 时 "*" 也不合法)。
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],  # 让前端能读到请求编号,便于报障
)


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


# 认证路由:注册 / 登录 / 查当前用户
app.include_router(auth_routes.router)


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
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试"})


# ── 健康检查(可运维)──────────────────────────────────────────
# 两种探针,别混:
#   存活 /health       = "进程还活着吗" —— 挂了就重启它。故意【不查】数据库。
#   就绪 /health/ready = "能真正干活吗" —— 依赖(数据库)连不上时,虽活着但不该接流量。
# 负载均衡 / K8s 靠这两个探针决定:要不要重启、要不要把流量切走。
@app.get("/health")
def health():
    """存活探针:只证明进程能响应,不碰任何外部依赖。"""
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    """就绪探针:数据库连得上才算 ready;连不上 -> 503,让上游把流量切走。"""
    from sqlalchemy import text

    from ruixue_agent.persistence.engine import get_engine

    try:
        # ===== (你写)=====
        # 跑一句最轻的查询证明数据库是通的,通了就回 ready:
        #   with get_engine().connect() as conn:
        #       conn.execute(text("SELECT 1"))
        #   return {"status": "ready"}
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        # 连不上不是"服务器 bug",是"暂时没准备好" -> 用 503(Service Unavailable),
        # 不是 500。完整原因进日志,响应只给一个状态词。
        logger.exception("就绪检查失败:数据库连不上")
        return JSONResponse(status_code=503, content={"status": "not_ready"})


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
    result = _agent.invoke({"messages": [{"role": "user", "content": req.message}]}, config=config)
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
        announced: set[str] = set()  # 已通知"开始"的工具,避免流式分片重复推
        seen_tools: set[str] = set()
        # _meta 用不到,用下划线开头表示"我知道它在,但故意不用"(避免 lint 报未用变量)
        for chunk, _meta in _agent.stream(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            stream_mode="messages",
        ):
            # 工具【执行完毕】:ToolMessage 带工具名。只报"哪个工具跑完了",
            # 不把工具返回的原文推给用户(那是给模型看的中间结果,会和正式回答重复)。
            if isinstance(chunk, ToolMessage):
                if chunk.name and chunk.name not in seen_tools:
                    seen_tools.add(chunk.name)
                payload = json.dumps(
                    {"type": "tool_end", "name": chunk.name or ""}, ensure_ascii=False
                )
                yield f"data: {payload}\n\n"
                continue

            if not isinstance(chunk, AIMessageChunk):
                continue

            # 工具【开始调用】:模型决定用某工具时,tool_call_chunks 里会陆续带出名字。
            # 前端据此显示"正在检索知识库…"这类进度 —— agent 产品的核心体验:
            # 让用户看见它在做什么,而不是干等一个黑盒。
            for tc in chunk.tool_call_chunks or []:
                name = tc.get("name")
                if name and name not in announced:
                    announced.add(name)
                    payload = json.dumps({"type": "tool_start", "name": name}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            reasoning = chunk.additional_kwargs.get("reasoning_content")
            if reasoning:
                payload = json.dumps({"type": "thinking", "text": reasoning}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            if chunk.content:
                payload = json.dumps({"type": "answer", "text": chunk.content}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
