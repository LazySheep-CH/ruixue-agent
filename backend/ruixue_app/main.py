"""FastAPI 服务:把 agent 包成 HTTP 接口(POST /chat)。

起服务:uv run uvicorn ruixue_app.main:app --reload
测试:  浏览器开 http://127.0.0.1:8000/docs —— 点右上角 Authorize 填 API Key 再测。
前置:  docker 起着(agent 要调 RAG)。
"""

import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ruixue_agent.agents import create_ruixue_agent
from ruixue_app import mcp_server, metrics, report, runs
from ruixue_app.auth import get_current_user  # 查询类端点只需认证,不消耗配额
from ruixue_app.observability import RequestIdMiddleware, configure_logging
from ruixue_app.quota import enforce_quota
from ruixue_app.routes import auth as auth_routes

# 用带 request_id 的结构化日志格式,替代默认的 logging.basicConfig。
configure_logging()

# agent 先占位;真正在服务【启动】时才建(见下面的 lifespan)。
_agent = None

# MCP server:把我们的工具暴露给【外部 agent】用。默认关闭,见 mcp_server 模块。
# 在这里建(模块级)而不是 lifespan 里:挂载子应用必须在 app 建好后立刻做,
# 而 lifespan 是启动时才跑,那时路由表已经定型了。
_mcp = mcp_server.build_server()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务的生命周期钩子:yield 之前 = 启动时跑一次;yield 之后 = 关闭时跑一次。

    为什么要它:之前 `_agent = create_ruixue_agent()` 写在模块顶层,
    意味着【任何人 import 这个文件】都会立刻建 agent(连库、加载模型)。
    跑测试、被别的脚本导入,都会莫名其妙连库。放进 lifespan 后,
    只有真正【起服务】时才建,一次,之后所有请求复用同一个 agent。
    """
    global _agent
    _agent = create_ruixue_agent()
    # 清理残留:进程上次被 kill 时,后台任务没了但库里还写着 running,
    # 用户会一直等一个永远不会完成的运行。启动时统一标记为失败。
    runs.reap_stale()
    # 记忆库幂等建表(和文档库一样,启动时确保存在)
    try:
        from ruixue_agent.memory.store import ensure_collection

        ensure_collection()
    except Exception:
        logging.getLogger("ruixue.app").warning("记忆库初始化失败,长期记忆将不可用", exc_info=True)

    # MCP server 有【自己的 lifespan】(会话管理器)。挂进来的子应用不会自动跑它的
    # lifespan —— 必须在这里手动进上下文,否则请求会报 "task group is not initialized"。
    # 这是挂载 ASGI 子应用最常见的坑:路由通了、启动不报错,一发请求才崩。
    async with contextlib.AsyncExitStack() as stack:
        if _mcp is not None:
            await stack.enter_async_context(_mcp.session_manager.run())
        yield
    # ↑ yield 前是"启动";yield 后是"关闭"。
    # 停机:停收新活 → 等在途 agent 跑完 → 剩下的立刻标记失败。
    # 不做的话,每次重新部署都会把正在跑的运行连人带钱一起丢掉,
    # 而且用户要对着转圈等 15 分钟才被 reap_stale 清理。详见 runs.shutdown。
    # 放线程里跑:shutdown 内部要轮询等待(同步阻塞)最多 45 秒。
    # 直接在 async 函数里调会【卡死事件循环】—— 连正在收尾的 SSE 都发不出去。
    await asyncio.to_thread(runs.shutdown)


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
    # DELETE 是给 /datasets/{id} 用的。少写一个方法不会报错,
    # 只会让前端的删除按钮在跨域部署下静默失败(预检不通过)。
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],  # 让前端能读到请求编号,便于报障
)


# ── 限流(Rate Limiting)────────────────────────────────────────
def _rate_key(request: Request) -> str:
    """限流维度:按【谁】来限流(每个 key 一个独立的额度池)。"""
    # 优先按 API Key(≈按用户,每个用户各自额度),取不到就按来源 IP
    return request.headers.get("X-API-Key") or get_remote_address(request)


# 计数存哪里:必须是跨进程共享的。
# 上线一般起多个 worker(uvicorn --workers N),若计数各存各的内存,
# N 个 worker 就等于限额 ×N —— 限流形同虚设。故用 Redis 做共享存储。
# 没配 REDIS_URL 时退回内存(仅适合单 worker 的本地开发),并明确告警。
_REDIS_URL = os.getenv("REDIS_URL", "")
if _REDIS_URL:
    limiter = Limiter(key_func=_rate_key, storage_uri=_REDIS_URL)
else:
    logging.getLogger("ruixue.app").warning(
        "未配置 REDIS_URL,限流退回【单进程内存】—— 多 worker 部署时限额会被放大 N 倍,"
        "生产环境务必配置 REDIS_URL"
    )
    limiter = Limiter(key_func=_rate_key)
app.state.limiter = limiter  # 挂到 app 上(slowapi 要求)
# 超过限额时,自动返回 429 Too Many Requests(不用我们自己写)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# 认证路由:注册 / 登录 / 查当前用户
app.include_router(auth_routes.router)

# MCP server 挂载。AuthGate 包在外层 —— 子应用不吃 FastAPI 的 Depends,
# 鉴权必须在 ASGI 这一层做,否则是【静默无鉴权】(接口能通,只是没人验身份)。
if _mcp is not None:
    app.mount(mcp_server.MOUNT_PATH, mcp_server.AuthGate(_mcp.streamable_http_app()))


# ── 全局异常兜底(错误脱敏)────────────────────────────────────
logger = logging.getLogger("ruixue.app")


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """未预料的异常统一兜底:任何没被单独处理的异常,最后都掉进这里。

    原则:详情记【服务端日志】(供你排查),给用户返回【脱敏的通用消息】——
    DB 连接串、文件路径、堆栈,绝不能出现在返回给用户的响应里(泄露内部结构=送攻击者情报)。
    注:401/422/429 这些是"有意的"错误,由各自的处理器返回正确状态码,不会走到这里。
    """
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


@app.get("/metrics")
def prometheus_metrics(user_id: str = Depends(get_current_user)) -> Response:
    """运行指标(Prometheus 文本格式)。

    为什么要鉴权:
    这些指标会告诉外人:我们有多少用户在用、失败率多少、容量上限是多少、
    哪个依赖挂了。对攻击者来说这是一张现成的作战地图 ——
    知道容量上限就知道打多少并发能把你压垮,知道失败率就知道什么时候值得再试。

    Prometheus 支持在 scrape 配置里带 header,所以加鉴权不影响接入:
        authorization: [{ credentials: "<API Key>", type: "Bearer" }]

    (业界也有靠网络隔离而不鉴权的做法,但那前提是内网真的隔离 ——
     我们是单机 docker 部署,端口一暴露就没有内网了。)
    """
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/health/ready")
def health_ready():
    """就绪探针:数据库连得上才算 ready;连不上 -> 503,让上游把流量切走。"""
    from sqlalchemy import text

    from ruixue_agent.persistence.engine import get_engine

    try:
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

    # 长度上限:防超长输入烧 token;422 由框架自动返回
    thread_id: str = Field(..., max_length=64)
    message: str = Field(..., max_length=2000)


class ChatResponse(BaseModel):
    """返回给客户端的 JSON。"""

    answer: str
    # 需人工批准时:answer 为空,pending 给出待批准的操作(见 /chat/resume)
    pending: list[dict] | None = None


class ResumeRequest(BaseModel):
    """批准/拒绝一个待确认的操作。"""

    thread_id: str = Field(..., max_length=64)
    approve: bool = Field(..., description="True=批准执行,False=拒绝")


def _to_response(result: dict) -> ChatResponse:
    """把 agent 的返回统一成 ChatResponse。

    开启人工批准(Human-in-the-Loop)后,agent 可能在工具执行前【暂停】,
    此时返回里没有最终答案,而是 __interrupt__ —— 需要客户端确认后调
    /chat/resume 继续。不开启时该分支永远不会走到。
    """
    interrupts = result.get("__interrupt__")
    if interrupts:
        requests = interrupts[0].value.get("action_requests", [])
        return ChatResponse(
            answer="",
            pending=[{"tool": r.get("name"), "args": r.get("args")} for r in requests],
        )
    return ChatResponse(answer=result["messages"][-1].content)


# ── 端点 ──────────────────────────────────────────────────────
@app.post("/chat")
@limiter.limit("20/minute")  # 每个 key(用户)每分钟最多 20 次,超了自动 429。可按需调。
def chat(
    request: Request,  # slowapi 靠它拿限流 key,必须有这个参数
    req: ChatRequest,
    # enforce_quota 内部已含认证(Depends(get_current_user)),并额外消耗每日配额。
    # 只挂在【会花钱】的端点上;健康检查、查土壤这类不花钱的不挂。
    user_id: str = Depends(enforce_quota),
) -> ChatResponse:
    thread_id = f"{user_id}:{req.thread_id}"  # 命名空间隔离:用户只能碰自己的对话
    config = {"configurable": {"thread_id": thread_id}}
    result = _agent.invoke({"messages": [{"role": "user", "content": req.message}]}, config=config)
    return _to_response(result)


@app.post("/chat/resume")
@limiter.limit("20/minute")
def chat_resume(
    request: Request,
    req: ResumeRequest,
    user_id: str = Depends(enforce_quota),
) -> ChatResponse:
    """对 /chat 返回的 pending 操作作出批准/拒绝,让对话继续。

    thread_id 同样做命名空间隔离 —— 用户只能批准【自己】会话里的操作,
    否则就能通过猜 thread_id 去批准别人的待确认操作(越权)。
    """
    from langgraph.types import Command

    config = {"configurable": {"thread_id": f"{user_id}:{req.thread_id}"}}
    decision = "approve" if req.approve else "reject"
    result = _agent.invoke(Command(resume={"decisions": [{"type": decision}]}), config=config)
    return _to_response(result)


def _execute_run(run_id: str, thread_id: str, message: str) -> None:
    """在后台线程里跑 agent,把过程事件发到该 Run 的流上。

    这个函数【不绑请求生命周期】—— 客户端断了它照跑完,结果落库。
    这正是"刷新页面不丢结果、钱不白花"的关键。
    """
    config = {"configurable": {"thread_id": thread_id}}
    announced: set[str] = set()  # 已通知"开始"的工具,避免流式分片重复推
    answer_parts: list[str] = []
    try:
        for chunk, _meta in _agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode="messages",
        ):
            # 工具【执行完毕】:只报"哪个工具跑完了",不推工具返回的原文
            # (那是给模型看的中间结果,推给用户会和正式回答重复)。
            if isinstance(chunk, ToolMessage):
                runs.publish(run_id, {"type": "tool_end", "name": chunk.name or ""})
                continue
            if not isinstance(chunk, AIMessageChunk):
                continue
            # 工具【开始调用】:让用户看见 agent 在做什么,而不是干等黑盒。
            for tc in chunk.tool_call_chunks or []:
                name = tc.get("name")
                if name and name not in announced:
                    announced.add(name)
                    runs.publish(run_id, {"type": "tool_start", "name": name})
            if reasoning := chunk.additional_kwargs.get("reasoning_content"):
                runs.publish(run_id, {"type": "thinking", "text": reasoning})
            if chunk.content:
                answer_parts.append(str(chunk.content))
                runs.publish(run_id, {"type": "answer", "text": chunk.content})
    except Exception as e:
        # 详情进日志,给用户的只有脱敏短语(和全局异常兜底同一原则)
        logger.exception("运行 %s 失败", run_id)
        runs.finish_run(run_id, error=f"运行失败({type(e).__name__})")
        runs.publish(run_id, {"type": "error", "text": "生成失败,请重试"})
        runs.publish(run_id, {"type": "done"})
        return

    answer = "".join(answer_parts)
    runs.finish_run(run_id, answer=answer)
    runs.publish(run_id, {"type": "done"})

    # 长期记忆写入放在【最后】,而且在 done 事件之后。
    #
    # 顺序是刻意的:用户已经拿到答案、连接已经可以关了,记忆抽取多花的
    # 一两秒完全不占他的等待时间。反过来放在 finish_run 之前的话,
    # 每个用户都要为一件他根本感知不到的事多等一会。
    #
    # 而且它整个包在 try 里:记忆是锦上添花,抽取或存储失败绝不能
    # 让一次【已经成功】的运行看起来像失败了。
    try:
        _remember_async(thread_id, message, answer, run_id)
    except Exception:
        logger.warning("长期记忆写入失败(不影响本次回答)", exc_info=True)


def _remember_async(thread_id: str, question: str, answer: str, run_id: str) -> None:
    """抽取并存入长期记忆。

    thread_id 形如 "alice:t1",前缀就是 user_id(命名空间隔离的副产品)。
    拿不到用户身份就不记 —— 记忆按用户隔离,没身份的记忆无处安放,
    硬塞给一个默认用户等于让所有人共用一份记忆,是数据泄露。
    """
    user_id = thread_id.split(":", 1)[0] if ":" in thread_id else ""
    if not user_id or not answer.strip():
        return
    from ruixue_agent.memory import remember
    from ruixue_agent.memory.extract import extract_facts

    facts = extract_facts(question, answer)
    if not facts:
        return  # 没有值得长期记住的 —— 这是常态,不是失败
    n = remember(user_id, facts, run_id=run_id)
    if n:
        logger.info("run=%s 新增长期记忆 %d 条", run_id, n)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _stream_run(run_id: str, from_start: bool):
    """把某个 Run 的事件流转成 SSE。from_start=True 时【从头补发】(重连场景)。"""

    def gen():
        # 先把 run_id 告诉客户端 —— 前端存下它,断线后才能重连
        yield _sse({"type": "run", "run_id": run_id})
        last = "0-0" if from_start else "$"
        idle = 0
        while True:
            events = runs.read_events(run_id, last_id=last, block_ms=5000)
            if not events:
                idle += 1
                if idle > 24:  # 连续 2 分钟没有新事件 —— 判定异常,别让连接空挂
                    yield _sse({"type": "error", "text": "运行超时"})
                    return
                yield ": keepalive\n\n"  # SSE 注释行:防代理把空闲连接掐掉
                continue
            idle = 0
            for eid, ev in events:
                last = eid
                yield _sse(ev)
                if ev.get("type") == "done":
                    return

    # X-Accel-Buffering 是【响应头】:由后端贴在响应上、代理读它。
    # 曾配反过 —— 写成 nginx 的 proxy_set_header(那是贴在"寄给后端的请求"上,
    # 后端看了也没用),流式一直只靠 proxy_buffering off 单锁在撑。
    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
    )


@app.post("/chat/stream")
@limiter.limit("20/minute")
def chat_stream(
    request: Request,  # slowapi 要求
    req: ChatRequest,
    user_id: str = Depends(enforce_quota),
):
    """发起对话:创建 Run → 后台跑 agent → SSE 推送过程事件。

    与旧版的区别:agent 不再跑在请求里。客户端断开只是断了这条 SSE,
    agent 继续跑完并落库;客户端可用 /chat/runs/{run_id}/stream 重连补看。
    """
    thread_id = f"{user_id}:{req.thread_id}"  # 命名空间隔离
    run_id = runs.create_run(user_id, thread_id, req.message)
    try:
        runs.start_background(run_id, _execute_run, run_id, thread_id, req.message)
    except runs.CapacityError:
        # 容量满:明确告诉用户"现在忙",而不是让他排一个看不到头的队,
        # 也不是抛 500(那会让人以为是 bug)。503 是"暂时不可用"的正确语义。
        runs.finish_run(run_id, error="系统繁忙,请稍后重试")
        raise HTTPException(status_code=503, detail="当前请求过多,请稍后重试") from None
    return _stream_run(run_id, from_start=True)


@app.get("/chat/runs/{run_id}")
def get_run_status(run_id: str, user_id: str = Depends(get_current_user)) -> dict:
    """查询一次运行的状态与结果 —— 刷新页面后用它把答案取回来。"""
    row = runs.get_run(run_id, user_id)  # 内部校验归属,防越权看别人的对话
    if row is None:
        raise HTTPException(status_code=404, detail="运行不存在")
    return {
        "run_id": row.run_id,
        "status": row.status,
        "question": row.question,
        "answer": row.answer,
        "error": row.error,
    }


@app.get("/chat/runs/{run_id}/report.pdf")
def download_run_report(run_id: str, user_id: str = Depends(get_current_user)):
    """把一次运行导出成 PDF 报告。

    为什么是接口而不是 agent 的工具:见 ruixue_app/report.py 的模块说明 ——
    一句话是工具要保持只读,让 agent 能写文件会同时引入路径逃逸、
    磁盘清理和"提示注入诱导写文件"三个新问题。

    归属校验复用 runs.get_run(run_id, user_id):猜到别人的 run_id 也拿不到,
    和查状态、重连走的是同一道门 —— 新开的下载口不另起一套鉴权。
    """
    row = runs.get_run(run_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="运行不存在")

    data = report.ReportData(
        run_id=row.run_id,
        question=row.question,
        answer=row.answer or "",
        created_at=row.created_at,
        status=row.status,
    )
    pdf = report.render_pdf(data)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            # filename 只含日期和 run_id 前 8 位,不拼用户提问 —— 见 report.filename_for
            "Content-Disposition": f'attachment; filename="{report.filename_for(data)}"',
            "Content-Length": str(len(pdf)),
        },
    )


@app.get("/chat/runs/{run_id}/stream")
def resume_run_stream(run_id: str, user_id: str = Depends(get_current_user)):
    """重连:从头补发该 Run 已产生的事件,并继续推后续的。"""
    row = runs.get_run(run_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="运行不存在")
    if row.status != "running":  # 已结束的直接给结果,不必重放
        return StreamingResponse(
            iter(
                [
                    _sse({"type": "answer", "text": row.answer or row.error or ""}),
                    _sse({"type": "done"}),
                ]
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )
    return _stream_run(run_id, from_start=True)


# ── 实测数据上传与管理 ────────────────────────────────────────
#
# 为什么数据进 PG 而不是文件系统:见 persistence/models.DatasetRow 的说明 ——
# 一句话是"不落盘就没有路径逃逸、没有清理、没有配额"。
#
# 为什么上传是【接口】而不是 agent 的工具:和 PDF 报告同一条 ——
# 工具要保持只读。让 agent 能接收并落盘文件,等于给提示注入开一扇门。


@app.post("/datasets")
async def upload_dataset(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
) -> dict:
    """上传一张实测数据表(CSV),返回数据集编号与概览。

    体积闸门在【读进内存之前】就要判:先信任 Content-Length,读完再复核一次
    实际大小 —— 只信任声明的长度等于让客户端自己决定能塞多少进来。
    """
    from ruixue_agent.analysis import DatasetError, load_csv, summarize
    from ruixue_agent.analysis import store as ds_store
    from ruixue_agent.analysis.loader import MAX_BYTES

    declared = file.size or 0
    if declared > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_BYTES // 1024 // 1024}MB 上限")

    raw = await file.read()
    if len(raw) > MAX_BYTES:  # 复核:声明的长度不可信
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_BYTES // 1024 // 1024}MB 上限")

    try:
        cm, rows = load_csv(raw)
    except DatasetError as e:
        # 422 而不是 500:这是用户的数据不合契约,不是我们的服务出错。
        # detail 直接来自 DatasetError,那些消息本来就是写给用户看的。
        raise HTTPException(status_code=422, detail=str(e)) from e

    ds_id = ds_store.save(user_id, file.filename or "上传数据.csv", cm, rows)
    logger.info("用户 %s 上传数据集 %s(%d 行)", user_id, ds_id, len(rows))
    return {"dataset_id": ds_id, "filename": file.filename, **summarize(cm, rows)}


@app.get("/datasets")
def list_datasets(user_id: str = Depends(get_current_user)) -> dict:
    """列出我的数据集(不含数据行 —— 列表页用不上,拉出来纯属浪费)。"""
    from ruixue_agent.analysis import store as ds_store

    return {
        "datasets": [
            {
                "dataset_id": d.dataset_id,
                "filename": d.filename,
                "n_rows": d.n_rows,
                "targets": sorted(d.targets),
                "created_at": d.created_at.isoformat(),
            }
            for d in ds_store.list_for(user_id)
        ]
    }


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, user_id: str = Depends(get_current_user)) -> dict:
    """删除我的数据集。不属于我的一律 404 —— 403 会泄露"这个 id 存在"。"""
    from ruixue_agent.analysis import store as ds_store

    if not ds_store.delete_one(dataset_id, user_id):
        raise HTTPException(status_code=404, detail="数据集不存在")
    return {"ok": True}
