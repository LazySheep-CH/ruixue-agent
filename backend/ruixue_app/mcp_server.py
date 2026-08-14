"""MCP 服务端:把本项目的工具暴露给外部 agent 调用。

和 ruixue_agent/mcp/client.py 方向相反:client 是接别人的工具进来,这里是把
predict_* / screen_film_recipes / search_knowledge 这类自有能力开放出去,
对方在 MCP 客户端里配一个 url + key 就能用。

几个设计决定:

- 挂在现有 FastAPI 进程上,不单独起服务。embedding、reranker、三个树模型
  每 worker 占约 677MB(压测数据),另起进程等于再付一份内存,没有收益。
  代价是 MCP 出问题会波及主服务,可接受:本来就是同一批工具代码。
- 鉴权复用主服务的 JWT / X-API-Key,不设第二套账号;认证失败一律拒绝
  (fail closed),配额计数失败放行(fail open),方向和 quota.py 一致。
- 默认关闭,RUIXUE_MCP_SERVER=1 才启用;RUIXUE_MCP_TOOLS 可白名单限定。
- 每次工具调用计入 DAILY_CHAT_QUOTA,和 /chat 同一个计数器。外部调用方是
  程序不是人,search_knowledge 每次都有 LLM 成本,不计数撑不住。

对方接入示例:

    {"mcpServers": {"ruixue": {
        "transport": "streamable_http",
        "url": "https://<域名>/mcp",
        "headers": {"X-API-Key": "<签发的 key>"}}}}
"""

from __future__ import annotations

import contextvars
import functools
import logging
import os

from ruixue_app import quota
from ruixue_app.auth import API_KEYS
from ruixue_app.security.tokens import TokenError, decode_access_token

logger = logging.getLogger("ruixue.mcp.server")

# 挂载路径。对方配置里的 url 要写到这一层。
MOUNT_PATH = "/mcp"

# 当前 MCP 调用者的 user_id。
#
# 为什么用 contextvar 而不是参数:MCP 工具的签名是暴露给外部的契约,
# 不能为了内部需要塞一个 user_id 进去 —— 那样对方还得填,而且他填什么都不可信。
# 身份必须由网关从凭证里解析,工具层只读不写。
# contextvar 而非全局变量:每个请求各自一份,并发下不会串号。
_caller: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_caller", default="")


def _enabled() -> bool:
    return os.getenv("RUIXUE_MCP_SERVER", "").strip().lower() in {"1", "true", "yes", "on"}


def _allowed_tools() -> set[str] | None:
    """None = 全部暴露;否则只暴露白名单里的。"""
    raw = os.getenv("RUIXUE_MCP_TOOLS", "").strip()
    if not raw:
        return None
    return {t.strip() for t in raw.split(",") if t.strip()}


def authenticate(headers: dict[str, str]) -> str | None:
    """从请求头解析调用者身份;解析不出返回 None(调用方必须据此拒绝)。

    复用主服务的两种凭证,不另设账号体系:
      Authorization: Bearer <jwt>   网页登录用户
      X-API-Key: <key>              程序化调用

    注意:任何异常都返回 None(拒绝),不能返回一个"默认用户" ——
      认证是安全边界,出错时必须往严的方向倒。
    """
    bearer = headers.get("authorization", "")
    if bearer.lower().startswith("bearer "):
        try:
            payload = decode_access_token(bearer[7:].strip())
            return f"u{payload['sub']}"
        except (TokenError, KeyError, Exception):
            return None
    key = headers.get("x-api-key", "")
    # 注意读的是 auth.API_KEYS(模块导入时从环境变量加载的那份),
    # 和主服务用的是同一张表 —— 吊销一个 key,两边同时失效。
    if key and key in API_KEYS:
        return API_KEYS[key]
    return None


def _metered(fn, tool_name: str):
    """给工具套上配额闸门。超限时返回文字而不是抛异常。

    为什么不抛:MCP 客户端拿到异常通常只显示 "tool error",对方的 agent
    不知道发生了什么、更不知道该告诉用户什么。返回一句人话,
    对方的模型能直接转述给它的用户 —— 这和我们自己的
    ToolErrorHandlingMiddleware 是同一个思路。
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        user = _caller.get()
        allowed, used = quota.consume(user or "mcp-anonymous")
        if not allowed:
            logger.warning("MCP 调用方 %s 超配额(%d/%d)", user, used, quota.DAILY_LIMIT)
            return f"已达每日调用上限({quota.DAILY_LIMIT} 次),请明天再试。"
        logger.info("MCP 工具调用 tool=%s user=%s used=%d", tool_name, user, used)
        return fn(*args, **kwargs)

    return wrapper


def _transport_security():
    """DNS 重绑定防护:校验 Host 头,只接受我们认可的域名。

    这是在防什么:
    攻击者让受害者浏览器访问 evil.com,该域名的 DNS 先解析到攻击者的 IP、
    随后改解析到 127.0.0.1。浏览器认为还是同源,于是可以对受害者本机
    的服务发请求 —— 本地跑的 MCP server 就这样被外站脚本调用了。
    校验 Host 头能挡住这类攻击:请求头里写的仍是 evil.com,不在白名单里。

    为什么必须配:
    SDK 默认只信 localhost。**部署到真域名后不配这一项,MCP 端点会
    全量返回 421**,而且报错只有一句 "Invalid Host header" ——
    鉴权是通的、路由是通的,只在最后一步挂,极难定位。

    MCP_ALLOWED_HOSTS 为空时退回 localhost(本地开发够用),
    同时把 testserver 加进去让 TestClient 能跑 —— 测试环境不放行的话,
    这条链路就只能靠手工验证,等于没有回归保护。
    """
    from mcp.server.transport_security import TransportSecuritySettings

    raw = os.getenv("MCP_ALLOWED_HOSTS", "").strip()
    hosts = [h.strip() for h in raw.split(",") if h.strip()] or [
        "localhost",
        "localhost:*",
        "127.0.0.1",
        "127.0.0.1:*",
        "testserver",
    ]
    # allowed_origins 同步放行:浏览器端 MCP 客户端会带 Origin 头。
    origins = [h if "://" in h else f"http://{h}" for h in hosts] + [
        h if "://" in h else f"https://{h}" for h in hosts
    ]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


def build_server():
    """按配置构建 FastMCP server;未启用返回 None。

    返回 None 而不是抛异常:和客户端一样,"没开启"是正常状态不是错误。
    """
    if not _enabled():
        return None

    from mcp.server.fastmcp import FastMCP

    from ruixue_agent.tools import get_tools

    allow = _allowed_tools()
    # stateless_http=True:每个请求独立,不维护长连接会话。
    # 我们的工具全部无状态(输入参数 → 输出结果),不需要会话;
    # 而有状态模式要求客户端先 initialize 再调用,多一轮往返、且进程重启后会话失效。
    # streamable_http_path="/":FastMCP 默认把端点放在它自己的 "/mcp" 下,
    # 而我们又把整个子应用挂在 MOUNT_PATH("/mcp")上 —— 两段路径会叠成 "/mcp/mcp"。
    # 表现是 401 过了却 404,查起来很费劲(鉴权层看着是通的)。挂载方负责定路径,
    # 子应用就该挂在根上。
    mcp = FastMCP(
        "ruixue-mulch-film",
        stateless_http=True,
        streamable_http_path="/",
        transport_security=_transport_security(),
    )

    exposed = []
    for t in get_tools():
        if allow is not None and t.name not in allow:
            continue
        # t.func 是 @tool 装饰前的原函数 —— FastMCP 要靠它的类型注解和
        # docstring 自动生成工具 schema。传 LangChain 的 StructuredTool 对象不行。
        mcp.add_tool(_metered(t.func, t.name), name=t.name, description=t.description)
        exposed.append(t.name)

    if not exposed:
        logger.warning("RUIXUE_MCP_TOOLS 没匹配到任何工具,MCP server 不启用")
        return None

    logger.info("MCP server 已启用,暴露 %d 个工具:%s", len(exposed), exposed)
    return mcp


class AuthGate:
    """ASGI 网关:先验身份,再放行到 MCP app。

    为什么是 ASGI 中间件而不是 FastAPI 依赖:MCP app 是挂进来的子应用,
    FastAPI 的 Depends 不会作用到子应用的路由上 —— 写了也不生效,
    而且是静默不生效:接口能跑通,只是没鉴权。这类洞最危险。

    身份解析成功后写进 contextvar,工具层的配额闸门据此计费。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            # 非 HTTP(如 lifespan 事件)原样透传,不能在这里拦
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        user = authenticate(headers)
        if user is None:
            await self._reject(send)
            return

        token = _caller.set(user)
        try:
            await self.app(scope, receive, send)
        finally:
            # 必须复位:ASGI 下同一个协程上下文可能被复用,不复位会串号
            _caller.reset(token)

    @staticmethod
    async def _reject(send) -> None:
        body = b'{"detail":"\xe6\x9c\xaa\xe6\x8e\x88\xe6\x9d\x83"}'  # {"detail":"未授权"}
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
