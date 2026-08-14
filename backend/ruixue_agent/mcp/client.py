"""MCP 客户端:按配置连接外部 MCP server,把它们的工具转成 agent 可用的工具。

配置示例(config.yaml):

    mcp:
      servers:
        filesystem:                       # 本地进程型(stdio)
          transport: stdio
          command: npx
          args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
        weather:                          # 远程 HTTP 型
          transport: streamable_http
          url: https://example.com/mcp

没有 mcp 段 = 不启用,零开销。
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool

from ruixue_agent.config import load_config

logger = logging.getLogger("ruixue.mcp")

# 单个 server 的连接超时(秒)。外部服务不可控,必须设上限 ——
# 否则一个卡住的 server 会拖住整个 agent 的启动。
CONNECT_TIMEOUT = 15


def _servers_from_config() -> dict:
    """从 config.yaml 读 mcp.servers;缺失或读取失败都返回空(不启用)。"""
    try:
        cfg = load_config()
    except Exception as e:  # 配置文件不存在/格式错 —— 不该影响 agent 启动
        logger.warning("读取配置失败(%s),MCP 不启用", type(e).__name__)
        return {}
    servers = (cfg.get("mcp") or {}).get("servers") or {}
    if not isinstance(servers, dict):
        logger.warning("mcp.servers 配置格式不对(应为字典),MCP 不启用")
        return {}
    return servers


async def _fetch_tools(servers: dict) -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # tool_name_prefix=True:给工具名加 server 前缀,避免多个 server 的同名工具打架
    client = MultiServerMCPClient(connections=servers, tool_name_prefix=True)
    return await asyncio.wait_for(client.get_tools(), timeout=CONNECT_TIMEOUT)


def load_mcp_tools() -> list[BaseTool]:
    """连接配置里的 MCP server 并返回其工具;任何失败都降级为空列表。

    为什么整体 try:外部 server 连不上是常态(对方重启、网络抖动、命令不存在)。
    这时正确的行为是"少几个工具继续跑",而不是让整个 agent 起不来。
    """
    servers = _servers_from_config()
    if not servers:
        return []
    try:
        tools = asyncio.run(_fetch_tools(servers))
    except Exception as e:
        logger.warning("MCP 工具加载失败(%s):%s —— 本次不加载外部工具", type(e).__name__, e)
        return []
    logger.info("MCP 已加载 %d 个外部工具:%s", len(tools), [t.name for t in tools])
    return tools
