"""MCP(Model Context Protocol):把外部 MCP server 的工具接进本 agent。

MCP 是让 agent 使用【外部工具】的开放协议。价值在于:工具不必写进本仓库,
第三方(或你自己的其他服务)按协议暴露能力,agent 配置一下就能用 ——
从"改代码加工具"变成"改配置加工具"。

设计要点:
- 配置驱动:server 列表写在 config.yaml 的 mcp 段,不硬编码。
- 优雅降级:外部 server 连不上是常态(对方重启/网络抖动),
  绝不能因此让整个 agent 起不来 —— 失败只记日志、少几个工具,主流程照常。
- 默认关闭:没配置就完全不启用,零开销、零风险。
"""

from ruixue_agent.mcp.client import load_mcp_tools

__all__ = ["load_mcp_tools"]
