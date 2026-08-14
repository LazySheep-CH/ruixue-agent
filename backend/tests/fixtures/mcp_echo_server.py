"""最小 MCP server(测试用):验证 MCP 集成是真能连通,而不是只测了配置解析。

以 stdio 方式运行,暴露一个 echo 工具。测试通过 command/args 拉起本文件。
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-server")


@mcp.tool()
def echo(text: str) -> str:
    """原样返回输入(用于验证 MCP 通路)。"""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
