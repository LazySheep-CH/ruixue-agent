"""MCP 集成测试:真连一个本地 MCP server,而不是只测配置解析。

重点是【优雅降级】:外部 server 不可控(重启/网络抖动/命令不存在),
任何失败都必须降级为"少几个工具继续跑",绝不能让 agent 起不来。
"""

import asyncio
import sys
from pathlib import Path

import pytest

from ruixue_agent.mcp import client as mcp_client

ECHO_SERVER = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
_ECHO_CONN = {"echo": {"transport": "stdio", "command": sys.executable, "args": [str(ECHO_SERVER)]}}


# ── 真通路:起一个真 MCP server,把工具接进来并调用 ────────────
def test_connects_to_real_mcp_server_and_calls_tool():
    tools = asyncio.run(mcp_client._fetch_tools(_ECHO_CONN))
    assert tools, "没从 MCP server 拿到任何工具"
    tool = tools[0]
    assert tool.name == "echo_echo"  # 带 server 前缀,防多 server 同名工具打架
    result = asyncio.run(tool.ainvoke({"text": "地膜"}))
    assert "echo: 地膜" in str(result)


# ── 优雅降级:各种失败都只返回空,不抛异常 ────────────────────
def test_no_config_means_disabled(monkeypatch):
    monkeypatch.setattr(mcp_client, "_servers_from_config", lambda: {})
    assert mcp_client.load_mcp_tools() == []


def test_unreachable_server_degrades_gracefully(monkeypatch):
    """server 命令不存在 —— 必须返回空而不是崩,否则 agent 起不来。"""
    monkeypatch.setattr(
        mcp_client,
        "_servers_from_config",
        lambda: {
            "bad": {"transport": "stdio", "command": "this-command-does-not-exist", "args": []}
        },
    )
    assert mcp_client.load_mcp_tools() == []


def test_broken_config_degrades_gracefully(monkeypatch):
    """配置读取本身出错(文件缺失/格式错)也不能影响 agent 启动。"""

    def boom():
        raise FileNotFoundError("config.yaml 不存在")

    monkeypatch.setattr(mcp_client, "load_config", boom)
    assert mcp_client._servers_from_config() == {}


def test_malformed_servers_section(monkeypatch):
    """mcp.servers 写成了列表而不是字典 —— 要能识别并忽略。"""
    monkeypatch.setattr(mcp_client, "load_config", lambda: {"mcp": {"servers": ["oops"]}})
    assert mcp_client._servers_from_config() == {}


def test_reads_servers_from_config(monkeypatch):
    monkeypatch.setattr(
        mcp_client, "load_config", lambda: {"mcp": {"servers": {"a": {"transport": "stdio"}}}}
    )
    assert "a" in mcp_client._servers_from_config()


# ── 超时:外部服务卡住不能拖住 agent 启动 ─────────────────────
def test_has_connect_timeout():
    assert mcp_client.CONNECT_TIMEOUT > 0


@pytest.mark.parametrize("bad", [None, {}, {"mcp": None}, {"mcp": {}}])
def test_missing_mcp_section_variants(monkeypatch, bad):
    monkeypatch.setattr(mcp_client, "load_config", lambda: bad or {})
    assert mcp_client._servers_from_config() == {}
