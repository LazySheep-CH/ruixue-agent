"""中间件单元测试:用【测试替身】隔离测中间件自己的逻辑。

关键手法:不启动真 agent、不连 LLM/docker,而是给中间件一个【假的 request】
和【假的 handler】。这样测试快(毫秒级)、稳、可进 CI —— 测的是"中间件干了什么",
不是 LLM 或工具本身。这正是"单元测试测你的逻辑,不测框架和第三方"。
"""

import logging

from langchain_core.messages import ToolMessage

from ruixue_agent.agents.middlewares import (
    TimingLoggingMiddleware,
    ToolErrorHandlingMiddleware,
)


class _FakeReq:
    """假 request:中间件只用到 tool_call 里的 name / id,给这两个就够。"""

    def __init__(self, name="fake_tool", call_id="call_1"):
        self.tool_call = {"name": name, "id": call_id}


# ---------------------------------------------------------------------------
# TimingLoggingMiddleware:透传结果 + 记一条含工具名和耗时的 INFO 日志
# ---------------------------------------------------------------------------


def test_timing_passes_result_through():
    """中间件不该改工具结果 —— handler 返回什么,它就返回什么。"""
    mw = TimingLoggingMiddleware()
    result = mw.wrap_tool_call(_FakeReq(), lambda req: "工具结果")
    assert result == "工具结果"


def test_timing_logs_tool_name_and_duration(caplog):
    """记了一条 INFO 日志,含工具名与"耗时"。caplog 是 pytest 捕获日志的夹具。"""
    mw = TimingLoggingMiddleware()
    with caplog.at_level(logging.INFO, logger="ruixue.agent"):
        mw.wrap_tool_call(_FakeReq(name="search_knowledge"), lambda req: "ok")

    assert "search_knowledge" in caplog.text
    assert "耗时" in caplog.text


# ---------------------------------------------------------------------------
# ToolErrorHandlingMiddleware:工具抛异常时兜住,不崩,返回优雅的错误 ToolMessage
# ---------------------------------------------------------------------------


def test_error_handling_passes_result_through_on_success():
    """没出错时应【透明】—— 正常返回工具结果,不多管闲事。"""
    mw = ToolErrorHandlingMiddleware()
    result = mw.wrap_tool_call(_FakeReq(), lambda req: "正常结果")
    assert result == "正常结果"


def test_error_handling_catches_exception_and_returns_message():
    """核心:工具抛异常 → 不往外冒 → 返回一条错误 ToolMessage(优雅降级)。"""
    mw = ToolErrorHandlingMiddleware()

    def failing_handler(req):
        raise RuntimeError("boom")  # 模拟工具挂了

    # 能走到下一行(没抛异常),本身就证明异常被兜住了
    result = mw.wrap_tool_call(_FakeReq(call_id="call_42"), failing_handler)

    assert isinstance(result, ToolMessage)  # 返回的是 ToolMessage,不是异常
    assert "执行失败" in result.content  # 内容里带失败说明,agent 能读懂
    assert result.tool_call_id == "call_42"  # 对应回原来那次工具调用


def test_error_handling_does_not_leak_raw_detail():
    """【安全】原始错误细节绝不能进返回给模型的内容里 —— 只放行异常【类型名】。

    动机:工具的异常文字里常带内网 IP、数据库地址、用户名等敏感信息。
    它会进大模型上下文,可能被复述给用户 = 泄露内部结构。
    脱敏后:模型只知道"是 ConnectionError 这类错误",拿不到任何具体细节。
    这条测试就是那道脱敏的"防倒退"闸门 —— 谁把细节漏回去,它立刻变红。
    """
    mw = ToolErrorHandlingMiddleware()
    secret = "connect to host=10.0.1.5 user=admin password 认证失败"  # 假装的敏感信息

    def failing_handler(req):
        raise ConnectionError(secret)

    result = mw.wrap_tool_call(_FakeReq(call_id="c1"), failing_handler)

    assert secret not in result.content  # ← 核心:敏感原文绝不出现
    assert "10.0.1.5" not in result.content  # 连内网 IP 的片段也不能有
    assert "ConnectionError" in result.content  # 但类型名要给,模型才好应对


def test_error_handling_logs_error(caplog):
    """失败时记一条 ERROR 日志(级别比 info 高),含工具名 —— 便于线上排查。"""
    mw = ToolErrorHandlingMiddleware()

    def failing_handler(req):
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="ruixue.agent"):
        mw.wrap_tool_call(_FakeReq(name="boom_tool"), failing_handler)

    assert "boom_tool" in caplog.text
    assert any(r.levelno == logging.ERROR for r in caplog.records)
