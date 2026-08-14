# 第 6 课:中间件的【顺序】为什么是设计,不是随手排的。
#
# 运行:uv run python scripts/study/03_middleware_order.py
#
# 用同样两个中间件、只交换顺序,看一个失败的工具调用会不会被计时记录下来。
#
# 背景(读 middlewares.py 第 39-44 行):TimingLogging 没有 try/finally,
# 它的"算耗时 + 记日志"两行,只在 handler(request) 不抛异常时才会执行。

import logging
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")


class _NoTraceback(logging.Formatter):
    """只显示日志正文,不打印堆栈 —— 本课要看的是"有没有这条日志",不是异常细节。"""

    def format(self, record):
        record.exc_info = record.exc_text = None
        return f"      [日志] {record.getMessage()}"


_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(_NoTraceback())
logging.basicConfig(level=logging.INFO, handlers=[_h], force=True)

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from ruixue_agent.agents.middlewares import TimingLoggingMiddleware, ToolErrorHandlingMiddleware


@tool
def flaky_search(q: str) -> str:
    """模拟一个"卡了 0.5 秒才失败"的工具(比如 RAG 连不上)。"""
    time.sleep(0.5)
    raise ConnectionError("知识库连不上")


class Fake(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kw):
        return self


def run(label: str, middleware: list):
    print(f"\n{label}")
    msgs = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"name": "flaky_search", "args": {"q": "x"}, "id": "c1"}],
        ),
        AIMessage(content="收尾", id="a2"),
    ]
    agent = create_agent(Fake(responses=msgs), [flaky_search], middleware=middleware)
    state = agent.invoke({"messages": [{"role": "user", "content": "查一下"}]})
    tool_msg = [m for m in state["messages"] if m.type == "tool"][0]
    print(f"      模型收到的工具返回:{str(tool_msg.content)[:45]}…")


# 【我们线上的顺序】builder.py 第 129、131 行:计时在外,错误处理在内。
# 列表里【越靠前 = 越外层】。
run(
    "① 线上顺序(计时在外 → 错误处理在内):",
    [TimingLoggingMiddleware(), ToolErrorHandlingMiddleware()],
)

# 【反过来】错误处理在外,计时在内。
run(
    "② 反过来(错误处理在外 → 计时在内):",
    [ToolErrorHandlingMiddleware(), TimingLoggingMiddleware()],
)

print("\n对比上面两组日志:①有『工具 flaky_search 耗时 5xxms』,②没有。")
print("原因:②里异常从计时的 handler() 里冒出来,算耗时那两行根本没执行到。")
print("后果:线上所有【失败的、慢的】调用在计时日志里【集体隐身】——")
print("      而那恰恰是你排查问题时最想看到的那批数据。")
