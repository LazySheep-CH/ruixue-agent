# 第 5 课:亲眼看 ModelCallLimitMiddleware 掐断死循环。
#
# 运行:uv run python scripts/study/02_call_limit.py
#
# 造一个【永远想调工具】的假模型(模拟"钻牛角尖停不下来"的坏情况),
# 把上限临时设成 4,看它到底在第几轮被代码强制停下 —— 证明:
#   上限是【代码在数数】,和模型说什么、想循环几次,毫无关系。

import sys

sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

_round = 0


@tool
def do_nothing() -> str:
    """一个啥也不干的工具。"""
    global _round
    _round += 1
    print(f"    ↳ 第 {_round} 次真的执行了工具")
    return "ok"


# 假模型:不管收到什么,永远回"我还要再调一次 do_nothing"——永不给最终答案。
# 这就是"死循环"的最坏情况。每条 AIMessage 要有不同 id,否则框架会报错。
_loop = [
    AIMessage(
        content="", id=f"turn-{i}", tool_calls=[{"name": "do_nothing", "args": {}, "id": f"c{i}"}]
    )
    for i in range(50)
]


class ForeverLooping(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kw):
        return self


agent = create_agent(
    ForeverLooping(responses=_loop),
    [do_nothing],
    # 上限临时设成 4(生产是 25),好让你几秒钟就看到效果。
    middleware=[ModelCallLimitMiddleware(run_limit=4, exit_behavior="end")],
)

print("上限设为 4。假模型会【永远】想调工具。看它第几轮被强制停下:\n")
state = agent.invoke({"messages": [{"role": "user", "content": "开始"}]})

print(f"\n模型实际被允许说话的轮数:{sum(1 for m in state['messages'] if m.type == 'ai')}")
print("→ 到 4 就停了。假模型明明准备了 50 轮,代码根本没让它跑到第 5 轮。")
print("→ 这就是为什么'你连续思考100轮'突破不了:数数的是中间件,不是模型。")
