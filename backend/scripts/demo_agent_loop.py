"""教学演示看见 agent 到底在干什么 —— 不花钱、不连数据库,直接跑。

    uv run python scripts/demo_agent_loop.py

做法:用一个假模型代替真的 DeepSeek。假模型不思考,只会照剧本说话。
这样我们就能指定它说什么,把 agent 的运转过程完全暴露出来。
"""

import sys

# Windows 控制台默认不是 UTF-8,中文会变乱码。强制切成 UTF-8。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import ModelCallLimitMiddleware  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402


class ScriptedModel(FakeMessagesListChatModel):
    """照剧本说话的假模型。

    自带的 FakeMessagesListChatModel 没实现 bind_tools(绑定工具),
    而 create_agent 一定会调它 —— 所以这里补一个:假模型不需要真绑,
    直接把自己还回去就行。
    """

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def check_film_thickness(crop: str) -> str:
    """查询某作物推荐的地膜厚度。"""
    print(f"        >> [工具真的被执行了] 参数 crop={crop}")
    return "推荐厚度 0.010mm"


def print_conversation(title, messages):
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")
    for i, m in enumerate(messages, 1):
        kind = type(m).__name__
        if kind == "HumanMessage":
            print(f"  {i}. [用户] {m.content}")
        elif kind == "AIMessage":
            if m.tool_calls:
                for tc in m.tool_calls:
                    print(f"  {i}. [模型] 『我要调工具 {tc['name']},参数 {tc['args']}』")
            else:
                print(f"  {i}. [模型] {m.content}")
        elif kind == "ToolMessage":
            print(f"  {i}. [工具] 返回: {m.content}")
    print(f"\n  --> 一共 {len(messages)} 条消息")


def make_tool_call(call_id):
    """造一条"我要调工具"的模型消息。"""
    return AIMessage(
        content="",
        id=f"ai_{call_id}",  # 每条消息 id 必须不同(LangGraph 靠 id 判断是不是同一条)
        tool_calls=[{"name": "check_film_thickness", "args": {"crop": "玉米"}, "id": call_id}],
    )


def endless_tool_calls(n=500):
    """造一份"永远在要调工具"的剧本,模拟模型钻牛角尖停不下来。"""
    return [make_tool_call(f"call_{i}") for i in range(n)]


QUESTION = {"messages": [{"role": "user", "content": "玉米用多厚的地膜?"}]}
# 图最多允许转多少步,超了就抛 GraphRecursionError。这是 LangGraph 的最后一道保险,
# 但它是硬崩—— 我们要的是在崩之前就优雅停下(见演示三)。
LIMIT = {"recursion_limit": 25}


# ══════════════════════════════════════════════════════════════
# 演示一:正常的 agent 循环长什么样
# ══════════════════════════════════════════════════════════════
def demo1_normal_loop():
    print("\n\n##### 演示一:一次正常问答,agent 内部发生了什么 #####")

    # 剧本:模型第 1 次说"我要调工具",第 2 次说"这是最终答案"
    model = ScriptedModel(
        responses=[
            make_tool_call("call_1"),
            AIMessage(content="玉米推荐用 0.010mm 的地膜。"),
        ]
    )
    agent = create_agent(model, [check_film_thickness])
    result = agent.invoke(QUESTION)
    print_conversation("【完整过程】", result["messages"])

    print("""
  +-- 看懂了吗 ------------------------------------------------+
  | agent 不是"问一次答一次"。它是一个【循环】:               |
  |                                                            |
  |    模型想 -> 要调工具? -> 是 -> 执行工具 -> 结果塞回去 --+ |
  |      ^                                                   | |
  |      +---------------------------------------------------+ |
  |                    | 否                                    |
  |                    v                                       |
  |                 输出答案,循环结束                          |
  |                                                            |
  | 上面这几条消息 = 转了 1 圈。这就是 ReAct 循环。            |
  +------------------------------------------------------------+""")


# ══════════════════════════════════════════════════════════════
# 演示二:模型钻牛角尖 —— 没有刹车
# ══════════════════════════════════════════════════════════════
def demo2_runaway_no_brake():
    print("\n\n##### 演示二:模型钻牛角尖(一直要调工具),【没装刹车】 #####")

    # 剧本里只有"我要调工具",从来不给"最终答案"
    # -> 模拟模型永远说『我还要再查一次』。
    model = ScriptedModel(responses=endless_tool_calls())
    agent = create_agent(model, [check_film_thickness])
    try:
        result = agent.invoke(QUESTION, config=LIMIT)
        print_conversation("【居然没崩?】", result["messages"])
    except Exception as e:
        print(f"\n  !!! 崩了。异常类型:{type(e).__name__}")
        print(f"      内容:{str(e)[:160]}")
        print("""
  +-- 这就是"为什么要装刹车" ----------------------------------+
  | 模型钻牛角尖,循环停不下来,最后直接崩。                   |
  |                                                            |
  | 现在是假模型:不花钱、瞬间就崩,你只看到一个报错。         |
  | 换成【真的 DeepSeek】,同样的情况是:                       |
  |     每转一圈 = 一次真实 API 调用 = 花钱 + 等几秒            |
  |     转 25 圈 = 25 次调用的钱 + 用户干等半分钟               |
  |     结局还是崩,用户拿到一个 500                            |
  |                                                            |
  | 钱花了,时间耗了,用户还没拿到答案 —— 三头亏。             |
  +------------------------------------------------------------+""")


# ══════════════════════════════════════════════════════════════
# 演示三:同样钻牛角尖 —— 装上刹车
# ══════════════════════════════════════════════════════════════
def demo3_runaway_with_brake():
    print("\n\n##### 演示三:同样钻牛角尖,但【装了刹车中间件】 #####")

    model = ScriptedModel(responses=endless_tool_calls())  # 剧本和演示二完全一样
    agent = create_agent(
        model,
        [check_film_thickness],
        # vvv 唯一的区别就是多了这一行 —— 就是你在 builder.py 里要填的那个空 vvv
        middleware=[ModelCallLimitMiddleware(run_limit=3, exit_behavior="end")],
    )
    result = agent.invoke(QUESTION, config=LIMIT)
    print_conversation("【没崩,自己停住了】", result["messages"])

    print("""
  +-- 刹车干了什么 --------------------------------------------+
  | 它数着模型说了几轮话。到第 3 轮 -> 不让它再说,直接收尾。  |
  |                                                            |
  | 对比演示二:                                                |
  |     没刹车 -> 崩(500 错误 + 烧掉一堆钱)                   |
  |     有刹车 -> 优雅停下(exit_behavior="end")                |
  |                                                            |
  | 这就是 builder.py 里那个空的全部意义。                      |
  +------------------------------------------------------------+""")


if __name__ == "__main__":
    demo1_normal_loop()
    demo2_runaway_no_brake()
    demo3_runaway_with_brake()
    print("\n\n三个演示跑完了。\n")
