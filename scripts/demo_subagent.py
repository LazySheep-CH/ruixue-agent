"""【教学演示】看见"多 Agent 协作":主 agent 把子任务派给专家子 agent。

    uv run python scripts/demo_subagent.py

不花钱、不连库:用假模型照剧本演一遍委派全过程。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402

import ruixue_agent.subagents as sub  # noqa: E402


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def print_flow(messages):
    for m in messages:
        k = type(m).__name__
        if k == "HumanMessage":
            print(f"  [用户 → 主]     {m.content}")
        elif k == "AIMessage":
            if m.tool_calls:
                for tc in m.tool_calls:
                    a = tc["args"]
                    print(f"  [主 决定派活]   → {a.get('expert')}:{a.get('task')}")
            else:
                print(f"  [主 汇总答复]   {m.content}")
        elif k == "ToolMessage":
            print(f"  [专家 回给主]   {m.content}")


def main():
    # 用假模型替换掉"造专家"用的真模型 -> 专家用这个假模型(给个固定结论)
    sub.create_model = lambda name: ScriptedModel(
        responses=[AIMessage(content="据国标与文献,该 PBAT 地膜 90 天降解率约 90%。")]
    )

    # 主 agent 的假模型:先决定派活给文献专家,拿到结论后汇总
    lead_model = ScriptedModel(
        responses=[
            AIMessage(
                content="",
                id="lead_1",
                tool_calls=[
                    {
                        "name": "delegate_to_expert",
                        "args": {"expert": "文献检索专家", "task": "查 PBAT 地膜降解性能"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="综合文献专家的结论:该 PBAT 地膜 90 天降解率约 90%,降解性能良好。"),
        ]
    )

    lead = create_agent(lead_model, [sub.delegate_to_expert])
    out = lead.invoke({"messages": [{"role": "user", "content": "PBAT 地膜降解性能怎么样?"}]})

    print("\n===== 多 Agent 协作全过程 =====")
    print_flow(out["messages"])
    print("""
  +-- 看懂了吗 ------------------------------------------------+
  | 主 agent 没自己查,而是【派活】给文献专家(一个子 agent)。 |
  | 专家用自己的窄工具+窄提示独立跑完,只把【结论】回给主。    |
  | 对主 agent 来说,"调专家"和"调普通工具"没区别 —— 这就是   |
  | "子 agent = 工具",多 Agent 能层层组合的根本原因。         |
  +------------------------------------------------------------+""")


if __name__ == "__main__":
    main()
