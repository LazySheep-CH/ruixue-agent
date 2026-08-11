# 第 1 课:agent 解剖 —— 跑一次真实提问,把 agent 肚子里的【全部消息】倒出来看。
#
# 运行:uv run python scripts/study/01_agent_anatomy.py
#
# 你会看到,所谓 "agent" 在底层只是一个不断变长的【消息列表】:
#   human(你的问题)→ ai(带 tool_calls,表示"我要用工具")
#   → tool(工具的返回)→ ai(纯文字,这就是最终答案)
# 中间那两步可能循环好几轮 —— 这个循环就叫 ReAct(Reason 想 + Act 做)。

import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")

from ruixue_agent.agents import create_ruixue_agent

agent = create_ruixue_agent()

# thread_id = 会话的身份证。同一个 thread_id 的历史会被 checkpointer 记住(第 9 课)。
# 这里每次随机,保证从一张白纸开始。
cfg = {"configurable": {"thread_id": f"study-{uuid.uuid4().hex[:8]}"}}

print("提问:尉犁县的土壤 pH 大概多少?\n" + "=" * 60)
state = agent.invoke({"messages": [{"role": "user", "content": "尉犁县的土壤 pH 大概多少?"}]}, cfg)

# state["messages"] 就是 agent 的全部"思考过程"。逐条打印:
for i, m in enumerate(state["messages"]):
    kind = m.type  # human / ai / tool
    if kind == "ai" and m.tool_calls:
        # AI 说"我要调工具" —— 注意它此刻还没有答案,只有【调用意图】
        for tc in m.tool_calls:
            print(f"[{i}] ai  → 决定调工具 {tc['name']}({tc['args']})")
    elif kind == "tool":
        print(f"[{i}] tool→ 工具返回:{str(m.content)[:60]}…")
    else:
        text = str(m.content)[:80].replace("\n", " ")
        print(f"[{i}] {kind:5}: {text}…")

print("=" * 60)
print(f"共 {len(state['messages'])} 条消息。最后一条不带 tool_calls 的 ai 消息 = 最终答案。")
