"""演示多轮对话:同一 thread_id 的追问,agent 能记住上下文。

前置:docker 起着(会调 RAG),agent.py 里已加 checkpointer。
用法:uv run python scripts/demo_multiturn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from ruixue_agent.agents import create_ruixue_agent

agent = create_ruixue_agent()


def ask(config, question: str) -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    print(f"\n问: {question}\n答: {result['messages'][-1].content}")


# ---- 主组:同一 thread_id 的两轮,验证"记住上下文" ----
config_a = {"configurable": {"thread_id": "conv-a"}}
print("=== 主组(conv-a):连续两轮 ===")
ask(config_a, "PBAT地膜是什么材料?")          # 第一轮
ask(config_a, "它的降解周期大概多久?")         # 第二轮:"它" 应被理解成 PBAT

# ---- 对照组:新 thread_id 问"它",没有历史,应该不知道"它"指什么 ----
print("\n=== 对照组(conv-b):新会话问'它' ===")
ask({"configurable": {"thread_id": "conv-b"}}, "它的降解周期大概多久?")
