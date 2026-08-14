"""测试持久化:分两次【独立进程】运行,验证 PG checkpointer 跨进程记住会话。

关键:两次运行之间进程【退出】了(内存清空)。若还记得,说明状态确实
存进了 PostgreSQL、被下一个进程读回来了 —— 这正是"重启不丢"。

用法(依次跑两条,注意是两次独立启动):
    uv run python scripts/test_persist.py "PBAT地膜是什么材料?"      # 第一轮
    uv run python scripts/test_persist.py "它的降解周期大概多久?"     # 第二轮(全新进程)

第二轮若把"它"理解成 PBAT,持久化就成了。
(对照:InMemorySaver 时第二轮会不认识"它",因为上轮内存已随进程消失。)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from ruixue_agent.agents import create_ruixue_agent

question = sys.argv[1] if len(sys.argv) > 1 else "你好"

agent = create_ruixue_agent()
config = {"configurable": {"thread_id": "persist-demo"}}  # 固定会话ID,两次运行共用

result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
print(f"\n问: {question}\n答: {result['messages'][-1].content}")
