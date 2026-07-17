"""跑一次瑞雪 agent(手动 demo)。用法: uv run python main.py"""

import sys

from ruixue_agent.agent import create_ruixue_agent

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK,不改会打不出中文

if __name__ == "__main__":
    agent = create_ruixue_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "10亩地需要多少PE地膜?"}]}
    )
    print(result["messages"][-1].content)
