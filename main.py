import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows 编码

from ruixue_agent.agent import create_ruixue_agent

if __name__ == "__main__":
    agent = create_ruixue_agent()
    # 问一个会触发工具的问题
    result = agent.invoke({"messages": [{"role": "user", "content": "10亩地需要多少PE地膜？"}]})
    # 打印最后一条消息的内容（回忆 mini-deerflow：result["messages"][-1].content）
    print(result["messages"][-1].content)