from langchain.agents import create_agent
from ruixue_agent.models import create_model
from ruixue_agent.tools import get_tools

# 插槽4：系统提示（你写——瑞雪地膜决策助手的"人设 + 规矩"）
SYSTEM_PROMPT = """你是"瑞雪"地膜智能决策助手，专注地膜选型与用量决策。
遇到地膜用量、面积等需要计算的问题，必须调用提供的工具来计算，不要自己估算。
回答用中文，专业、简洁。"""


def create_ruixue_agent(model_name: str = "deepseek-v4-pro"):
    """组装并返回瑞雪 agent。"""
    #   1. model = create_model(model_name)   # 插槽1：你的模型工厂
    model = create_model(model_name)
    #   2. tools = get_tools()                 # 插槽2：你的工具注册表
    tools = get_tools()
    #   3. return create_agent(model=..., tools=..., system_prompt=SYSTEM_PROMPT)
    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)
