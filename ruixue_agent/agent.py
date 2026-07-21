"""瑞雪 agent 装配:模型 + 工具 + 系统提示。"""

from langchain.agents import create_agent

from ruixue_agent.models import create_model
from ruixue_agent.tools import get_tools

SYSTEM_PROMPT = """你是"瑞雪"地膜智能决策助手，专注地膜选型与用量决策。
遇到地膜用量、面积等需要计算的问题，必须调用提供的工具来计算，不要自己估算。
回答用中文，专业、简洁。"""


def create_ruixue_agent(model_name: str = "deepseek-v4-pro"):
    """按配置装配并返回瑞雪 agent。"""
    model = create_model(model_name)
    tools = get_tools()
    return create_agent(model, tools, system_prompt=SYSTEM_PROMPT)
