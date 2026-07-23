"""瑞雪 agent 装配:模型 + 工具 + 系统提示 + 会话记忆(checkpointer)。"""

from langchain.agents import create_agent

from ruixue_agent.agents.middlewares import (
    TimingLoggingMiddleware,
    ToolErrorHandlingMiddleware,
)
from ruixue_agent.agents.prompt import SYSTEM_PROMPT
from ruixue_agent.checkpointer import get_checkpointer
from ruixue_agent.models import create_model
from ruixue_agent.tools import get_tools


def create_ruixue_agent(model_name: str = "deepseek-v4-pro"):
    """按配置装配并返回瑞雪 agent。"""
    model = create_model(model_name)
    tools = get_tools()
    return create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        # 会话状态持久化到 PostgreSQL:重启不丢、多 worker 共享(按 thread_id 存取)
        checkpointer=get_checkpointer(),
        middleware=[TimingLoggingMiddleware(), ToolErrorHandlingMiddleware()],
    )
