"""agents 包:agent 的装配层(builder + prompt + 中间件)。

对标 deer-flow 的 deerflow/agents/。这里把 create_ruixue_agent re-export 出去,
让外部只需 `from ruixue_agent.agents import create_ruixue_agent`,
不必知道它具体在 builder.py 里 —— 包对外是一个整体,内部结构可自由重构。
"""

from ruixue_agent.agents.builder import create_ruixue_agent

__all__ = ["create_ruixue_agent"]
