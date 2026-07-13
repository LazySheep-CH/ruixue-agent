"""工具注册表（registry）。

把各类工具收集成一个列表，供 agent 使用。
【这就是你问的"怎么安排以后不断加的工具"的答案】：
    加新工具 = 在对应文件里写好 → 在下面 get_tools() 的列表里加一行。
    结构不用重构，一个个往里加就行（开闭原则）。
"""

from langchain_core.tools import BaseTool

from ruixue_agent.tools.calc import estimate_film_usage

# 以后写了 web 工具再解开下面这行的注释：
# from ruixue_agent.tools.web import web_search, query_price


def get_tools() -> list[BaseTool]:
    """返回 agent 能用的所有工具。"""
    return [
        estimate_film_usage,
        # web_search,      # ← 以后加
        # query_price,     # ← 以后加
    ]
