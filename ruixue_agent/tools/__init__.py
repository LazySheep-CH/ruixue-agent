"""工具注册表:把各文件中的工具收集为一个列表,供 agent 装配。

新增工具的方式:在对应文件实现后,加入 get_tools() 的返回列表(开闭原则,
不需要改动装配代码)。
"""

from langchain_core.tools import BaseTool

from ruixue_agent.tools.calc import estimate_film_usage
from ruixue_agent.tools.rag import search_knowledge


def get_tools() -> list[BaseTool]:
    """返回 agent 可用的全部工具。"""
    return [
        estimate_film_usage,  # 用量/成本计算(确定性)
        search_knowledge,  # 知识库检索(Agentic RAG,带出处)
        # web_search / query_price:待 tools/web.py 实现后加入
    ]
