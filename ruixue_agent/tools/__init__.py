"""工具注册表:把各文件中的工具收集为一个列表,供 agent 装配。

新增工具的方式:在对应文件实现后,加入 get_tools() 的返回列表(开闭原则,
不需要改动装配代码)。
"""

from langchain_core.tools import BaseTool

from ruixue_agent.tools.calc import estimate_film_usage
from ruixue_agent.tools.rag import search_knowledge


def get_tools() -> list[BaseTool]:
    """返回【基础叶子工具】。

    这一层是最底层的具体工具,不依赖上层。多 Agent 的 delegate_to_expert
    属于"组装层"能力,由 builder 拼给主 agent(见 builder.py),不放这里 ——
    否则 tools 反向依赖 subagents,会形成循环导入。
    """
    return [
        estimate_film_usage,  # 用量/成本计算(确定性)
        search_knowledge,  # 知识库检索(Agentic RAG,带出处)
        # web_search / query_price:待 tools/web.py 实现后加入
    ]
