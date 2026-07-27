"""工具注册表:把各文件中的工具收集为一个列表,供 agent 装配。

新增工具的方式:在对应文件实现后,加入 get_tools() 的返回列表(开闭原则,
不需要改动装配代码)。
"""

from langchain_core.tools import BaseTool

from ruixue_agent.tools.calc import estimate_film_usage
from ruixue_agent.tools.environment import get_environment_tools
from ruixue_agent.tools.optimize import get_optimize_tools
from ruixue_agent.tools.predictor import get_predictor_tools
from ruixue_agent.tools.rag import search_knowledge


def get_tools() -> list[BaseTool]:
    """返回【基础叶子工具】。

    这一层是最底层的具体工具,不依赖上层。多 Agent 的 delegate_to_expert
    属于"组装层"能力,由 builder 拼给主 agent(见 builder.py),不放这里 ——
    否则 tools 反向依赖 subagents,会形成循环导入。
    """
    return [
        estimate_film_usage,  # 用量估算(kg,确定性公式);成本待做,见下
        search_knowledge,  # 知识库检索(Agentic RAG,带出处)
        *get_environment_tools(),  # 环境查询:土壤(离线)/ 气候(NASA 在线)
        *get_predictor_tools(),  # 性能预测:按地点综合 + 降解率/透过率/拉伸强度
        *get_optimize_tools(),  # 配方批量试算(对比表);权衡推荐见「配方优化专家」
        # 【待做】成本估算:价格是时变+商业敏感数据,不写死在代码里。设计为
        #   成本 = 用量(kg) × 单价 —— 单价优先用用户传入(他知道本地采购价),
        #   否则读 config.yaml 的参考价(标注日期/来源)。
        # 调研结论(已实测):淘宝/京东/1688/一亩田均有反爬,直接爬不通;
        #   合规路径是平台开放平台 API(需企业资质)或行情数据服务(付费)。
        #   更关键的是零售价 ≠ 农业大宗价,直接用会严重高估,故不急于自动化。
        # 【待做】web_search:见 tools/web.py 占位
    ]
