"""子智能体(多 Agent 协作):把专项子任务【委派】给"专家"子 agent。

核心思想(业界通行做法):**子 agent = 一个工具**。
主 agent(项目经理)不亲自干细活,而是通过一个 delegate 工具,把自包含的子任务
派给某位专家(子 agent);专家用自己的【窄工具集 + 窄提示】独立完成,只把结论回给主 agent。

为什么这样设计(四维):
- 架构:子 agent 复用 create_agent,不引入新框架;"专家=工具"契合已有的工具注册表模式。
- 扩展:加一位专家 = 往 _EXPERTS 加一条,别处不动(开闭原则)。
- 安全:最小权限 —— 每位专家只拿它该有的工具(计算专家碰不到 RAG);子任务上下文隔离。
- 成本/防递归:专家用便宜模型(flash);且专家工具集【不含】delegate,故专家不能再派活,
  从根上杜绝"无限递归派活"。
"""

from functools import cache

from langchain.agents import create_agent
from langchain_core.tools import tool

from ruixue_agent.models import create_model
from ruixue_agent.tools.calc import estimate_film_usage
from ruixue_agent.tools.environment import get_climate_info, get_soil_info
from ruixue_agent.tools.optimize import screen_film_recipes
from ruixue_agent.tools.rag import search_knowledge

# ── 专家注册表:专家名 -> {描述, 工具集, 系统提示}。加专家 = 加一条 ──────────
# 注意:这里【不放】delegate 工具进任何专家 —— 专家不能再派活(防递归)。
#
# description 字段不是注释,是【给主 agent 看的路由依据】:它会被拼进
# delegate_to_expert 的工具描述(见文件末尾)。写清楚"什么时候派给它",
# 主 agent 才派得准。
_EXPERTS: dict[str, dict] = {
    "文献检索专家": {
        "description": "深挖文献与标准条文:需要多轮检索、汇总多篇资料、给出处引用时派它",
        "tools": [search_knowledge],
        "prompt": (
            "你是地膜领域的文献检索专家。只依据检索到的材料回答,"
            "标注出处,数字与原文一致,材料不足就说不足,绝不编造。"
        ),
    },
    "配方优化专家": {
        "description": "为某地某作物推荐配方:要批量对比候选配方、权衡三大性能并给取舍理由时派它",
        # 给它"批量试算"而不是单个预测工具:一次拿到全部候选的对比表,
        # 少转很多圈 LLM 循环(省钱省时)。再配环境查询和知识库供佐证。
        "tools": [screen_film_recipes, get_soil_info, get_climate_info, search_knowledge],
        "prompt": (
            "你是地膜配方选型专家。任务:结合当地环境,为用户的作物场景推荐合适的"
            "生物降解地膜配方(PBAT/PLA 比例与厚度)。\n"
            "工作方法:\n"
            "1. 先用 screen_film_recipes 拿到候选配方在当地的三大性能对比表"
            "(天数取作物生育期,如棉花约150天、玉米约120天);必要时查土壤/气候佐证。\n"
            "2. 权衡三者,不存在单一最优:\n"
            "   · 降解率——生育期内不能降解过头(否则提前破膜,失去保墒除草作用),"
            "     生育期后又要能降解干净(否则残留污染);\n"
            "   · 拉伸强度——要能承受铺膜机械力和风,过低易破;\n"
            "   · 水蒸气透过率——越低越保墒,干旱区尤其重要。\n"
            "3. 给出推荐配方 + 明确的理由(引用表中数字),并说明取舍在哪、风险是什么。\n"
            "诚实原则:模型预测有不确定性(部分参数用了默认估计),务必说明"
            "结论为参考、建议小面积试用验证。不要编造表中没有的数字。"
        ),
    },
    # 将来加专家就在这加一条,比如:
    # "用量计算专家": {"tools": [estimate_film_usage], "prompt": "你是用量/成本计算专家……"},
}

# 消掉"导入了但暂未使用"的告警;上面注释里的用量专家启用后即用到。
_ = estimate_film_usage


@cache
def _build_expert(name: str, model_name: str = "deepseek-v4-flash"):
    """按注册表造一个专家子 agent。**构建一次,之后查缓存复用。**

    为什么要缓存:agent 图的编译 + 模型客户端创建,实测每次约 127ms ——
    没缓存时【每次派活都重付一遍】。deepagents 的 SubAgentMiddleware 也是
    同一做法:在中间件构造时把所有子 agent 编译好,task 工具只查字典。

    复用实例安全吗?安全,关键在下面这条:
    刻意【不传 checkpointer】:agent 图本身是无状态的(状态全在 invoke 传入的
    messages 里),没有 checkpointer 就没有任何跨调用的残留 —— 这正是
    "上下文隔离"(专家不背主对话的历史包袱,主 agent 也不被专家的中间步骤污染)。
    同一个图对象被并行调用也互不干扰,LangGraph 对每次 invoke 单独跑。
    """
    spec = _EXPERTS[name]
    return create_agent(
        create_model(model_name),  # 便宜模型:专家干的是内部活
        spec["tools"],
        system_prompt=spec["prompt"],
    )


@tool
def delegate_to_expert(expert: str, task: str) -> str:
    """把一个子任务委派给某位专家子智能体,返回其结论。"""
    if expert not in _EXPERTS:
        # 名字不对时,给模型一个能自我纠正的提示,而不是抛异常崩掉
        return f"没有名为「{expert}」的专家。可选:{list(_EXPERTS)}"

    agent = _build_expert(expert)
    # ===== (你写)=====
    # 让专家子 agent 独立跑一遍这个子任务,把它的【最终答案】(最后一条消息的内容)返回:
    #   result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    #   return result["messages"][-1].content
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content


# 工具描述【从注册表生成】,不手写。
#
# 为什么必须这样:工具描述就是主 agent 的路由依据 —— 派不派活、派给谁,
# 全凭这段文字。手写清单一定会和注册表跑偏:实测就发生过,注册表里有两位专家,
# docstring 却只写了"可选专家:文献检索专家",主 agent 压根不知道还有配方专家。
# deepagents 的 task 工具同样是 TASK_TOOL_DESCRIPTION.format(available_agents=...)。
delegate_to_expert.description = (
    "把一个子任务委派给某位专家子智能体,返回其结论。\n"
    "什么时候用:当问题需要下列某项专长时,把该部分作为【自包含】的任务描述交给对应专家"
    "(专家看不到主对话上下文,task 里要带全必要信息)。\n"
    "可选专家:\n" + "\n".join(f"- {n}:{s['description']}" for n, s in _EXPERTS.items())
)
