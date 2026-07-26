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

from langchain.agents import create_agent
from langchain_core.tools import tool

from ruixue_agent.models import create_model
from ruixue_agent.tools.calc import estimate_film_usage
from ruixue_agent.tools.rag import search_knowledge

# ── 专家注册表:专家名 -> {工具集, 系统提示}。加专家 = 加一条 ──────────
# 注意:这里【不放】delegate 工具进任何专家 —— 专家不能再派活(防递归)。
_EXPERTS: dict[str, dict] = {
    "文献检索专家": {
        "tools": [search_knowledge],
        "prompt": (
            "你是地膜领域的文献检索专家。只依据检索到的材料回答,"
            "标注出处,数字与原文一致,材料不足就说不足,绝不编造。"
        ),
    },
    # 将来加专家就在这加一条,比如:
    # "用量计算专家": {"tools": [estimate_film_usage], "prompt": "你是用量/成本计算专家……"},
}

# 消掉"导入了但暂未使用"的告警;上面注释里的用量专家启用后即用到。
_ = estimate_film_usage


def _build_expert(name: str, model_name: str = "deepseek-v4-flash"):
    """按注册表造一个专家子 agent。

    刻意【不传 checkpointer】:每次委派都是一次性、无状态的独立执行 —— 这正是
    "上下文隔离"(专家不背主对话的历史包袱,主 agent 也不被专家的中间步骤污染)。
    """
    spec = _EXPERTS[name]
    return create_agent(
        create_model(model_name),  # 便宜模型:专家干的是内部活
        spec["tools"],
        system_prompt=spec["prompt"],
    )


@tool
def delegate_to_expert(expert: str, task: str) -> str:
    """把一个子任务委派给某位专家子智能体,返回其结论。

    什么时候用:当问题需要某项专长/某类工具时,把该部分交给对应专家。
    可选专家:文献检索专家。
    参数:
        expert: 专家名称(必须是上面列出的之一)
        task:   交给该专家的、【自包含】的子任务描述(专家看不到主对话上下文)
    """
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
