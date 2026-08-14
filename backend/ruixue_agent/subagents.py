"""子智能体(多 Agent 协作):把专项子任务委派给"专家"子 agent。

核心思想(业界主流 agent 的通行做法):子 agent = 一个工具。
主 agent(项目经理)不亲自干细活,而是通过一个 delegate 工具,把自包含的子任务
派给某位专家(子 agent);专家用自己的窄工具集 + 窄提示独立完成,只把结论回给主 agent。

这样拆的理由:
- 架构:子 agent 复用 create_agent,不引入新框架;"专家=工具"契合已有的工具注册表模式。
- 扩展:加一位专家 = 往 _EXPERTS 加一条,别处不动(开闭原则)。
- 安全:最小权限 —— 每位专家只拿它该有的工具(计算专家碰不到 RAG);子任务上下文隔离。
- 成本/防递归:专家用便宜模型(flash);且专家工具集不含delegate,故专家不能再派活,
  从根上杜绝"无限递归派活"。
"""

import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache

from langchain.agents import create_agent
from langchain_core.tools import tool

from ruixue_agent.models import create_model
from ruixue_agent.tools.calc import estimate_film_usage
from ruixue_agent.tools.dataset import (
    check_dataset_against_standard,
    compare_dataset_with_model,
    describe_dataset,
    detect_dataset_outliers,
)
from ruixue_agent.tools.environment import (
    get_climate_info,
    get_soil_info,
    get_weather_forecast,
)
from ruixue_agent.tools.optimize import screen_film_recipes
from ruixue_agent.tools.predictor import predict_by_location
from ruixue_agent.tools.rag import search_knowledge

logger = logging.getLogger("ruixue.subagent")

# ── 专家注册表:专家名 -> {描述, 工具集, 系统提示}。加专家 = 加一条 ──────────
# 注意:这里不放delegate 工具进任何专家 —— 专家不能再派活(防递归)。
#
# description 字段不是注释,是给主 agent 看的路由依据:它会被拼进
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
    "故障诊断专家": {
        "description": (
            "地里已经出问题时派它:膜提前破裂/降解太快或太慢/残留清不掉/保墒不达预期。"
            "特征是用户描述的是【已发生的异常现象】,要逐项排查原因,而不是选型或查资料"
        ),
        # 诊断的第一步永远是"这个配方在当地本该表现如何"——没有这个基准,
        # 后面无从判断是"膜不对"还是"预期不对"。所以 predict_by_location 是核心工具。
        # 不给 screen_film_recipes:诊断是找原因,不是重新选型;
        # 混进来会让它跳过排查直接推荐新配方,而用户想知道的是"这次为什么坏了"。
        # get_weather_forecast 给的是风速—— 这一项此前完全没有,
        # 而风是揭膜/撕裂的主要外力。实测第一版诊断只能写"工具未返回风速数据,
        # 大风揭膜无法在此验证,需用户自查",排查在这一支上直接断掉。
        "tools": [
            predict_by_location,
            get_climate_info,
            get_weather_forecast,
            get_soil_info,
            search_knowledge,
        ],
        "prompt": (
            "你是地膜故障诊断专家。用户的地里已经出了问题,你的任务是【找原因】,"
            "不是重新推荐配方。\n"
            "排查方法(按顺序,别跳步):\n"
            "1. **先算基准**:用 predict_by_location 算出该配方在当地、该天数下的"
            "【预期表现】。没有基准就无法判断偏差有多大。\n"
            "2. **比对实际**:用户描述的现象与预期差多少?两种情况要分清 ——\n"
            "   · 实际 ≈ 预期 → 问题出在【用户的预期】,这个配方在当地本来就是这个表现;\n"
            "   · 实际 ≠ 预期 → 才是真异常,继续往下查。\n"
            "3. **逐项排除**,按可能性从大到小,每条都要给判断依据:\n"
            "   · 厚度不足 → 拉伸强度不够,机械铺膜或大风下易破;\n"
            "   · 配方比例 → PLA 占比高则难降解、PBAT 占比高则降解快;\n"
            "   · 环境异常 → 查当地气候(紫外/温度/降水),看是否显著偏离常年;\n"
            "   · **风致破损 → 用 get_weather_forecast 查风速**,大风是揭膜、撕裂的"
            "主要外力,薄膜光老化变脆后尤其怕风;\n"
            "   · 土壤条件 → 查土壤(pH、有机碳),影响微生物降解速率;\n"
            "   · 施工因素 → 铺膜张力、压土、机械划伤(这类查不到数据,只能提示用户自查)。\n"
            "4. **给结论**:指出【最可能的原因】并排序,说明每条的依据和排除理由;"
            "对查不到数据的因素(如施工),明确说这是需要用户自己核实的。\n"
            "诚实原则:信息不足时【明确说缺什么】,不要硬凑一个原因 —— "
            "诊断错了会让用户下一季继续踩同一个坑。不要编造工具没返回的数字。"
        ),
    },
    "数据分析专家": {
        "description": (
            "用户上传了实测数据表时派它:要把实测值和我们的预测模型对比、"
            "找异常记录、判国标符合性,并解读偏差的原因。"
            "**派它时必须在 task 里带上数据集编号(dataset_id)**,否则它读不到数据"
        ),
        # 这三个工具的 schema 共 2181 字,占主 agent 工具总量的 20% ——
        # 而绝大多数用户从不上传数据。放进专家的窄工具集,主 agent 每轮省下这部分。
        # (describe_dataset 留在主 agent:一次调用就出结果,不值得为它起一条嵌套循环。)
        "tools": [
            describe_dataset,
            compare_dataset_with_model,
            detect_dataset_outliers,
            check_dataset_against_standard,
            search_knowledge,
        ],
        "prompt": (
            "你是地膜田间数据分析专家。用户上传了自己的实测记录,"
            "你的任务是把它和我们的预测模型对照,并解释差异。\n"
            "工作方法(按顺序):\n"
            "1. **先看数据本身**:用 describe_dataset 看有哪些列、多少行、缺失多少。"
            "数据质量决定了后面所有结论的可信度 —— 缺失过半的列要明确说它基本无效。\n"
            "2. **排查记录问题**:用 detect_dataset_outliers 找离群行。"
            "离群不等于错误,但结论异常时要先排除「是不是记错了」。\n"
            "3. **比对模型**:用 compare_dataset_with_model 对每个有实测值的指标做对比。"
            "**重点看偏差的方向**:实测整体高于还是低于预测,这比幅度更能指向原因。\n"
            "4. **必要时判国标**:check_dataset_against_standard;"
            "它只判有明确条文的项(目前是厚度),别的指标要查条文就用 search_knowledge。\n"
            "5. **解读**:系统性偏差通常来自【当地条件与训练数据分布不同】"
            "(灌溉方式、覆膜工艺、地方品种),不一定是测量错误;单行的大偏差先怀疑记录。\n"
            "诚实原则:样本量小的时候【明确说样本量小】,不要拿 5 行数据下强结论;"
            "缺失严重的列不要假装它有效。**不要编造工具没返回的数字。**"
        ),
    },
    # 将来加专家就在这加一条,比如:
    # "用量计算专家": {"tools": [estimate_film_usage], "prompt": "你是用量/成本计算专家……"},
}

# 消掉"导入了但暂未使用"的告警;上面注释里的用量专家启用后即用到。
_ = estimate_film_usage


@cache
def _build_expert(name: str, model_name: str = "deepseek-v4-flash"):
    """按注册表造一个专家子 agent。构建一次,之后查缓存复用。

    为什么要缓存:agent 图的编译 + 模型客户端创建,实测每次约 127ms ——
    没缓存时每次派活都重付一遍。deepagents 的 SubAgentMiddleware 也是
    同一做法:在中间件构造时把所有子 agent 编译好,task 工具只查字典。

    复用实例安全吗?安全,关键在下面这条:
    刻意不传 checkpointer:agent 图本身是无状态的(状态全在 invoke 传入的
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


# ── 子 agent 的可观测性 ────────────────────────────────────────
#
# ## 不做这件事会怎样(实测发现的真缺陷)
#
# delegate_to_expert 原本只返回一个字符串,子 agent 的消息从不进入父状态。
# 后果比"少个 id"严重得多:
#
#   · 成本漏算 —— 我们的 Trace 是从父 agent 的消息累加 token 的,
#     子 agent 烧掉的 token 一分钱都没统计进去。评测报的 "5595 tokens/题"
#     只要发生委派就是偏低的,而我们还拿它做版本成本对比。
#   · 内部全黑箱 —— 评测只看见一次 delegate_to_expert,专家在里面调了几次
#     什么工具、转了几圈、慢在哪,全都看不到。排查时无从下手。
#
# deepagents 的 task 工具用三招解决:打追踪标记、挂进父 trace、
# 用 Command 把子 agent 的消息写回父状态。
#
# ## 我们的做法:ContextVar 收集器
#
# 不引入 Command(那要改父状态 schema、牵动整条装配链),而是用一个
# 运行域的收集器:调用方在跑 agent 前放一个空列表进 ContextVar,
# 子 agent 跑完把自己的账单 append 进去,跑完调用方取走。
#
# 为什么可行(已实测):工具跑在线程池里,但 ContextVar 会随上下文复制传进去;
# 而列表是按引用共享的 —— 复制的是变量映射,不是列表本身,
# 所以工具线程里的 append 主线程看得见。
# 并行委派也安全:CPython 里 list.append 是原子的。
_collector: ContextVar[list | None] = ContextVar("subagent_runs", default=None)


@dataclass
class SubAgentRun:
    """一次委派的账单。给可观测性用,不参与业务逻辑。"""

    sub_run_id: str
    expert: str
    task: str  # 截断存,只为排查时能认出是哪一次
    tools: list[str]  # 专家【内部】调了哪些工具 —— 原本完全不可见
    input_tokens: int
    output_tokens: int
    latency_ms: int
    ok: bool
    error: str = ""


@contextmanager
def collect_subagent_runs():
    """把本次运行期间发生的所有委派收集起来。

    用法:
        with collect_subagent_runs() as runs:
            agent.invoke(...)
        # runs 里就是这次运行的全部委派账单
    """
    box: list[SubAgentRun] = []
    token = _collector.set(box)
    try:
        yield box
    finally:
        _collector.reset(token)


def _record(run: SubAgentRun) -> None:
    """记账:进日志(生产可查)+ 进收集器(评测可算)。"""
    logger.info(
        "子agent委派 %s expert=%s tools=%s tokens=%d+%d %dms ok=%s",
        run.sub_run_id,
        run.expert,
        ",".join(run.tools) or "-",
        run.input_tokens,
        run.output_tokens,
        run.latency_ms,
        run.ok,
    )
    box = _collector.get()
    if box is not None:
        box.append(run)


@tool
def delegate_to_expert(expert: str, task: str) -> str:
    """把一个子任务委派给某位专家子智能体,返回其结论。"""
    if expert not in _EXPERTS:
        # 名字不对时,给模型一个能自我纠正的提示,而不是抛异常崩掉
        return f"没有名为「{expert}」的专家。可选:{list(_EXPERTS)}"

    sub_run_id = f"sa-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    agent = _build_expert(expert)
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    except Exception as e:
        _record(
            SubAgentRun(
                sub_run_id,
                expert,
                task[:120],
                [],
                0,
                0,
                int((time.perf_counter() - t0) * 1000),
                False,
                f"{type(e).__name__}",
            )
        )
        raise

    msgs = result.get("messages", [])
    # 把专家内部的账单扒出来 —— 这些信息原本随 result 一起被丢掉了
    tools, tin, tout = [], 0, 0
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            tools.append(tc.get("name", "?"))
        usage = getattr(m, "usage_metadata", None) or {}
        tin += usage.get("input_tokens", 0)
        tout += usage.get("output_tokens", 0)
    _record(
        SubAgentRun(
            sub_run_id,
            expert,
            task[:120],
            tools,
            tin,
            tout,
            int((time.perf_counter() - t0) * 1000),
            True,
        )
    )
    return msgs[-1].content if msgs else ""


# 工具描述从注册表生成,不手写。
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
