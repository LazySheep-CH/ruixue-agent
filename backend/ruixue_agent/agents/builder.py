"""瑞雪 agent 装配:模型 + 工具 + 系统提示 + 会话记忆(checkpointer)+ 中间件链。"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
)

from ruixue_agent.agents.middlewares import (
    MemoryRecallMiddleware,
    PromptInjectionGuardMiddleware,
    SkillInjectionMiddleware,
    TimingLoggingMiddleware,
    ToolErrorHandlingMiddleware,
)
from ruixue_agent.agents.prompt import SYSTEM_PROMPT
from ruixue_agent.checkpointer import get_checkpointer
from ruixue_agent.mcp import load_mcp_tools
from ruixue_agent.models import create_model
from ruixue_agent.subagents import delegate_to_expert
from ruixue_agent.tools import get_tools

# ── 稳定性 / 成本的几个闸门(集中在这里,便于调参)──────────────────
# 单次请求最多让模型说几轮话。agent 是"想→调工具→再想"的循环,
# 万一它钻牛角尖(反复调同一个工具),这个数就是刹车。
# 25 的取法:正常问答 2~5 轮足够,复杂多跳检索约 10 轮;到 25 基本可判定是转圈了。
MAX_MODEL_CALLS_PER_RUN = 25
# 整个会话(同一个 thread_id,可能几十轮对话)累计上限,防长期消耗。
MAX_MODEL_CALLS_PER_THREAD = 200
# 上下文涨到这么多 token 就先做摘要压缩(控成本的关键闸门)。
SUMMARIZE_AT_TOKENS = 50_000
# 压缩时保留最近多少条原始消息(更早的被摘要成一段)。
KEEP_RECENT_MESSAGES = 20


# 需要人工批准才执行的工具(Human-in-the-Loop)。
# 为什么是 delegate_to_expert:它会启动子 agent—— 嵌套的 LLM 循环,是本系统
# 最贵的单次操作。让用户确认"要不要花这个钱做深度分析"是真实的成本控制,
# 而不是为了用而用。工具全为只读,故不是出于"防破坏"的考虑。
# 默认关闭:开启会改变 /chat 的返回形态(需前端配合批准流程),按需启用。
REQUIRE_APPROVAL_TOOLS = ["delegate_to_expert"]


# 用一个哨兵对象区分"没传"和"显式传 None"。
# 不能用 None 当默认值:那样就没法表达"我不要 checkpointer"了。
_DEFAULT_CKPT = object()


def create_ruixue_agent(
    model_name: str = "deepseek-v4-pro",
    summary_model_name: str = "deepseek-v4-flash",
    require_approval: bool = False,
    checkpointer=_DEFAULT_CKPT,
    **model_overrides,
):
    """按配置装配并返回瑞雪 agent。

    summary_model_name 单独给便宜模型:摘要是内部动作、用户看不到,
    没必要用贵的 pro —— 同样的活,flash 便宜一个量级。

    checkpointer 可替换的原因:评测要跑几百个一次性会话,用生产的 PG
    checkpointer 会把评测数据写进生产表 —— 评测不该污染线上数据。
    评测传内存版即可(见 ruixue_agent/eval/runner.py)。

    model_overrides 透传给模型构造(如 temperature=0)。评测必须压温度,
    否则量到的是采样噪声而不是能力 —— 实测默认温度下同版本极差 12.1%。
    """
    model = create_model(model_name, **model_overrides)
    # 主 agent 的工具 = 基础叶子工具 + 委派工具(多 Agent)。
    # delegate 只给主 agent、不给专家(专家用各自窄工具集),专家因此无法再派活(防递归)。
    # 工具 = 内置叶子工具 + 委派工具 + 外部 MCP 工具(配置了才有,连不上则为空)。
    # MCP 让"加工具"从改代码变成改配置。
    tools = [*get_tools(), delegate_to_expert, *load_mcp_tools()]

    return create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        # 会话状态持久化到 PostgreSQL:重启不丢、多 worker 共享(按 thread_id 存取)
        checkpointer=(get_checkpointer() if checkpointer is _DEFAULT_CKPT else checkpointer),
        middleware=_build_middleware(summary_model_name, require_approval),
    )


def _build_middleware(summary_model_name: str, require_approval: bool) -> list:
    """中间件链:顺序即设计,不是随便排的(每条的理由见注释)。"""
    chain = [
        # 0) 提示注入防护:放最前—— 安全检查要在花钱(摘要/调模型)之前做,
        #    且要在最靠近用户输入的位置,后续中间件都在其保护之下。
        PromptInjectionGuardMiddleware(),
        # 0.5 作业规程注入:按提问场景注入已验证的 SOP(skills/*.md),
        #     告诉模型"这类问题该怎么做"。每条规程一个会话只注入一次,
        #     但不限定首轮 —— 限定首轮的话,"你好"开场就会让它永远等不到注入。
        SkillInjectionMiddleware(),
        # 0.6 长期记忆注入:技能是"这类问题该怎么做"(对所有人一样),
        #     记忆是"这个用户是谁"(因人而异)。两者都属于"开工前先给背景",
        #     所以挨着放。去重判据同样是"注入过没有",不是"第几轮"——
        #     记忆最该发挥作用的"还是上次那块地",几乎不会是第一句话。
        MemoryRecallMiddleware(),
        # 1) 死循环刹车:已经到上限就该立刻停,不能先花钱做摘要、再发现"哦该停了"。
        ModelCallLimitMiddleware(
            run_limit=MAX_MODEL_CALLS_PER_RUN,
            thread_limit=MAX_MODEL_CALLS_PER_THREAD,
            exit_behavior="end",  # "end"=优雅收尾出答案;"error"=直接抛异常
        ),
        # 2) 上下文压缩:过了闸门就把老对话摘要成一段,替换掉原文。
        #    放在模型调用之前,压完再发给模型,省的就是这次的钱。
        SummarizationMiddleware(
            model=create_model(summary_model_name),  # 用便宜模型干这活
            trigger=("tokens", SUMMARIZE_AT_TOKENS),
            keep=("messages", KEEP_RECENT_MESSAGES),
        ),
        # 3) 模型调用重试:网络抖动、对方 503 —— 退避后重试,别让用户白跑一趟。
        #    jitter=True 是"随机抖动":避免所有失败请求在同一刻一起重试,
        #    那样等于自己给对方来一波脉冲(惊群),越重试越挂。
        ModelRetryMiddleware(max_retries=2, initial_delay=1.0, backoff_factor=2.0, jitter=True),
    ]
    if require_approval:
        # 3.5 人工批准:在工具执行前暂停,等外部 resume。放在重试之后、
        #      计时之前 —— 等待人批准的时间不该被算进"工具耗时"。
        chain.append(
            HumanInTheLoopMiddleware(
                interrupt_on=dict.fromkeys(REQUIRE_APPROVAL_TOOLS, True),
                description_prefix="该操作会启动子智能体(成本较高),需要你确认",
            )
        )
    chain += [
        # 4) 工具计时:放在错误处理外层,这样失败的调用也能被记进耗时日志。
        TimingLoggingMiddleware(),
        # 5) 工具错误降级:最内层,离真正的工具执行最近,第一时间接住异常。
        ToolErrorHandlingMiddleware(),
    ]
    return chain
