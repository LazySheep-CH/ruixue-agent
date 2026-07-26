"""瑞雪 agent 装配:模型 + 工具 + 系统提示 + 会话记忆(checkpointer)+ 中间件链。"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
)

from ruixue_agent.agents.middlewares import (
    TimingLoggingMiddleware,
    ToolErrorHandlingMiddleware,
)
from ruixue_agent.agents.prompt import SYSTEM_PROMPT
from ruixue_agent.checkpointer import get_checkpointer
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


def create_ruixue_agent(
    model_name: str = "deepseek-v4-pro",
    summary_model_name: str = "deepseek-v4-flash",
):
    """按配置装配并返回瑞雪 agent。

    summary_model_name 单独给【便宜模型】:摘要是内部动作、用户看不到,
    没必要用贵的 pro —— 同样的活,flash 便宜一个量级。
    """
    model = create_model(model_name)
    # 主 agent 的工具 = 基础叶子工具 + 委派工具(多 Agent)。
    # delegate 只给主 agent、不给专家(专家用各自窄工具集),专家因此无法再派活(防递归)。
    tools = [*get_tools(), delegate_to_expert]
    return create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        # 会话状态持久化到 PostgreSQL:重启不丢、多 worker 共享(按 thread_id 存取)
        checkpointer=get_checkpointer(),
        # ── 中间件链:顺序即设计,不是随便排的(理由见下方注释)──
        middleware=[
            # ① 死循环刹车:放【最前】。已经到上限就该立刻停,
            #    不能先花钱做摘要、再发现"哦该停了"。省钱的顺序。
            # ===== (你写一行)=====
            #   ModelCallLimitMiddleware(
            #       run_limit=MAX_MODEL_CALLS_PER_RUN,
            #       thread_limit=MAX_MODEL_CALLS_PER_THREAD,
            #       exit_behavior="end",   # "end"=优雅收尾出答案;"error"=直接抛异常
            #   ),
            ModelCallLimitMiddleware(
                run_limit=MAX_MODEL_CALLS_PER_RUN,
                thread_limit=MAX_MODEL_CALLS_PER_THREAD,
                exit_behavior="end",
            ),
            # ② 上下文压缩:过了闸门就把老对话摘要成一段,替换掉原文。
            #    放在模型调用【之前】,压完再发给模型,省的就是这次的钱。
            SummarizationMiddleware(
                model=create_model(summary_model_name),  # 用便宜模型干这活
                trigger=("tokens", SUMMARIZE_AT_TOKENS),
                keep=("messages", KEEP_RECENT_MESSAGES),
            ),
            # ③ 模型调用重试:网络抖动、对方 503 —— 退避后重试,别让用户白跑一趟。
            #    jitter=True 是"随机抖动":避免所有失败请求在同一刻一起重试,
            #    那样等于自己给对方来一波脉冲(惊群),越重试越挂。
            ModelRetryMiddleware(max_retries=2, initial_delay=1.0, backoff_factor=2.0, jitter=True),
            # ④ 工具计时:放在错误处理【外层】,这样失败的调用也能被记进耗时日志。
            TimingLoggingMiddleware(),
            # ⑤ 工具错误降级:最内层,离真正的工具执行最近,第一时间接住异常。
            ToolErrorHandlingMiddleware(),
        ],
    )
