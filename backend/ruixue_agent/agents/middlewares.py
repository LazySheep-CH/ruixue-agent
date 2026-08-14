"""Agent 中间件:横切关注点(计时/日志/错误处理…),套在模型/工具调用外,不动业务逻辑。

create_agent(middleware=[...]) 接收它们。参考架构 的中间件链也是这么叠的。
钩子(和你 mini-harness 写过的一样):
    before_model(state, runtime)      模型调用前
    after_model(state, runtime)       模型调用后
    wrap_tool_call(request, handler)  包裹工具执行(handler(request) 才是真正执行)
"""

import logging
import time

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage

from ruixue_agent.guardrails.injection import REINFORCE_NOTICE, detect_injection
from ruixue_agent.skills import injected_names, render, select_skills

logger = logging.getLogger("ruixue.agent")


class TimingLoggingMiddleware(AgentMiddleware):
    """记录每次工具调用的名称与耗时 —— 可观测性的起点。

    有了它,一条请求里 agent 调了哪些工具、各花多久,日志里一目了然,
    排查"为什么这次慢""RAG 是不是超时"就有据可依。
    """

    def wrap_tool_call(self, request, handler):
        t0 = time.perf_counter()
        name = request.tool_call["name"]
        result = handler(request)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("工具 %s 耗时 %.0fms", name, ms)
        return result


class PromptInjectionGuardMiddleware(AgentMiddleware):
    """提示注入防御:检测用户消息里的注入企图,记日志并重申系统规则。

    为什么【不直接拒绝】:正则判定必然有误差,把真实用户挡在门外的代价,
    远大于让一次注入企图通过(何况本 agent 的工具全是只读的,破坏上限有限)。
    故采取"检测 → 告警留痕 → 在上下文里重申身份与边界"的策略 ——
    既不误伤,又让模型带着防御提示去回答。

    留痕很重要:安全事件要能在日志里查得到(配合 request_id 可定位到具体用户/请求)。
    """

    def before_model(self, state, runtime):
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        # 只检最后一条【用户】消息:历史消息已检过,工具返回的内容由 RAG 侧边界隔离
        if type(last).__name__ != "HumanMessage":
            return None
        threats = detect_injection(str(last.content))
        if not threats:
            return None
        logger.warning("检测到疑似提示注入: %s", threats)
        # 追加一条系统消息重申边界(不改用户原文,便于排查时看到真实输入)
        return {"messages": [SystemMessage(content=REINFORCE_NOTICE)]}


class SkillInjectionMiddleware(AgentMiddleware):
    """按场景注入【作业规程(技能)】:告诉模型"这类问题该怎么做"。

    工具解决"能做什么",技能解决"该怎么做"——领域经验(先看生育期、三项指标
    如何权衡、有哪些坑)写死在系统提示会越堆越长、稀释注意力;写进代码更糟,
    因为它要由领域专家反复迭代。故做成 skills/*.md,按关键词命中才注入。

    ## 每条规程只注入一次,但【不限定在首轮】

    早先的写法是"只在首轮注入",看起来省 token,实际有个致命漏洞:

        用户:你好              ← 首轮,但没命中关键词 → 不注入
        助手:你好,我是……
        用户:帮我选个配方       ← 命中了,却已经不是首轮 → 【永远不会注入】

    只要开场寒暄一句,整个技能系统就失效了 —— 而寒暄开场恰恰是最常见的用法。
    "省 token"的本意是"同一份别重复塞",不是"错过就再也不给"。

    所以判据换成:**这条规程在本会话里注入过吗?** 没有就注入,不管第几轮。
    判断方式是扫历史里的 `【作业规程:名称】` 标题(见 skills/loader.py)——
    不额外存状态,消息列表本身就是唯一事实来源。
    """

    def before_model(self, state, runtime):
        messages = state.get("messages") or []
        last = _last_human(messages)
        if last is None:
            return None

        hit = select_skills(str(last.content))
        if not hit:
            return None  # 这次提问没命中任何规程

        # 扫【全部】历史(含本轮已注入的),看哪些规程已经在上下文里了。
        # 用 messages[:-1] 会漏掉刚注入的那条 —— 配合 _last_human 之后
        # 本方法在同一轮里可能被调用多次,漏扫就会重复注入。
        already: set[str] = set()
        for m in messages:
            already |= injected_names(str(getattr(m, "content", "")))
        fresh = [s for s in hit if s.name not in already]
        if not fresh:
            return None  # 命中的这几条都已经注入过,不重复塞

        logger.info("注入作业规程(技能): %s", [s.name for s in fresh])
        return {"messages": [SystemMessage(content=render(fresh))]}


# 记忆注入的标题。和技能的 SKILL_HEADER 同理:它是【约定好的标记】,
# 中间件靠它判断"本会话注入过没有"。改了这行就等于把去重判据改了。
MEMORY_HEADER = "【关于这位用户的已知背景】"


class MemoryRecallMiddleware(AgentMiddleware):
    """把这个用户的相关长期记忆注入对话。

    ## 位置:紧跟在技能注入之后

    技能是"这类问题该怎么做"(对所有人一样);记忆是"这个用户是谁"(因人而异)。
    两者都属于"开工前先给背景",所以挨着放。

    ## 只注入一次,但【不限定在首轮】

    同一会话里记忆进过一次上下文就够了,每轮再塞一遍纯属浪费 token,
    还会让同样的内容出现好几次、稀释注意力。但"只注入一次"不等于"只在首轮":

        用户:你好                        ← 首轮,recall("你好") 召不回什么 → 不注入
        助手:你好……
        用户:还是上次那块地,再帮我看看     ← 【最需要记忆的一句】
                                          ← 若限定首轮,就永远等不到注入了

    记忆最该发挥作用的时刻,恰恰是用户说"上次""还是那个"的时候 ——
    而那几乎不可能是第一句话。所以判据是"注入过没有",不是"第几轮"。
    (技能注入踩过同一个坑,见 SkillInjectionMiddleware 的说明。)

    相关性由 recall() 自己按当前问题判断,召不回就不注入 —— 不需要靠"首轮"来兜。

    ## 用 SystemMessage 而不是伪装成用户说的话

    有人会把记忆拼进用户消息里假装"用户刚说的"。那样做有两个坏处:
      ① 模型分不清哪句是用户【现在】说的、哪句是系统补的背景;
      ② 万一记忆里被写入了恶意内容,它就以"用户指令"的身份进了上下文 ——
         而用户消息的指令权重远高于系统背景。
    所以必须作为系统背景注入,并明确标注"这是历史背景,不是本次请求"。

    ## 取不到 user_id 就不注入

    记忆是【按用户隔离】的。拿不到身份时宁可不注入,绝不能"给个默认用户" ——
    那会让所有匿名请求共用一份记忆,是数据泄露。
    """

    def before_model(self, state, runtime):
        messages = state.get("messages") or []
        last = _last_human(messages)
        if last is None:
            return None
        # 本会话是否已经注入过记忆?靠扫历史里的标题判断,不额外存状态。
        # 扫【全部】消息:配合 _last_human,本方法在同一轮里可能被调用多次。
        if any(MEMORY_HEADER in str(getattr(m, "content", "")) for m in messages):
            return None

        user_id = _user_id_from(runtime)
        if not user_id:
            return None  # 没身份不注入 —— 见类文档

        from ruixue_agent.memory import recall

        rows = recall(user_id, str(last.content))
        if not rows:
            return None
        lines = "\n".join(f"- [{r.kind}] {r.text}" for r in rows)
        logger.info("注入长期记忆 %d 条", len(rows))
        return {
            "messages": [
                SystemMessage(
                    # 用同一个 MEMORY_HEADER 常量拼,不硬编码 —— 第 172 行的去重
                    # 判据认的就是它。两处各写一份字面量,改一处忘一处就静默失效。
                    content=(
                        f"{MEMORY_HEADER}(来自以往对话,仅供参考,"
                        "不是本次请求的内容;与本次问题冲突时以本次为准):\n" + lines
                    )
                )
            ]
        }


def _last_human(messages):
    """取【最后一条用户消息】;没有就返回 None。

    ## ⚠ 2026-08-12 修:不能用 messages[-1]

    原来两个注入中间件都写的是:

        last = messages[-1]
        if type(last).__name__ != "HumanMessage":
            return None

    这个判据默认"用户消息永远在最后"。但**前面的中间件会往消息尾部追加
    SystemMessage** —— 技能注入干的就是这件事。链的顺序是
    技能 → 记忆,所以只要这一轮命中了技能:

        messages[-1] 变成技能的 SystemMessage
            → 记忆中间件的判据不成立
            → 直接 return None
            → **记忆永远注不进去**

    实测(2026-08-12):

        "帮我算一下要买多少地膜"      不触发技能 → 记忆注入 ✅
        "帮我在赤峰选个合适的配方"    触发技能   → 记忆被挡 ❌

    最讽刺的是**最需要记忆的问题(选型、配方、推荐)恰恰最容易触发技能** ——
    这个 bug 精准地打掉了记忆最有价值的那部分场景。

    改成"往回找最后一条 HumanMessage",中间件之间就不再靠"谁在最后"这种
    隐式约定耦合。配套地,去重判据要扫**全部**消息(见两处调用点)。
    """
    for m in reversed(messages):
        if type(m).__name__ == "HumanMessage":
            return m
    return None


def _user_id_from(runtime) -> str:
    """从运行配置里取 user_id。

    我们的 thread_id 形如 "alice:t1"(见 main.py 的命名空间隔离),
    所以用户身份可以从它前缀还原 —— 不必再单独传一个参数。

    ## ⚠ 2026-08-12 修:原来读 `runtime.config`,而它根本不存在

    LangGraph 的 `Runtime` **没有 `config` 属性**(官方文档明确写着
    "Runtime does not include config",要用 `langgraph.config.get_config()`)。
    原实现 `getattr(runtime, "config", None) or {}` 于是恒为 `{}`,
    user_id 恒为空字符串,`MemoryRecallMiddleware` 直接 return None ——
    **长期记忆从上线起就没注入过一次**。

    为什么一直没被发现,有两层原因,都值得记:

    ① **失败方式是"什么都不做"**:没身份就不注入,这条分支本身是对的
       (宁可不给,不可给错人),所以没有任何报错、没有异常日志,
       只是记忆功能静静地不生效。

    ② **测试是假的**:`tests/test_memory.py` 里的 `_Rt` 是手搓的假 runtime,
       特意带了 `.config`。它测的是"如果 runtime 长这样,逻辑对不对",
       而真 runtime 不长这样 —— 于是测试全绿、功能全废。
       和 SkillInjectionMiddleware 那次是同一类:**假的输入喂出假的信心**。
       现在补了一条走真 agent 的集成测试(test_memory_injection_integration),
       用假模型不花钱,但 runtime 是真的。

    两条来源都试:`get_config()` 是正路;`runtime.config` 留着兼容
    (以及让手搓 runtime 的单元测试仍能用)。
    """
    thread_id = ""
    try:
        from langgraph.config import get_config

        thread_id = ((get_config() or {}).get("configurable") or {}).get("thread_id", "")
    except Exception:
        # 不在图的执行上下文里调用时 get_config() 会抛 —— 属正常,退回下面那条
        pass
    if not thread_id:
        try:
            cfg = getattr(runtime, "config", None) or {}
            thread_id = (cfg.get("configurable") or {}).get("thread_id", "")
        except Exception:
            return ""
    return thread_id.split(":", 1)[0] if ":" in thread_id else ""


# 工具失败消息的机器可读标记。
#
# 为什么要单独抽一个常量:评测需要区分【工具挂了】和【模型没选对工具】——
# 前者是环境问题(修 Milvus),后者是能力问题(改提示词/工具描述)。两者混在
# 一起会把优化方向带偏。靠在评测里硬编码这句话的措辞太脆(改个字就失效),
# 所以在产生它的地方定义常量,消费方(eval/trace.py)引用同一个。
TOOL_FAILURE_MARKER = "[TOOL_FAILED]"


class ToolErrorHandlingMiddleware(AgentMiddleware):
    """工具执行失败时优雅降级:不让单个工具异常拖垮整个对话。

    工具会失败(RAG 断连、坏输入抛异常…)。若异常一路往上冒,整个请求崩、
    用户得到 500。这里在 wrap_tool_call 里兜住异常,记 ERROR 日志,并返回一条
    "工具失败"的 ToolMessage —— agent 收到它,能回一句"知识库暂时不可用",
    而不是整个崩掉。这就是"故障隔离 + 优雅降级"。
    """

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            name = request.tool_call["name"]
            # 完整详情(含堆栈)只进【服务端日志】,给你排查用。
            # logger.exception 比 logger.error 多带堆栈,能看到错在哪一行。
            logger.exception("工具 %s 失败", name)
            # 【错误脱敏】只把异常【类型名】给模型,不给 str(e) 的具体内容 ——
            # 原始文字可能含内网 IP、数据库地址等,进了模型上下文可能被复述给用户。
            # 给类型是为了让模型能贴切回应(连不上 vs 参数错),又不泄露任何细节。
            safe_reason = type(e).__name__
            return ToolMessage(
                content=(
                    f"{TOOL_FAILURE_MARKER} 工具 {name} 执行失败({safe_reason}),"
                    f"请告知用户该功能暂时不可用,不要编造结果。"
                ),
                tool_call_id=request.tool_call["id"],
            )
