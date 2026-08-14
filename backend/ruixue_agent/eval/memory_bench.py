"""长期记忆收益实验:量化"有记忆"比"没记忆"好多少。

## 要回答的问题

我们建了长期记忆(抽取事实 → PG 存权威 → 向量召回 → 下一轮注入),
但**它到底有没有用、有多大用**,此前没有数字。没有数字的模块,
在评审和评审里都等同于"听起来有用"。

## 实验设计:一次只变一个东西

每道题两轮,跑在【不同 thread_id、同一 user_id】下:

    setup 轮   thread = "<user>:s1"   用户陈述一个只有他自己知道的事实
    probe 轮   thread = "<user>:s2"   一个必须用上那个事实才能答好的问题

**换 thread 是这个实验成立的前提。** 同一个 thread 里,checkpointer 会把
上下文原样带过去 —— 那测的是短期记忆,和长期记忆一点关系没有。换了 thread
之后上下文归零,probe 轮唯一可能的信息来源就是长期记忆。

两个 arm 的唯一差别是 **recall 有没有返回内容**:

    on  组:正常召回
    off 组:把 recall 打成返回空

刻意**不去掉中间件本身**:去掉了就同时改变了中间件链、token 数、执行路径,
测出来的差值不知道该归给谁。只掐掉召回结果,变量才只有一个。
两个 arm 连"写记忆"这一步都照跑 —— 反正 off 组读不到,留着能保证两条流水线
完全一致。

## 两个 arm 必须用不同的 user_id

记忆是**软删 + 内容寻址**的(删过的事实不会复活,见 memory 模块)。
如果两个 arm 共用 user_id、中间靠删除来重置,第二个 arm 会因为
"这条删过"而永远存不进去 —— 于是 on 组也读不到记忆,**实验静默失效,
测出来的收益是 0**,而你会以为记忆没用。

所以 arm 之间用 `<user>-on` / `<user>-off` 两个身份,互不干扰,也不用删。

## 出题的铁律

probe 里**绝不能**重复 setup 里的关键信息 —— 否则不用记忆也能答对,
这道题的收益恒为 0。这类错误不会报错,只会**静默稀释真实收益**,
所以在加载期就校验死(见 `_validate`)。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ruixue_agent.eval.schema import KeyPoint
from ruixue_agent.eval.scoring import match_keypoint


@dataclass(frozen=True)
class MemoryCase:
    id: str
    user: str
    setup: str  # 第一轮:陈述事实
    probe: str  # 第二轮:必须用上该事实的问题
    keypoints: tuple[KeyPoint, ...] = ()
    note: str = ""


@dataclass
class ArmResult:
    """一道题在一个 arm 下的结果。"""

    case_id: str
    with_memory: bool
    answer: str = ""
    passed: bool = False
    hit: tuple[str, ...] = ()  # 命中的要点
    missed: tuple[str, ...] = ()  # 漏掉的要点
    injected: bool = False  # 记忆【真的】进上下文了吗(扫 MEMORY_HEADER,不是另调 recall)
    asked_back: bool = False  # 反问用户要信息 = 没用上记忆
    stored: int = 0  # setup 轮实际存进去几条事实
    visible_after_s: float = 0.0  # 新记忆写入后多久才可召回(Milvus flush 延迟)
    tool_args: tuple[str, ...] = ()  # probe 轮的全部工具调用参数(判据改了能离线重算)
    error: str = ""


@dataclass
class BenchReport:
    on: list[ArmResult] = field(default_factory=list)
    off: list[ArmResult] = field(default_factory=list)

    def summary(self) -> dict:
        def rate(rs: list[ArmResult]) -> float:
            ok = [r for r in rs if not r.error]
            return sum(r.passed for r in ok) / len(ok) if ok else 0.0

        on_rate, off_rate = rate(self.on), rate(self.off)
        return {
            "n": len(self.on),
            "off_rate": off_rate,
            "on_rate": on_rate,
            "delta": on_rate - off_rate,
            "errors": sum(1 for r in self.on + self.off if r.error),
            # 没存进任何事实的题 —— 这类题的收益必然是 0,但原因在【抽取】不在【召回】,
            # 混进总分会让人以为"记忆没用",实际是"根本没记住"。必须单列。
            "no_fact_stored": sum(1 for r in self.on if r.stored == 0 and not r.error),
            # 存了却没进上下文 —— 第三种病:抽取对了、召回或注入断了。
            # 三种病(没抽到 / 没注入 / 注入了没用上)解法完全不同,不能混成一个数。
            "stored_not_injected": sum(
                1 for r in self.on if r.stored and not r.injected and not r.error
            ),
            "injected": sum(1 for r in self.on if r.injected and not r.error),
            # 反问率:诊断用,【不进总分】。反问不一定是失败 ——
            # 问一个记忆里从来没有的信息(如 m06 的地点)是正确行为。
            "on_asked_back": sum(1 for r in self.on if r.asked_back and not r.error),
            "off_asked_back": sum(1 for r in self.off if r.asked_back and not r.error),
        }


# 等新写入的记忆变为可召回的上限。实测最坏 11.4s,留一倍余量。
# 等不到就继续跑 —— 那道题会以"存了但没注入"出现在报告里,是有用的信号,
# 比在这儿卡死好。
VISIBILITY_TIMEOUT_S = 25.0


def _wait_visible(recall_fn, user_id: str, query: str) -> float:
    """轮询到记忆可召回为止,返回等了多少秒(等不到返回 -1)。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < VISIBILITY_TIMEOUT_S:
        if recall_fn(user_id, query):
            return round(time.monotonic() - t0, 2)
        time.sleep(0.3)
    return -1.0


def _kp_used(kp: KeyPoint, turn: Turn) -> bool:
    """要点是否【真的被用上】。

    ## 数值要点:只认工具调用参数,不看正文(判据 v3)

    正文里的数字**分不清是"用上了"还是"举例说明"**:

        m05 的 off 组反问时列举「棉花约150天、玉米约120天」→ 正则命中 120
        m10 的 off 组反问时举例「比如:"新疆尉犁县,棉花,100亩"」→ 命中"尉犁"

    两条都在反问(=没用上记忆),却被判成用上了。方向还是固定的:
    **只抬高 off 组,从而系统性低估记忆的收益**。

    而 `estimate_film_usage(area_mu=50)` 这样的工具调用参数是铁证 ——
    模型不会"举例调一次工具"。所以数值要点**只认参数**,正文一律不算。

    代价说清楚:agent 复述了"您的 50 亩棉花地"但最终没调工具,会被判为
    没用上。这个取舍是**故意**的 —— 记忆的价值在于让它把活干完,
    知道了却不用,和不知道的区别不大。

    ## 文字要点:看正文

    "推荐理由要落在保墒上"这类没法从参数里读,只能看正文。误判风险存在
    但方向可控:off 组要泛泛提到"保墒"也得先谈到这个维度,不像数字那样
    随便举个例就撞上。
    """
    if kp.value is not None:
        return any(
            abs(float(tok) - kp.value) <= (kp.tol or 0)
            for a in turn.tool_args
            for tok in re.findall(r"\d+(?:\.\d+)?", a)
        )
    return match_keypoint(kp, turn.answer)


# ── 加载与校验 ────────────────────────────────────────────────


def _validate(c: MemoryCase) -> None:
    """probe 里不许泄露 setup 的关键信息 —— 否则这道题恒为 0 收益且静默。"""
    for kp in c.keypoints:
        if kp.value is not None:
            # 数字要点:probe 里出现同一个数,就说明问题自带答案
            for tok in re.findall(r"\d+(?:\.\d+)?", c.probe):
                if abs(float(tok) - kp.value) <= (kp.tol or 0):
                    raise ValueError(
                        f"用例 {c.id}:probe 里出现了要点数值 {kp.value} —— "
                        f"不用记忆也能答对,这道题测不出任何收益"
                    )
        elif kp.pattern and re.search(kp.pattern, c.probe):
            raise ValueError(f"用例 {c.id}:probe 命中了要点正则「{kp.pattern}」—— 问题自带答案")
    if not c.keypoints:
        raise ValueError(f"用例 {c.id}:没有 keypoints,无法判分")


def load_memory_cases(path: str | Path) -> list[MemoryCase]:
    cases: list[MemoryCase] = []
    seen: set[str] = set()
    for ln, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        try:
            d = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"第 {ln} 行不是合法 JSON:{e}") from e
        if d["id"] in seen:
            raise ValueError(f"id 重复:{d['id']} —— 报告里会互相覆盖")
        seen.add(d["id"])
        c = MemoryCase(
            id=d["id"],
            user=d["user"],
            setup=d["setup"],
            probe=d["probe"],
            keypoints=tuple(KeyPoint(**k) for k in d.get("keypoints", [])),
            note=d.get("note", ""),
        )
        _validate(c)
        cases.append(c)
    if not cases:
        raise ValueError(f"{path} 里一道题都没有")
    return cases


# ── 跑一个 arm ────────────────────────────────────────────────


@dataclass
class Turn:
    """一轮对话的观测结果 —— 判分要用的东西全在这里。"""

    answer: str = ""
    injected: bool = False  # 记忆【真的】进上下文了吗
    tool_args: tuple[str, ...] = ()  # 所有工具调用参数的字符串形式
    asked_back: bool = False  # 是不是在反问用户要信息


def _ask(agent, user_id: str, thread_suffix: str, question: str) -> Turn:
    """跑一轮对话并观测。thread_id 形如 "<user>:<suffix>"。

    ## injected 为什么扫消息而不是再调一次 recall

    "记忆有没有生效"的**唯一真相**是它在不在上下文里。另调一次 recall 只能
    说明"现在能召回",不代表刚才那轮注入了 —— 实测这两者会不一致
    (刚写入的向量有短暂的可见性延迟,而中间件晚几百毫秒执行反而召回成功),
    于是诊断字段会撒谎:显示"召回 0 条"但这道题其实是靠记忆过的。

    ## tool_args 为什么要收

    判"有没有用上 50 亩"这件事,**工具调用参数是结构证据,答案文字只是旁证**。
    模型可能在正文里举例说"比如 50 亩",那不算用上;但如果它调了
    `estimate_film_usage(area_mu=50)`,那就是铁证。
    """
    from ruixue_agent.agents.middlewares import MEMORY_HEADER
    from ruixue_agent.eval.scoring import _CLARIFY_RE

    cfg = {
        "configurable": {"thread_id": f"{user_id}:{thread_suffix}-{uuid.uuid4().hex[:6]}"},
        "recursion_limit": 40,
    }
    state = agent.invoke({"messages": [{"role": "user", "content": question}]}, cfg)
    msgs = state.get("messages") or []

    t = Turn()
    args: list[str] = []
    for m in msgs:
        content = str(getattr(m, "content", ""))
        if MEMORY_HEADER in content:
            t.injected = True
        for tc in getattr(m, "tool_calls", None) or []:
            args.append(json.dumps(tc.get("args", {}), ensure_ascii=False))
    t.tool_args = tuple(args)

    for m in reversed(msgs):
        if type(m).__name__ == "AIMessage" and not getattr(m, "tool_calls", None):
            t.answer = str(m.content)
            break
    t.asked_back = bool(_CLARIFY_RE.search(t.answer))
    return t


def run_arm(agent, case: MemoryCase, with_memory: bool, run_tag: str = "") -> ArmResult:
    """跑一道题的一个 arm:setup → 写记忆 → probe → 判分。

    run_tag 给每次实验一个独立的记忆命名空间。**没有它这个实验不可重复**:
    记忆是软删 + 内容寻址的(删过的不复活),所以既不能靠删除重置,
    跑第二轮时上一轮的记忆还在 —— 召回条数越跑越多,而且会混进
    调试时留下的脏数据。换个 user_id 前缀是最干净的隔离。
    """
    import ruixue_agent.memory as mem
    from ruixue_agent.memory.extract import extract_facts

    arm = "on" if with_memory else "off"
    user_id = f"{case.user}-{run_tag}-{arm}" if run_tag else f"{case.user}-{arm}"
    res = ArmResult(case_id=case.id, with_memory=with_memory)

    real_recall = mem.recall
    try:
        # ① setup 轮:让用户陈述事实。这一轮不判分,只为了产生可抽取的对话。
        #    setup 轮也要掐掉召回 —— 否则 on 组的第一轮就先注入了一次记忆,
        #    两个 arm 的第一轮就不一样了。
        mem.recall = lambda uid, q: []
        setup_turn = _ask(agent, user_id, "s1", case.setup)

        # ② 写记忆:和生产同一条路径(main.py::_remember_async),不另写一套。
        facts = extract_facts(case.setup, setup_turn.answer)
        # stored == 0 会在报告里单列("瓶颈在抽取层")—— 不能混进"记忆没用"里。
        res.stored = mem.remember(user_id, facts) if facts else 0

        # ②.5 等新记忆变得可召回。**这一步不是实验的一部分,是在排除干扰。**
        #
        # 实测:Milvus 写入后到能被搜到,延迟 0.12s ~ 11.43s(三次采样,抖动极大)——
        # 向量插入要等一次 flush 才进可搜索段。不等的话,probe 轮召回为空,
        # 测出来的是【Milvus 的刷盘时机】,不是【记忆的价值】。
        #
        # 只在 on 组等:off 组的 recall 被打成返回空,等多久都一样。
        if with_memory and res.stored:
            res.visible_after_s = _wait_visible(real_recall, user_id, case.probe)

        # ③ probe 轮:换 thread → 上下文归零 → 只有长期记忆能救。
        if with_memory:
            mem.recall = real_recall
        turn = _ask(agent, user_id, "s2", case.probe)
        res.answer = turn.answer
        res.injected = turn.injected
        res.asked_back = turn.asked_back
        # 原始轨迹要留下来 —— **判据会改,轨迹不该重跑**。
        # v2 那次改判据时因为没存 tool_args,只能把 48 次对话整个重跑一遍。
        res.tool_args = turn.tool_args

        hit, missed = [], []
        for kp in case.keypoints:
            label = kp.note or (kp.pattern or str(kp.value))
            (hit if _kp_used(kp, turn) else missed).append(label)
        res.hit, res.missed = tuple(hit), tuple(missed)

        # 【判据 v3,2026-08-12】只看"要点有没有被用上",反问【不】单独致命。
        #
        # v2 加过"反问即失败",实测太粗:m07 明明完美用上了记忆
        # (「结合您的实际情况:春季风大、地膜曾被吹烂」,要点全中),
        # 只因正文里有个问号就被判失败 —— 这次是**压低 on 组**,方向反了。
        #
        # 而且"反问"本身不一定是失败:m06 用上了偏好(保墒),但还要问地点
        # ——地点从来就不在记忆里,问它是【正确行为】。
        #
        # 防"举例命中"这件事,交给 _kp_used 的结构性判据(数值只认工具参数),
        # 不再靠反问这条粗规则兜。反问率单独作为诊断指标报出来,不进总分。
        res.passed = bool(case.keypoints) and not missed
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"[:200]
    finally:
        mem.recall = real_recall
    return res


def run_bench(agent, cases: list[MemoryCase], on_case=None, run_tag: str = "") -> BenchReport:
    """跑完整实验。**off 组先跑** —— 顺序固定,免得有人怀疑是热身效应。

    run_tag 缺省自动生成:每次跑都是全新的记忆命名空间,历史数据不会串进来。
    """
    run_tag = run_tag or uuid.uuid4().hex[:8]
    rep = BenchReport()
    for i, c in enumerate(cases, 1):
        off = run_arm(agent, c, with_memory=False, run_tag=run_tag)
        on = run_arm(agent, c, with_memory=True, run_tag=run_tag)
        rep.off.append(off)
        rep.on.append(on)
        if on_case:
            on_case(i, len(cases), c, off, on)
    return rep
