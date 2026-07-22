"""查询改写:把农户口语问题改写成术语规范、适合向量检索的查询。

真实 agent 走 RAG 前会先理解并改写用户的大白话,因此检索链路把"改写"作为
标准预处理层。现役 embedding(bge-small-zh)对术语敏感、对口语弱,一线用户
"浇点水管用不"这类问法与文献术语存在词汇鸿沟,改写正是补这条鸿沟。

两段式(闸门与改写解耦,实测比单 prompt 自判断更稳):
  ① 闸门:先判"这条查询是不是口语/模糊"。术语已规范的直接放过 ——
     误改一条规范查询会拖累检索(fact −0.05、multihop −0.22),漏改只是少赚,故从严。
  ② 改写:仅当判为口语时,专职改写器把它术语化(不带闸门职责,保证改写质量)。

约束:改写只做措辞术语化,不臆造原问题没有的数值/结论 —— 改写器不知道答案,
也不该替用户假设答案,否则会把检索带偏到"编出来的那个数"上。
"""

from __future__ import annotations

from ruixue_agent.models import create_model

_GATE_SYS = (
    "判断下面这条农业地膜领域的检索查询是不是【口语化/模糊/缺术语】的。\n"
    "- 口语:像农户随口问的,用大白话、缺领域术语(如「浇点水管用不」「膜暖和不」"
    "「地里残膜多了咋办」)。\n"
    "- 规范:已经用了领域术语、表述精确(如「玉米吐丝期普通黑膜膜下温度比降解膜高多少度」"
    "「加厚地膜当季捡净率」)。\n"
    "只输出一个词:口语 或 规范。"
)

_REWRITE_SYS = (
    "你是农业地膜领域的检索查询改写器。把农户的口语问题改写成术语规范、"
    "适合向量检索的查询。要求:\n"
    "1. 用领域术语替换口语(如「浇点水」→「灌溉/坐水种」,「膜烂了」→「地膜降解/开裂期」,"
    "「暖和」→「土壤温度/保温」,「捡膜」→「残膜回收/捡净率」)。\n"
    "2. 保留原问题的【全部信息点】(作物、时期、对比对象、要问的指标)。\n"
    "3. 只做措辞术语化,【不要】臆造原问题没有的具体数值、结论或答案。\n"
    "4. 输出改写后的查询本身,不加解释、不加标点包裹。\n"
    "示例:\n"
    "  口语:膜盖上多久能开始烂?\n"
    "  改写:生物降解地膜 降解进程 诱导期 开裂期 天数\n"
    "  口语:这膜保温到底比普通膜好还是差?\n"
    "  改写:地膜 保温效果 土壤温度 与普通地膜对比"
)


class QueryRewriter:
    """两段式口语查询改写。model_name 复用 config 中已配的模型。"""

    def __init__(self, model_name: str = "deepseek-v4-flash") -> None:
        self.llm = create_model(model_name)

    def is_colloquial(self, query: str) -> bool:
        """闸门:判查询是否口语。异常/不确定回退 False(从严,不乱改)。"""
        try:
            out = self.llm.invoke(
                [("system", _GATE_SYS), ("user", query)]
            ).content.strip()
            return "口语" in out and "规范" not in out
        except Exception:
            return False

    def _rewrite_raw(self, query: str) -> str:
        resp = self.llm.invoke(
            [("system", _REWRITE_SYS), ("user", f"口语:{query}\n改写:")]
        ).content.strip()
        line = resp.splitlines()[0].strip().strip("\"'“”「」")
        return line or query

    def rewrite(self, query: str) -> str:
        """闸门判为口语才改写,否则原样返回。任何异常回退原查询(改写是增强,不该成故障点)。"""
        try:
            if not self.is_colloquial(query):
                return query
            return self._rewrite_raw(query)
        except Exception:
            return query
