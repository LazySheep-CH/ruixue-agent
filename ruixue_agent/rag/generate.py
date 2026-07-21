"""生成(Generation)—— RAG 的最后一个字母 G。

检索负责【找到】材料,生成负责【读完材料、用一句话回答】。

═══ 唯一的技术要点:grounding(接地)═══

这一步不是"把检索结果喂给 LLM"那么简单。核心是【逼着 LLM 只用检索到的材料
回答,不许用自己脑子里的知识】。否则:
    问"地膜厚度国标多少" → LLM 用训练时记的旧知识答"0.008mm"
    → 但我们的标准库里写的是 0.010mm → 它【自信地答错了】

这正是 RAG 存在的意义:让答案【可溯源、可核实、跟着你的知识库更新】,
而不是赌 LLM 的记忆。做法有三条,都写进 prompt:
    ① 只用【提供的材料】回答,材料里没有就说"资料里没有"
    ② 每句话标引用 [1][2],让用户能核实
    ③ 不许编造材料里没有的数字和结论

地膜标准这种场景尤其致命 —— 答错一个厚度、一个降解率,是要赔钱的。
没有 grounding 的 RAG 不能上生产。
"""

from __future__ import annotations

from dataclasses import dataclass

from ruixue_agent.models import create_model
from ruixue_agent.rag.retriever import Hit

_SYSTEM = """你是地膜领域的专业助手,服务于研发、技术服务和质检人员。

回答规则(必须严格遵守):
1. 只依据【下面提供的资料】回答,不要用你自己的知识补充或猜测。
2. 资料里如果没有能回答问题的信息,直接说"提供的资料中没有相关信息",不要编造。
3. 每个关键结论后面标注引用来源,格式 [1]、[2],对应资料编号。
4. 涉及数字、指标、标准号时,必须与资料完全一致,一个字都不能改。
5. 回答简洁、直接,像行业内行说话,不要客套。"""

_USER = """问题:{question}

资料:
{context}

请依据以上资料回答。"""


@dataclass
class Answer:
    text: str  # LLM 生成的答案
    hits: list[Hit]  # 用到的检索结果(带出处),供用户核实
    used_context: bool  # 有没有检索到材料(没有时 LLM 该说"资料里没有")


def _format_context(hits: list[Hit]) -> str:
    """把检索结果编号,拼成给 LLM 的资料块。

    编号是为了让 LLM 能引用 [1][2] —— 引用的前提是每段有个稳定的号。
    带上标题/章节,让 LLM(和用户)知道每段的出处。
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        path = " > ".join(h.section_path[:2]) if h.section_path else ""
        head = f"[{i}] 出处:{h.document_id}" + (f" · {path}" if path else "")
        blocks.append(f"{head}\n{h.text.strip()}")
    return "\n\n".join(blocks)


class Generator:
    """检索 → 拼 prompt → LLM 生成带引用的答案。

    retriever 从外部传入(依赖注入)—— 和整个项目一样的模式:
    上层不关心检索是纯向量还是混合还是带 rerank,只要它有 .search()。
    """

    def __init__(self, retriever, model_name: str = "deepseek-v4-flash") -> None:
        self.retriever = retriever
        self.llm = create_model(model_name)

    def answer(
        self,
        question: str,
        k: int = 5,
        year_min: int | None = None,
        source: str | None = None,
    ) -> Answer:
        hits = self.retriever.search(question, k=k, year_min=year_min, source=source)

        # 没检索到任何材料 —— 不调 LLM 了,直接如实说。
        # (省一次 API,也避免 LLM 在空上下文里凭记忆瞎答 —— 那是最危险的情况)
        if not hits:
            return Answer(
                text="提供的资料中没有相关信息。",
                hits=[],
                used_context=False,
            )

        prompt = _USER.format(question=question, context=_format_context(hits))
        resp = self.llm.invoke(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        return Answer(text=resp.content.strip(), hits=hits, used_context=True)
