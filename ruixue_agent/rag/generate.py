"""生成层:把检索结果交给大模型,产出带引用的答案。

关键约束是 grounding —— 强制模型只依据检索到的材料回答:
材料中没有的信息不得补充,数字与原文严格一致,关键结论标注引用编号。
不做此约束时,模型会用训练语料里的旧知识作答,答案无法溯源、
不随知识库更新,且在数值类问题上错得毫无征兆。
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
    text: str  # 生成的答案
    hits: list[Hit]  # 引用的检索结果,含出处,供核实
    used_context: bool  # 是否检索到了材料


def _format_context(hits: list[Hit]) -> str:
    """把检索结果编号拼装为资料块。编号供模型标注引用,出处随行附带。"""
    blocks = []
    for i, h in enumerate(hits, start=1):
        path = " > ".join(h.section_path[:2]) if h.section_path else ""
        head = f"[{i}] 出处:{h.document_id}" + (f" · {path}" if path else "")
        blocks.append(f"{head}\n{h.text.strip()}")
    return "\n\n".join(blocks)


class Generator:
    """检索 → 组装 prompt → 生成带引用的答案。retriever 由外部注入。"""

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

        # 未检索到材料时直接返回,不调用模型 —— 空上下文下模型只能凭记忆作答,
        # 恰是最不可控的路径,同时省一次调用
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
