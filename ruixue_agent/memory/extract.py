"""从一次对话里抽出【值得长期记住的事实】。

## 为什么要抽,不直接存原文

存原文看着省事,代价在后面:
  · 检索噪声大 —— "你好""谢谢"也会被召回,挤占上下文
  · 没法删 —— 用户说"别记我的地块了",你无从下手
  · 浪费 token —— 召回 5 条闲聊,顶不上 1 条事实

抽成一句话的事实,上面三个问题一起解决。代价是多一次 LLM 调用,
但用便宜模型、且在【运行结束后异步】做,不占用户的等待时间。

## 什么该记、什么不该记(这条边界很重要)

该记的是**跨会话仍然成立**的事实:地块在哪、多大、种什么、用户的偏好。
不该记的是**一次性的**:这次问的问题、这次的计算结果、闲聊。

把一次性的东西记成长期记忆,后果是它会在无关的对话里被召回,
让模型基于过期信息瞎推理 —— **那是主动制造幻觉**。
"""

from __future__ import annotations

import json
import logging

from ruixue_agent.models import create_model

logger = logging.getLogger("ruixue.memory")

# 抽取用便宜模型:这是内部动作,用户看不到,没必要用贵的。
_EXTRACT_MODEL = "deepseek-v4-flash"

_SYSTEM = """你是记忆抽取器。从用户与地膜助手的一轮对话里,抽出【值得长期记住】的事实。

只抽这四类,每条一句话,用第三人称陈述:
- plot:地块信息(位置、面积、灌溉方式等)
- crop:作物与种植计划(作物、生育期、播期)
- preference:用户的偏好或约束(更看重保墒/成本/强度、当地风大、要求可降解等)
- other:其他确实值得跨会话记住的

【绝对不要抽】:
- 这次问的问题本身、这次算出的结果(它们是一次性的)
- 寒暄、感谢、对系统能力的询问
- 助手说的话里的领域知识(那是知识库的事,不是这个用户的事实)
- 任何你不确定的推测——宁可少抽,不要抽错

把一次性的信息记成长期记忆,它会在无关的对话里被召回,让模型基于过期信息
推理,等于主动制造幻觉。所以【没有值得记的就返回空列表】,这是常态,不是失败。

只输出 JSON,格式:
{"facts": [{"kind": "plot", "text": "用户在新疆尉犁县有 50 亩棉花地"}]}"""


def extract_facts(question: str, answer: str) -> list[tuple[str, str]]:
    """从一问一答里抽事实。返回 (kind, text) 列表,抽不到就返回空。

    任何异常都吞掉返回空 —— 记忆是锦上添花,抽取失败绝不能影响主流程。
    """
    if not question.strip():
        return []
    try:
        resp = create_model(_EXTRACT_MODEL, max_retries=1).invoke(
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    # 答案截断:抽事实靠的是用户说了什么,助手的长篇回答里
                    # 绝大部分是领域知识(属于知识库),留个开头做上下文即可。
                    "content": f"用户说:{question}\n\n助手答(节选):{answer[:600]}",
                },
            ]
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = json.loads(_strip_fence(text))
        facts = [
            (f.get("kind", "other"), f["text"].strip())
            for f in data.get("facts", [])
            if isinstance(f, dict) and f.get("text", "").strip()
        ]
        # 上限:一轮对话抽出十几条几乎肯定是抽错了(把知识当事实了)
        return facts[:5]
    except Exception:
        logger.warning("记忆抽取失败,本次不记", exc_info=True)
        return []


def _strip_fence(s: str) -> str:
    """模型爱把 JSON 包在 ```json 里。不剥掉就解析失败 —— 而失败是静默的
    (被上面的 except 吞掉),表现为"记忆功能好像没生效",极难查。"""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("```", 1)[0]
    return s.strip()
