"""知识检索工具:把 RAG 检索+生成包成 agent 可调用的工具(Agentic RAG)。

agent 遇到"专业知识问题"时自主调用本工具查知识库,遇到"用量计算"时调 calc 工具。
检索不再是固定管道,而是 agent 可编排的一个能力。

组装照 main.py:昂贵组件(Milvus 连接、rerank 模型)懒加载单例复用;
便宜且随会话的组件(DB session、repo、bm25)每次调用新建。
"""

import logging
from functools import cache as cache_func

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.repository import PgRepository
from ruixue_agent.rag import cache
from ruixue_agent.rag.bm25 import Bm25Search
from ruixue_agent.rag.generate import Generator, generation_fingerprint
from ruixue_agent.rag.milvus_store import MilvusVectorStore
from ruixue_agent.rag.rerank import Reranker
from ruixue_agent.rag.retriever import Retriever

logger = logging.getLogger("ruixue.agent")

# 昂贵组件:懒加载单例,整个进程只建一次、复用
_store: MilvusVectorStore | None = None
_reranker: Reranker | None = None


def _get_store() -> MilvusVectorStore:
    global _store
    if _store is None:
        _store = MilvusVectorStore()
    return _store


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()  # 首次加载模型较慢
    return _reranker


@cache_func
def _kb_version() -> str:
    """知识库版本指纹:用向量库条目数近似。灌了新数据 → 指纹变 → 旧缓存自然失效。

    进程内缓存(灌数据是离线动作,不会在服务运行期间变);重启即刷新。
    取不到时返回 "unknown",不影响主流程。
    """
    try:
        return str(_get_store().client.get_collection_stats("chunks").get("row_count", "?"))
    except Exception:
        return "unknown"


@tool
def search_knowledge(question: str) -> str:
    """查询地膜领域知识库,回答专业知识问题(答案带出处引用)。

    知识库含 1500+ 篇地膜文献与国家/行业/地方标准,可回答:
    - 材料性能与配方(PBAT/PLA/PE、力学/阻隔/热性能等)
    - 降解机制与影响因素(土壤、温湿度、微生物等)
    - 各作物覆膜栽培规程(玉米/小麦/水稻/棉花等)
    - 标准指标与合规要求(厚度、拉伸强度、降解率等国标)
    - 残留污染与回收利用

    适用:任何需要领域知识、文献依据、标准条文的问题。
    不适用:地膜【用量/成本】的数值计算——那用 estimate_film_usage 工具。
    """
    # 缓存:同一个问题不重复走"检索+重排+生成"(这条路径含一次 LLM 调用,最贵)。
    # 用【精确匹配】而非语义匹配 —— 实测语义匹配在本领域会把"降解太快/太慢"
    # 判为同一问题(相似度 0.97),给出相反建议。详见 rag/cache.py 的说明。
    # 缓存版本 = 知识库版本 + 【生成逻辑版本】。
    # 只用知识库版本是不够的:改了系统规则或资料拼装格式,同一问题就该有不同答案,
    # 而旧缓存会照样命中 —— 改动对老用户静默失效。详见 generate.generation_fingerprint。
    kb_ver = f"{_kb_version()}|{generation_fingerprint()}"
    cached = cache.get(question, kb_ver)
    if cached is not None:
        logger.info("知识问答命中缓存")
        return cached

    # 每次调用开一个新会话:repo/bm25 绑在会话上,会话结束自动释放
    with Session(get_engine()) as sess:
        retriever = Retriever(
            _get_store(),
            PgRepository(sess),
            bm25=Bm25Search(sess),  # 混合检索(向量 + 词法)
            reranker=_get_reranker(),  # cross-encoder 精排
        )
        ans = Generator(retriever).answer(question, k=4)

    # 没检索到资料时,Generator 已返回兜底话术,直接回传
    if not ans.used_context:
        return ans.text  # 兜底话术不缓存:知识库补充后同一问题应能重新检索

    # 附上出处,便于用户/agent 核实答案来源
    sources = []
    for i, h in enumerate(ans.hits, start=1):
        path = " > ".join(h.section_path[:2]) if h.section_path else ""
        sources.append(f"[{i}] {h.document_id}" + (f" · {path}" if path else ""))
    result = ans.text + "\n\n出处:\n" + "\n".join(sources)
    cache.put(question, kb_ver, result)
    return result
