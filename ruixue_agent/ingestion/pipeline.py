"""管道编排:把各阶段按正确顺序串起来。

  每篇独立:  parse → clean
  聚齐之后:  dedup(跨文档,要看全集) → metadata → quality(门禁) → chunk

为什么 dedup 在 metadata 前:要丢掉的重复论文,没必要先给它抽元数据。
为什么 quality 在 metadata 后:门禁要检查"标题/年份抽到没"。
"""

from __future__ import annotations

from ruixue_agent.ingestion.stages.chunk import chunk_document
from ruixue_agent.ingestion.stages.clean import clean_document
from ruixue_agent.ingestion.stages.dedup import dedup_documents
from ruixue_agent.ingestion.stages.metadata import extract_metadata
from ruixue_agent.ingestion.parsers.mineru_parser import parse_content_list
from ruixue_agent.ingestion.stages.quality import assess_quality
from ruixue_agent.ingestion.schema import Chunk, Document


def parse_and_clean(content_list: list[dict], meta: dict) -> Document:
    """每篇独立的前两步:MinerU 输出 → Document → 文本规范化。"""
    return clean_document(parse_content_list(content_list, meta))


def build_chunks(docs: list[Document]) -> tuple[list[Chunk], dict]:
    """聚齐后:去重 → 抽元数据 → 质量门禁 → 切块。返回 (chunks, 统计对账)。"""
    kept, dup_dropped = dedup_documents(docs)  # 跨文档:近似重复丢掉

    passed: list[Document] = []
    failed: list[Document] = []
    for doc in kept:
        extract_metadata(doc)  # 抽 title/year/keywords…
        if assess_quality(doc)["passed"]:  # 门禁:不合格不入库
            passed.append(doc)
        else:
            failed.append(doc)

    chunks: list[Chunk] = []
    for doc in passed:
        chunks += chunk_document(doc)  # 父子分块

    stats = {  # 对账:每篇都有下落
        "输入": len(docs),
        "重复丢弃": len(dup_dropped),
        "质量不合格": len(failed),
        "入库": len(passed),
        "产出chunk": len(chunks),
    }
    return chunks, stats
