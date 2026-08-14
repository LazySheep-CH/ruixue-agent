"""瑞雪地膜知识库 —— 命令行问答入口。

用法:
    uv run python main.py                    # 交互式,连续问
    uv run python main.py "地膜厚度国标多少"   # 单次问答

前置:基础设施已起、数据已入库(见 docs/操作手册.md)。
"""

import sys

from sqlalchemy.orm import Session

from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.repository import PgRepository
from ruixue_agent.rag.bm25 import Bm25Search
from ruixue_agent.rag.generate import Generator
from ruixue_agent.rag.milvus_store import MilvusVectorStore
from ruixue_agent.rag.rerank import Reranker
from ruixue_agent.rag.retriever import Retriever

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK,不改打不出中文


def _answer(gen: Generator, question: str) -> None:
    ans = gen.answer(question, k=4)
    print(f"\n{ans.text}\n")
    if ans.hits:
        print("出处:")
        for i, h in enumerate(ans.hits, 1):
            path = " > ".join(h.section_path[:2]) if h.section_path else ""
            print(f"  [{i}] {h.document_id} · {path}")


def main() -> None:
    store = MilvusVectorStore()
    reranker = Reranker()  # 首次加载模型较慢

    with Session(get_engine()) as sess:
        retriever = Retriever(
            store,
            PgRepository(sess),
            bm25=Bm25Search(sess),  # 混合检索
            reranker=reranker,  # 精排
        )
        gen = Generator(retriever)

        # 单次模式:命令行带了问题
        if len(sys.argv) > 1:
            _answer(gen, " ".join(sys.argv[1:]))
            return

        # 交互模式
        print("瑞雪地膜知识库(输入问题,回车提问;空行退出)")
        while True:
            try:
                q = input("\n问> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            _answer(gen, q)


if __name__ == "__main__":
    main()
