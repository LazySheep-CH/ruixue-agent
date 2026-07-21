# 瑞雪地膜知识库（ruixue-agent）

面向地膜行业的中文检索问答系统。把 1500+ 篇文献和国标规程做成可检索、可溯源的知识库：
用户用中文提问，系统检索相关段落、交大模型生成带**引用出处**的答案。

服务对象:研发工程师(材料/配方/机理)、技术服务(田间问题)、质检采购(合规/指标)。

## 能力概览(实测数据)

| 环节 | 规模 / 指标 |
| --- | --- |
| 语料 | 1,578 篇 → 262,782 检索块(父 39,396 + 子 223,386) |
| 向量 | 223,386 条(BAAI/bge-small-zh-v1.5, 512 维) |
| 检索质量 | Recall@1 **0.807** / Recall@10 **0.953** / MRR **0.866**(向量+BM25+重排) |
| 逐层增益 | 纯向量 R@1 0.587 → +混合 0.647 → +重排 0.807 |
| 测试 | 123 个,`ruixue_agent/rag` 全覆盖 |

指标由 150 题评测集测出(合成 + 人工质检 + TREC pooling),方法见 `docs/评测方法.md`。

## 架构

三层,划分依据是**调用频率** —— 频率决定每层的设计目标:

```
ruixue_agent/
├── ingestion/          离线入库(一份文档一辈子跑一次 → 要准,慢无妨)
│   ├── schema.py         统一中间结构 IR(Document / Element / Chunk)
│   ├── parsers/          MinerU 输出 → IR
│   ├── stages/           clean · dedup · metadata · quality · chunk
│   └── utils/            latex · table · frontmatter
├── persistence/        存储(数据的 source of truth → 别丢)
│   ├── models.py         SQLAlchemy 表结构(documents / chunks / term_df)
│   ├── repository.py     对象 ⇄ 数据库行,幂等 upsert
│   └── migrations/       Alembic 版本化迁移(可回滚可追溯)
└── rag/                在线检索(每次提问都跑 → 要快)
    ├── embedding.py      文本 → 向量
    ├── milvus_store.py   向量索引(只存 id+向量+过滤字段,不存文本)
    ├── bm25.py           词法检索(PG 全文,靠罕见词精确定位)
    ├── text_segment.py   中文分词(jieba + 领域词典 + 标准号保护)
    ├── fusion.py         RRF 融合多路排名
    ├── rerank.py         cross-encoder 精排
    ├── retriever.py      编排:Small-to-Big + 混合 + 重排
    ├── metrics.py        Recall@k / MRR
    └── generate.py       检索结果 → 带引用的答案(grounding)
```

设计要点:

- **数据是资产,索引是派生物** — PostgreSQL 存全部数据(丢了要重跑 5 小时解析),
  Milvus 只存向量(丢了几分钟重建)。所以文本不进 Milvus。
- **Small-to-Big** — 子块(短)进向量库负责被搜到,命中后返回父块(长)给大模型。
  检索要短才准,生成要长才全。
- **混合检索** — 向量(语义)+ BM25(罕见词精确匹配),RRF 融合,cross-encoder 重排。
- **grounding** — 只依据检索到的材料回答、标引用、数字与原文一致,不许编造。

## 数据流

```
PDF ─MinerU─▶ IR ─ clean·dedup·metadata·quality·chunk ─▶ PostgreSQL(全部数据)
                                                            │
                                              embed ────────┴─▶ Milvus(向量索引)

提问 ─▶ 向量+BM25 检索 ─▶ RRF 融合 ─▶ 重排 ─▶ 取父块正文 ─▶ 大模型生成带引用答案
```

## 快速开始

```bash
# 1. 依赖(项目以 editable 包安装,含 GPU 版 torch)
uv sync

# 2. 配置(复制模板,填入自己的模型 key —— 通过环境变量,不写死)
cp config/config.example.yaml config/config.yaml
export DEEPSEEK_API_KEY=你的key

# 3. 起基础设施(PostgreSQL + Milvus)
cd docker && cp .env.example .env && docker compose up -d && cd ..

# 4. 建库(数据库结构由 Alembic 管理)
uv run alembic -c ruixue_agent/persistence/migrations/alembic.ini upgrade head

# 5. 测试
uv run pytest
```

完整的入库 / 灌向量 / 评测流程见 [docs/操作手册.md](docs/操作手册.md)。

## 文档

- [docs/评测方法.md](docs/评测方法.md) — 评测集怎么造、指标怎么算、逐层增益
- [docs/操作手册.md](docs/操作手册.md) — 从 PDF 到可检索的完整脚本流程
- [DECISIONS.md](DECISIONS.md) — 关键技术决策与取舍(为什么这么选)
- [docs/数据来源.md](docs/数据来源.md) — 领域参数的权威出处

## 技术栈

Python 3.11 · PostgreSQL 17 · Milvus 2.6 · SQLAlchemy + Alembic ·
sentence-transformers(bge)· jieba · uv
