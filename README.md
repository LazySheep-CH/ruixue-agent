# 瑞雪智能体（ruixue-agent）（还没写完，还在持续更新）

面向地膜行业、从零正规搭建的中文**智能体(agent)**。
核心能力是一套可溯源的知识库 —— 把 1500+ 篇文献和国标规程做成可检索的知识,包装成工具交由
agent 按需调用,生成带**引用出处**的答案(另配用量估算工具);支持多轮对话、会话持久化(PostgreSQL)、
流式输出,并带可观测与错误降级中间件。

服务对象:研发工程师(材料/配方/机理)、技术服务(田间问题)、质检采购(合规/指标)。

## 能力概览(实测数据)

| 环节 | 规模 / 指标 |
| --- | --- |
| 语料 | 1,578 篇 → 262,782 检索块(父 39,396 + 子 223,386) |
| 向量 | 223,386 条(BAAI/bge-small-zh-v1.5, 512 维) |
| 检索质量 | Recall@1 **0.769** / Recall@5 0.952 / Recall@10 **0.979** / MRR **0.849**(向量+BM25+重排) |
| 逐层增益 | 纯向量 R@1 0.593 → +混合 0.631 → +重排 **0.769** |
| **Agent 质量** | 33 题冻结评测集 **32/33**;工具选择 recall 1.000 / precision 0.878;噪声地板 0% |
| 测试 | 328 个(RAG 全覆盖 + agent 中间件层 + 运行时/限流/记忆/评测) |

检索指标由 338 题评测集测出(其中 290 题有答案计入 Recall,48 题为不可答题用于验证拒答;
合成 + 人工质检 + TREC pooling 多标注),方法见 `docs/评测方法.md`。
指标口径以 `uv run python scripts/run_eval.py --ab` 的输出为准 —— 本表若与之不符,以脚本为准。
Agent 指标是**端到端**的(工具选择 / 拒答 / 追问 / 注入 / 成本),方法与首次基线见
[docs/Agent评测方法.md](docs/Agent评测方法.md)。两者分工:检索层只说明"料是对的",
不说明 agent 会不会去查、查完会不会用。

## 架构

本项目是一个**智能体(agent)**:以 LangChain `create_agent` 的 ReAct 循环为核心,
把知识检索、用量计算等能力做成**工具**交它自主调用。代码分两个包,对标参考架构的
**harness / app 分离**:

- **`ruixue_agent/`(harness)** — agent 框架,**不认识 HTTP**,可被 Web / 飞书 / CLI 复用
- **`ruixue_app/`(app)** — 服务层(FastAPI),依赖 harness、**反向不行**(ports-and-adapters)

```
ruixue_agent/                # HARNESS:agent 框架(不认识 HTTP)
├── agents/                  agent 装配层
│   ├── builder.py             create_ruixue_agent(模型+工具+提示+记忆+中间件)
│   ├── prompt.py             系统提示(独立资产,便于迭代)
│   └── middlewares.py        计时日志(可观测)+ 工具错误降级(扛得住)
├── tools/                   agent 的工具(calc 用量估算 · rag 知识检索 · web 兜底)
├── models.py                模型工厂(按配置装配 LLM)
├── checkpointer.py          会话记忆(PostgreSQL 持久化,按 thread_id 存取)
├── config.py                配置加载(YAML + $ENV 解析)
│
│   ——— 以下 RAG 子系统,按【调用频率】分三层(频率决定设计目标)———
├── ingestion/              离线入库(一份文档一辈子跑一次 → 要准,慢无妨)
│   ├── schema.py             统一中间结构 IR(Document / Element / Chunk)
│   ├── parsers/              MinerU 输出 → IR
│   ├── stages/               clean · dedup · metadata · quality · chunk
│   └── utils/                latex · table · frontmatter
├── persistence/            存储(数据的 source of truth → 别丢)
│   ├── models.py             SQLAlchemy 表结构(documents / chunks / term_df)
│   ├── repository.py         对象 ⇄ 数据库行,幂等 upsert
│   └── migrations/           Alembic 版本化迁移(可回滚可追溯)
└── rag/                    在线检索(每次提问都跑 → 要快)
    ├── embedding.py          文本 → 向量
    ├── milvus_store.py       向量索引(只存 id+向量+过滤字段,不存文本)
    ├── bm25.py               词法检索(PG 全文,靠罕见词精确定位)
    ├── text_segment.py       中文分词(jieba + 领域词典 + 标准号保护)
    ├── fusion.py             RRF 融合多路排名
    ├── rerank.py             cross-encoder 精排
    ├── retriever.py          编排:Small-to-Big + 混合 + 重排
    ├── metrics.py            Recall@k / MRR
    └── generate.py           检索结果 → 带引用的答案(grounding)

ruixue_app/                  # APP:服务层(FastAPI),依赖 harness
├── main.py                  POST /chat(多轮)· /chat/stream(SSE)· /health · CORS
├── auth.py                  API Key 认证 + 按用户命名空间隔离
└── observability.py         request_id 追踪 + 结构化日志

frontend/                    # WEB:聊天界面(Next.js + TS + Tailwind + zustand)
├── src/core/                业务核心(类型 · SSE 客户端 · 状态),不认识 React
├── src/components/chat/     组件(会话面板 · 顶栏 · 消息列表 · 输入框)
└── src/app/                 App Router 页面
```

设计要点:

- **harness / app 分离** — agent 核心不 import fastapi,服务层才依赖核心。
  将来接飞书/Slack 只需在 app 侧加适配器,agent 逻辑一行不改(ports-and-adapters)。
- **中间件即横切关注点** — 计时日志(可观测性)、工具错误优雅降级(扛得住)套在工具调用外,
  不侵入业务;**顺序即设计**(错误处理在内层,失败调用也能记耗时)。
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

# 6. 起服务(FastAPI:多轮对话 + 流式 SSE)
uv run uvicorn ruixue_app.main:app --reload    # API 文档 http://127.0.0.1:8000/docs

# 7. 起前端(聊天界面,另开一个终端)
cd frontend && npm install && npm run dev      # 浏览器开 http://127.0.0.1:3000
```

完整的入库 / 灌向量 / 评测流程见 [docs/操作手册.md](docs/操作手册.md)。

## 文档

- [docs/评测方法.md](docs/评测方法.md) — 检索评测:评测集怎么造、指标怎么算、逐层增益
- [docs/Agent评测方法.md](docs/Agent评测方法.md) — Agent 评测:七类用例、噪声地板、配对显著性检验
- [docs/运维手册.md](docs/运维手册.md) — 部署、备份恢复、并发容量、排查手册
- [docs/项目问答卡.md](docs/项目问答卡.md) — **一问一答,用来背**(全项目 80+ 题)
- [docs/项目讲解.md](docs/项目讲解.md) — 按「评审怎么问」重组的项目全景:9 条追问链 + 真实数字 + 边界
- [docs/操作手册.md](docs/操作手册.md) — 从 PDF 到可检索的完整脚本流程
- [DECISIONS.md](DECISIONS.md) — 关键技术决策与取舍(为什么这么选)
- [docs/数据来源.md](docs/数据来源.md) — 领域参数的权威出处

## 技术栈

Python 3.11 · PostgreSQL 17 · Milvus 2.6 · SQLAlchemy + Alembic ·
sentence-transformers(bge)· jieba · uv
