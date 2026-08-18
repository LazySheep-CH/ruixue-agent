# 瑞雪智能体（ruixue-agent）

[![CI](https://github.com/LazySheep-CH/ruixue-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LazySheep-CH/ruixue-agent/actions/workflows/ci.yml)

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
| **Agent 质量** | 33 题冻结评测集 **32/33(97.0%)**;工具选择 recall 1.000 / precision 0.921;噪声地板 0% |
| 能力面 | 14 个领域工具 · 4 位专家子 agent · 3 个自训预测模型 · MCP 双向 |
| 记忆收益 | 12 题跨会话对照,三轮实测任务完成率 44.4% → 77.8%(+33.3pp) |
| 安全 | 10 类攻击面专项评测 10/10 拦截(注入/越权/路径逃逸/诱导编造等) |
| 测试 | 427 个(RAG / 中间件 / 运行时 / 记忆 / 分析 / MCP / 指标告警) |

检索指标由 338 题评测集测出(其中 290 题有答案计入 Recall,48 题为不可答题用于验证拒答;
合成 + 人工质检 + TREC pooling 多标注),方法见 `docs/评测方法.md`。
指标口径以 `cd backend && uv run python scripts/run_eval.py --ab` 的输出为准 —— 本表若与之不符,以脚本为准。
Agent 指标是**端到端**的(工具选择 / 拒答 / 追问 / 注入 / 成本),方法与首次基线见
[docs/Agent评测方法.md](docs/Agent评测方法.md)。两者分工:检索层只说明"料是对的",
不说明 agent 会不会去查、查完会不会用。

## 架构

本项目是一个**智能体(agent)**:以 LangChain `create_agent` 的 ReAct 循环为核心,
把知识检索、用量计算等能力做成**工具**交它自主调用。代码分两个包,**harness / app 分离**:

- **`ruixue_agent/`(harness)** — agent 框架,**不认识 HTTP**,可被 Web / 飞书 / CLI 复用
- **`ruixue_app/`(app)** — 服务层(FastAPI),依赖 harness、**反向不行**(ports-and-adapters)

```
backend/                       # Python 侧(harness 与 app 分离)
├── ruixue_agent/              HARNESS:agent 框架,不认识 HTTP,可被 Web/飞书/CLI 复用
│   ├── agents/                装配:builder(模型+工具+提示+checkpointer+8 层中间件)
│   ├── tools/                 14 个领域工具:用量 · 土壤/气候 · 天气预报 · 三项性能预测
│   │                          · 配方筛选 · 知识检索 · 上传数据分析
│   ├── subagents.py           4 位专家子 agent(文献检索/配方优化/故障诊断/数据分析)
│   ├── memory/                长期记忆:运行后抽取事实,PG 存权威 + 向量召回,按用户隔离
│   ├── analysis/              上传数据:列名归一 → 契约校验 → 入库 → 统计与模型对比
│   ├── userkb/                用户自有知识库:上传 PDF/TXT/MD → 切块向量化 → 隔离检索,
│   │                          问答自动引用并标注「您上传的资料」
│   ├── predictors/            三个自训性能预测模型(降解率/透过率/拉伸强度)+ 环境取数
│   ├── mcp/                   MCP 客户端:按配置接入外部工具,连不上自动降级
│   ├── skills/                技能(SOP)按需注入
│   ├── eval/                  评测框架:轨迹抽取 · 判分 · 噪声地板 · 记忆收益对照
│   ├── ingestion/             离线入库(MinerU → 清洗/去重/分块 → 双写)
│   ├── persistence/           SQLAlchemy 模型 + Alembic 迁移(数据的 source of truth)
│   ├── rag/                   在线检索:向量+BM25 → RRF → 重排 → 带引用生成
│   └── models.py · checkpointer.py · config.py · guardrails/
├── ruixue_app/                APP:FastAPI 服务层,依赖 harness,反向不行
│   ├── main.py                /chat(SSE)· /datasets · /kb/docs · /metrics · 报告导出 · 健康探针
│   ├── runs.py                异步 Run:后台线程池 + Redis Stream,断线续跑/重连补发
│   ├── auth.py · quota.py     JWT / API Key · 每日配额(fail-open)
│   ├── mcp_server.py          MCP 服务端:把自有能力暴露给外部 agent(默认关闭)
│   └── metrics.py · report.py · observability.py · security/
├── tests/ · scripts/ · data/ · config/
frontend/                      # 聊天界面(Next.js + TS),登录 · 流式 · 上传 · 报告下载
docker/                        # compose 编排 + nginx(TLS 模板)
scripts/ops/                   # 部署 / 备份(含异地同步)/ 告警巡检
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
# 1. 依赖(Python 侧都在 backend/ 下)
cd backend && uv sync

# 2. 配置(复制模板,填入自己的模型 key —— 通过环境变量,不写死)
cp config/config.example.yaml config/config.yaml   # 在 backend/ 目录下
export DEEPSEEK_API_KEY=你的key

# 3. 起基础设施(PostgreSQL + Milvus)
cd ../docker && cp .env.example .env && docker compose up -d && cd ../backend

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
- [docs/操作手册.md](docs/操作手册.md) — 从 PDF 到可检索的完整脚本流程
- [DECISIONS.md](DECISIONS.md) — 关键技术决策与取舍(为什么这么选)
- [docs/数据来源.md](docs/数据来源.md) — 领域参数的权威出处

## 技术栈

Python 3.11 · PostgreSQL 17 · Milvus 2.6 · SQLAlchemy + Alembic ·
sentence-transformers(bge)· jieba · uv

## 版权

本项目归**江苏省农业科学院农业设施与装备研究所**,保留所有权利;
仅作展示交流,未经书面许可不得复制、分发或商用。详见 [COPYRIGHT](COPYRIGHT)。
