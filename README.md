# 瑞雪智能体（ruixue-agent）

[![CI](https://github.com/LazySheep-CH/ruixue-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/LazySheep-CH/ruixue-agent/actions/workflows/ci.yml)

地膜领域的中文智能体：用自然语言提问（选膜、性能、
田间问题、标准条文），agent 自己决定查土壤气候、调预测模型还是检索文献，答案带引用出处。
后端 FastAPI + LangChain，前端 Next.js。

语料整理了 1,500 多篇文献和标准，检索 Recall@1 做到 0.769（338 题评测集），
另外有一套 agent 端到端评测和五百多个自动化测试。
指标明细和评测方法在 [docs/评测方法.md](docs/评测方法.md) 和 [docs/Agent评测方法.md](docs/Agent评测方法.md)。

## 架构

代码分两个包：ruixue_agent 是 agent 框架本身，不依赖 HTTP；ruixue_app 是 FastAPI 服务层，
依赖前者。这样以后要接飞书或 CLI，只用在 app 侧加适配，agent 逻辑不动。

```
backend/
├── ruixue_agent/     agent 框架：工具、检索(rag)、记忆、预测、评测
├── ruixue_app/       FastAPI 服务层：接口、认证、运行记录
├── tests/ scripts/ config/
frontend/             Next.js 聊天界面
docker/               compose 编排 + nginx
```

完整目录和一次请求的路径见 [docs/架构说明.md](docs/架构说明.md)。

几个设计上的决定（详细取舍见 [DECISIONS.md](DECISIONS.md)）：

- 文本只存 PostgreSQL，Milvus 只放向量。PG 丢了要重跑 5 小时解析，Milvus 丢了几分钟重建，
  所以备份只备 PG。
- 检索用子块（短，容易命中），命中后返回父块（长，上下文全）给大模型。
- 检索到的材料强制带出处和年份，答案里的数字要和原文一致，检不到就说没有。
- agent 不绑 HTTP 请求，跑在后台线程池，用户断线了照样跑完落库。

## 数据流

```
PDF -MinerU-> 解析结果 -清洗/去重/分块-> PostgreSQL(全部数据)
                                          |
                              embed ------+--> Milvus(向量索引)

提问 -> 向量+BM25 检索 -> RRF 融合 -> 重排 -> 取父块正文 -> 生成带引用答案
```

## 快速开始

```bash
# 1. 依赖（Python 都在 backend/ 下）
cd backend && uv sync

# 2. 配置，模型 key 走环境变量
cp config/config.example.yaml config/config.yaml
export DEEPSEEK_API_KEY=你的key

# 3. 起基础设施（PostgreSQL + Milvus）
cd ../docker && cp .env.example .env && docker compose up -d && cd ../backend

# 4. 建库
uv run alembic -c ruixue_agent/persistence/migrations/alembic.ini upgrade head

# 5. 测试
uv run pytest

# 6. 起服务，API 文档在 http://127.0.0.1:8000/docs
uv run uvicorn ruixue_app.main:app --reload

# 7. 另开一个终端起前端，浏览器开 http://127.0.0.1:3000
cd frontend && npm install && npm run dev
```

入库、灌向量、评测的完整流程见 [docs/操作手册.md](docs/操作手册.md)。

## 文档

- [docs/架构说明.md](docs/架构说明.md) 目录结构和请求路径
- [docs/评测方法.md](docs/评测方法.md) 检索评测方法
- [docs/Agent评测方法.md](docs/Agent评测方法.md) agent 评测方法
- [docs/运维手册.md](docs/运维手册.md) 部署、备份恢复、容量、排查
- [docs/操作手册.md](docs/操作手册.md) 从 PDF 到可检索的脚本流程
- [DECISIONS.md](DECISIONS.md) 技术决策记录
- [docs/数据来源.md](docs/数据来源.md) 领域参数的出处

## 已知问题

- 评测 rf02 稳定失败：用户问题带错误前提时 agent 不会先纠正 [#20](https://github.com/LazySheep-CH/ruixue-agent/issues/20)
- 嵌入模型每个 worker 各载一份，内存是扩容瓶颈 [#21](https://github.com/LazySheep-CH/ruixue-agent/issues/21)

其余见 [issues](https://github.com/LazySheep-CH/ruixue-agent/issues)。

## 后面想做的

- [ ] 嵌入/重排抽成独立服务，worker 不再受内存限制
- [ ] 真实用户 query 日志回流，重建口语化评测集
- [ ] HTTPS（模板已备好，等证书）
- [ ] 飞书接入

## 技术栈

Python 3.11 / PostgreSQL 17 / Milvus 2.6 / SQLAlchemy + Alembic /
sentence-transformers(bge) / jieba / uv / Next.js

## 版权

本项目归江苏省农业科学院农业设施与装备研究所所有，保留所有权利，
仅作展示交流，未经书面许可不得复制、分发或商用。详见 [COPYRIGHT](COPYRIGHT)。
