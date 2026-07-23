# 瑞雪智能体（ruixue-agent）(还没写完，在持续更新)

从零、正规搭建的 LangGraph 智能体。配置驱动、支持多模型。

## 结构

```
ruixue-agent-mvp/
├── config/config.yaml     # 配置（模型列表等）—— 配置驱动的核心
├── ruixue_agent/          # 核心包（import: ruixue_agent.*）
│   ├── config.py          # 读配置
│   ├── models.py          # 模型工厂（对标 参考架构 factory.py）
│   ├── agent.py           # 组装 agent（5 个插槽）
│   ├── tools.py           # 工具
│   └── middlewares.py     # 中间件
├── tests/                 # 测试
└── main.py                # 入口，跑一次
```

## 运行

```bash
uv sync                    # 安装依赖
uv run python main.py      # 跑一次
uv run pytest              # 跑测试
```
