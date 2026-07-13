# 瑞雪智能体（ruixue-agent）

从零、正规搭建的 LangGraph 智能体。配置驱动、支持多模型。
以 deer-flow 为参考蓝图，但每一行核心代码都自己写、自己懂。

## 结构

```
ruixue-agent-mvp/
├── config/config.yaml     # 配置（模型列表等）—— 配置驱动的核心
├── ruixue_agent/          # 核心包（import: ruixue_agent.*）
│   ├── config.py          # 读配置
│   ├── models.py          # 模型工厂（对标 deer-flow factory.py）
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
