"""手动测中间件:开 INFO 日志,问个会触发工具的问题,看"工具 xxx 耗时"日志冒出来。

需要 docker 起着(会调 RAG)。
用法:uv run python scripts/test_middleware.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

# 关键:把日志级别降到 INFO,logger.info 才会显示
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from ruixue_agent.agents import create_ruixue_agent

agent = create_ruixue_agent()
agent.invoke(
    {"messages": [{"role": "user", "content": "PBAT地膜的降解机制?"}]},
    config={"configurable": {"thread_id": "mw-test"}},
)
print("\n--- 完成。往上找有没有 'INFO:ruixue.agent:工具 search_knowledge 耗时 xxxms' ---")
