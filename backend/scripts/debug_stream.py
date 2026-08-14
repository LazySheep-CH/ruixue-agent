"""调试:跑完整个 stream,统计 content / reasoning,看答案内容到底来不来。
uv run python scripts/debug_stream.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from ruixue_agent.agents import create_ruixue_agent

agent = create_ruixue_agent()
config = {"configurable": {"thread_id": "dbg2"}}

total = content_chunks = reasoning_chunks = 0
answer = ""
for chunk, meta in agent.stream(
    {"messages": [{"role": "user", "content": "PBAT地膜的降解机制是什么?"}]},
    config=config,
    stream_mode="messages",
):
    total += 1
    if chunk.content:
        content_chunks += 1
        answer += chunk.content
    if chunk.additional_kwargs.get("reasoning_content"):
        reasoning_chunks += 1

print(f"\n总块={total}  有content的块={content_chunks}  有reasoning的块={reasoning_chunks}")
print(f"\n拼出的答案(前 300 字):\n{answer[:300]}")
