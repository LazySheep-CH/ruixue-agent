"""流式测试客户端:请求 /chat/stream,解析 JSON-SSE,区分"思考"和"答案"逐块打印。

用法:
    1. 另开终端起服务:uv run uvicorn ruixue_agent.api:app --reload
    2. 本脚本:        uv run python scripts/test_stream.py
"""

import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

body = json.dumps(
    {"thread_id": "s2", "message": "PBAT地膜的降解机制是什么?"}
).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8000/chat/stream",
    data=body,
    headers={"Content-Type": "application/json"},
)

in_answer = False  # 是否已从"思考"切到"答案"
with urllib.request.urlopen(req) as resp:
    print("💭 思考中:", end=" ", flush=True)
    for raw in resp:  # 逐行读:到一块处理一块,直观看流式
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: "):])
        if event["type"] == "thinking":
            print(event["text"], end="", flush=True)
        else:  # answer
            if not in_answer:
                print("\n\n📝 答案:\n", flush=True)
                in_answer = True
            print(event["text"], end="", flush=True)
print("\n\n[流结束]")
