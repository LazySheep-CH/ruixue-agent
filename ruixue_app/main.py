"""FastAPI 服务:把 agent 包成 HTTP 接口(POST /chat)。

起服务:uv run uvicorn ruixue_app.main:app --reload
测试:  浏览器开 http://127.0.0.1:8000/docs —— 自动生成的接口文档,能直接点着测。
前置:  docker 起着(agent 要调 RAG)。
"""

from fastapi import FastAPI
from fastapi import Depends
from ruixue_app.auth import get_current_user
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from ruixue_agent.agents import create_ruixue_agent

import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="瑞雪地膜智能助手")

# 启动时建一次 agent,所有请求复用。建 agent 含加载模型,贵 —— 别每次请求都建。
_agent = create_ruixue_agent()


# ============================================================
# (你写 1):请求模型 —— 客户端 POST 过来的 JSON 长什么样。
#   继承 BaseModel,两个字段:
#     thread_id: str   会话ID(决定用哪场对话的记忆)
#     message: str     用户这句话
class ChatRequest(BaseModel):
    thread_id: str
    message: str


# (你写 2):响应模型 —— 你返回给客户端的 JSON 长什么样。
#   继承 BaseModel,一个字段:
#     answer: str      agent 的回答
class ChatResponse(BaseModel):
    answer: str


#(你写 3):POST /chat 端点(函数)。
#   1. 用 @app.post("/chat") 装饰,函数签名:def chat(req: ChatRequest) -> ChatResponse
#   2. 用 req.thread_id 组 config = {"configurable": {"thread_id": req.thread_id}}
#   3. 调 result = _agent.invoke(
#          {"messages": [{"role": "user", "content": req.message}]}, config=config)
#   4. 取 result["messages"][-1].content,包成 ChatResponse(answer=...) 返回
# ============================================================
@app.post("/chat")
def chat(
    req: ChatRequest,
    user_id: str = Depends(get_current_user),      # ← 加：先认证，拿 user_id
) -> ChatResponse:
    thread_id = f"{user_id}:{req.thread_id}"
    config = {"configurable": {"thread_id": thread_id}}
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]}, config=config
    )
    return ChatResponse(answer=result["messages"][-1].content)


# ============================================================
# 流式端点:边生成边返回(SSE)。这里【不用】ChatResponse,因为返回的是一个
# 持续的字节流,不是一次性 JSON。用 StreamingResponse 包一个"产出 SSE 块的生成器"。
#
# (你写):event_generator 的循环体。
#   遍历 _agent.stream(
#           {"messages": [{"role": "user", "content": req.message}]},
#           config=config, stream_mode="messages")  ← "messages" 模式产出逐 token 的消息块
#   每次拿到 (chunk, meta),若 chunk.content 非空,
#   就 yield 一条 SSE:  f"data: {chunk.content}\n\n"   (data: 开头、\n\n 结尾是 SSE 格式)
# ============================================================
@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    user_id: str = Depends(get_current_user),      # ← 加
):
    thread_id = f"{user_id}:{req.thread_id}"
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        for chunk, meta in _agent.stream(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            stream_mode="messages",
        ):
            reasoning = chunk.additional_kwargs.get("reasoning_content")
            if reasoning:
                yield f"data: {json.dumps({'type': 'thinking', 'text': reasoning}, ensure_ascii=False)}\n\n"
            # 答案内容
            if chunk.content:
                yield f"data: {json.dumps({'type': 'answer', 'text': chunk.content}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
