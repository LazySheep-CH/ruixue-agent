"""ruixue_app:服务层(APP)。

对标参考架构的 app/(Gateway)。这里放"把 agent 核心接入外界"的适配器——
目前是 FastAPI(HTTP)。将来的飞书/Slack 网关、CLI 也放这层。

依赖方向单向:ruixue_app ──→ ruixue_agent。
本包可以 import ruixue_agent;ruixue_agent 永远不 import 本包,
也不认识 FastAPI —— 核心保持"传输无关",才能被多种接入方式复用。
"""
