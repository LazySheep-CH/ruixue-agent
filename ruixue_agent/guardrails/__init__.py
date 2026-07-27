"""护栏(guardrails):提示注入防御等安全能力。

对标 deer-flow 的 guardrails/。与 agents/middlewares 的区别:
中间件是"横切关注点的挂载点",护栏是"具体的安全策略实现";
策略放这里,由中间件或工具调用,便于独立测试与迭代。
"""

from ruixue_agent.guardrails.injection import (
    INJECTION_PATTERNS,
    detect_injection,
    wrap_untrusted,
)

__all__ = ["INJECTION_PATTERNS", "detect_injection", "wrap_untrusted"]
