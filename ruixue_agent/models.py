import importlib

from langchain_core.language_models import BaseChatModel

from ruixue_agent.config import load_config

# 调 DeepSeek 多久还没回,就认定它挂了、主动放弃(秒)。
# 不设的话默认是【无限等】—— 对方卡住,你这个请求就永远挂着,占着的资源永远不放。
# 60s 的取法:正常一次调用 3~15s,长回答可能 30s+;60s 已经明显不正常了。
DEFAULT_TIMEOUT_SECONDS = 60


# 根据config自动创建模型对象
def create_model(name: str, **overrides) -> BaseChatModel:
    """按配置创建模型。overrides 覆盖配置项(调用方比配置更清楚自己要什么)。

    目前唯一的真实用途是【评测】:评测要把 temperature 压到 0。
    原因很实在 —— 温度不设时用的是服务端默认(DeepSeek 是 1.0),
    实测同一版本连跑三轮通过率 84.8% / 93.9% / 97.0%,极差 12.1% ≈ 4 道题。
    也就是说这把尺子测不出任何小于 4 道题的改进,拿它做版本对比毫无意义。

    ⚠ 温度 0 【不等于】完全确定:服务端批处理、MoE 路由都会带来残余抖动。
      它只是把最大的那个来源(采样)去掉。
    ⚠ 评测跑 temperature=0,而线上跑配置里的值 —— 两者不一致时,评测结论是
      "指示性"的而非"等同"的。要消除这个差距,应在 config.yaml 里给线上也
      显式钉一个温度(本项目答案要给标准号和数值,低温本来就更合适)。
    """
    config = load_config()
    for model in config["models"]:
        if model["name"] == name:
            module_name, class_name = model["class"].split(":")
            cls = getattr(importlib.import_module(module_name), class_name)
            params = {
                "model": model["model"],
                "base_url": model["base_url"],
                "api_key": model["api_key"],
                # ===== (你写一行)=====
                # 超时:配置里写了就用配置的,没写就用上面的默认值。
                # dict 的 .get(键, 默认值) = "有就取,没有就给我这个默认值"(不会报错):
                #   timeout=model.get("timeout", DEFAULT_TIMEOUT_SECONDS),
                "timeout": model.get("timeout", DEFAULT_TIMEOUT_SECONDS),
                # 关掉 SDK 自带的重试 —— 重试统一交给中间件做,避免【重试叠加】:
                # 若 SDK 重试 2 次、中间件再重试 3 次,最坏情况是 2×3=6 次调用、
                # 等待时间也翻几倍,用户早就等不下去了。责任只放一层。
                "max_retries": 0,
            }
            # 配置里写了 temperature 才传 —— 没写就沿用服务端默认,不擅自改线上行为
            if "temperature" in model:
                params["temperature"] = model["temperature"]
            params.update(overrides)  # 调用方显式指定的优先级最高
            return cls(**params)
    raise Exception(
        f"Model {name} not found, you can choose {[model['name'] for model in config['models']]}"
    )
