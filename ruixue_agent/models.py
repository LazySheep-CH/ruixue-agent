import importlib

from langchain_core.language_models import BaseChatModel

from ruixue_agent.config import load_config

# 调 DeepSeek 多久还没回,就认定它挂了、主动放弃(秒)。
# 不设的话默认是【无限等】—— 对方卡住,你这个请求就永远挂着,占着的资源永远不放。
# 60s 的取法:正常一次调用 3~15s,长回答可能 30s+;60s 已经明显不正常了。
DEFAULT_TIMEOUT_SECONDS = 60


# 根据config自动创建模型对象
def create_model(name: str) -> BaseChatModel:
    config = load_config()
    for model in config["models"]:
        if model["name"] == name:
            module_name, class_name = model["class"].split(":")
            cls = getattr(importlib.import_module(module_name), class_name)
            chatmodel = cls(
                model=model["model"],
                base_url=model["base_url"],
                api_key=model["api_key"],
                # ===== (你写一行)=====
                # 超时:配置里写了就用配置的,没写就用上面的默认值。
                # dict 的 .get(键, 默认值) = "有就取,没有就给我这个默认值"(不会报错):
                #   timeout=model.get("timeout", DEFAULT_TIMEOUT_SECONDS),
                timeout=model.get("timeout", DEFAULT_TIMEOUT_SECONDS),
                # 关掉 SDK 自带的重试 —— 重试统一交给中间件做,避免【重试叠加】:
                # 若 SDK 重试 2 次、中间件再重试 3 次,最坏情况是 2×3=6 次调用、
                # 等待时间也翻几倍,用户早就等不下去了。责任只放一层。
                max_retries=0,
            )
            return chatmodel
    raise Exception(
        f"Model {name} not found, you can choose {[model['name'] for model in config['models']]}"
    )
