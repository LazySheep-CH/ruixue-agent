import importlib
from langchain_core.language_models import BaseChatModel
from ruixue_agent.config import load_config


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
            )
            return chatmodel
    raise Exception(f"Model {name} not found, you can choose {[model['name'] for model in config["models"]]}")
