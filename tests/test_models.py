import pytest
from ruixue_agent.models import create_model

def test_unknown_model_raises():
    # 给一个配置里不存在的名字，应该抛异常
    with pytest.raises(Exception):
        create_model("这个模型根本不存在")