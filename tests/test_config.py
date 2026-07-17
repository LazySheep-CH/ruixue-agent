from ruixue_agent.config import load_config


def test_env_var_resolved(tmp_path, monkeypatch):
    # Arrange：造一个临时 config + 设一个临时环境变量
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n  - name: test\n    api_key: $TEST_KEY\n", encoding="utf-8"
    )
    monkeypatch.setenv("TEST_KEY", "sk-fake-123")

    # Act：调用被测函数（传入临时路径）
    data = load_config(cfg)

    # Assert：断言 $TEST_KEY 被解析成了真值
    assert data["models"][0]["api_key"] == "sk-fake-123"
