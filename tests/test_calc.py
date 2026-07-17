import pytest
from ruixue_agent.tools.calc import estimate_film_usage


def test_pe_usage():
    # 10亩 × 0.01mm × 系数900 = 90.0 kg
    result = estimate_film_usage.func(area_mu=10, film_type="PE")
    assert "90.0" in result


def test_lowercase_normalized():
    # 小写 pe 也该算对（验证 Bug1 的修复）
    result = estimate_film_usage.func(area_mu=10, film_type="pe")
    assert "90.0" in result


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        estimate_film_usage.func(area_mu=10, film_type="不存在的膜")


def test_negative_area_raises():
    with pytest.raises(ValueError):
        estimate_film_usage.func(area_mu=-5, film_type="PE")
