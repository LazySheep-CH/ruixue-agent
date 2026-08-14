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


# ── 类型归一:宽进严出 ──────────────────────────────────────────
#
# 2026-08-08 上课实测:系数表的 key 是 "PE"/"生物降解",但用户和模型
# 不会照着念。"全生物降解"(最自然的说法,docstring 里都写着这四个字)
# 直接报错。评测 mt02 问的就是"全生物降解膜",它过了只是因为模型碰巧
# 归一成了"生物降解" —— 靠运气,不靠设计。


@pytest.mark.parametrize(
    ("raw", "expect_kg"),
    [
        ("PE", 90.0),
        ("pe", 90.0),
        ("聚乙烯地膜", 90.0),
        ("普通地膜", 90.0),
        # ⚠ 含"降解"二字但意思相反 —— 判断顺序反了就会错成 100.0(差 11%)
        ("不可降解地膜", 90.0),
        ("非降解膜", 90.0),
        ("生物降解", 100.0),
        ("全生物降解", 100.0),
        ("生物降解膜", 100.0),
        ("PBAT", 100.0),
        ("PLA/PBAT 共混", 100.0),
    ],
)
def test_film_type_accepts_common_phrasings(raw, expect_kg):
    out = estimate_film_usage.invoke({"area_mu": 10, "film_type": raw})
    assert f"{expect_kg}公斤" in out, f"{raw} 应算作 {expect_kg}kg,实际:{out}"


def test_truly_unknown_film_type_still_raises():
    """宽进不等于什么都认 —— 认不出必须报错,交给中间件降级,
    而不是默默按 PE 算出一个错数字。"""
    with pytest.raises(ValueError, match="暂不支持"):
        estimate_film_usage.invoke({"area_mu": 10, "film_type": "塑料布"})
