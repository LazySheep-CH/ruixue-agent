"""指标与告警的测试。

监控这层有个特有的失败模式:**它自己变成故障源**。
数据库抖一下 /metrics 返回 500,监控系统就认为服务挂了 —— 而服务好好的。
所以这里盯三件事:格式合规、部分失败不整体崩、以及**不许发假警报**。

假警报的真正代价不是吵醒人,是让所有人开始忽略警报 —— 那时真出事也没人看。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_alerts import Rule, evaluate, parse_metrics  # noqa: E402

from ruixue_app import metrics  # noqa: E402

# ── Prometheus 格式合规:只有接上采集器才会暴露的那类错误 ──────────


def _rendered(monkeypatch) -> str:
    """渲染一份指标,依赖探测全部打桩 —— 测试不该依赖外部服务是否在跑。"""
    monkeypatch.setattr(metrics, "_runs_window", lambda: {"succeeded": 9, "failed": 1})
    monkeypatch.setattr(metrics, "_table_bytes", lambda: {"checkpoints": 1024, "runs": 2048})
    monkeypatch.setattr(metrics, "_probe_pg", lambda: True)
    monkeypatch.setattr(metrics, "_probe_redis", lambda: True)
    monkeypatch.setattr(metrics, "_probe_milvus", lambda: True)
    return metrics.render()


def test_help_and_type_never_carry_labels(monkeypatch):
    """`# HELP name{label="x"}` 是非法的 —— 解析器会拒绝。

    这类错误肉眼读输出完全正常,**只有真接上采集器才会暴露**。
    """
    for line in _rendered(monkeypatch).splitlines():
        if line.startswith("#"):
            assert "{" not in line, f"HELP/TYPE 带了标签:{line}"


def test_each_metric_name_is_declared_only_once(monkeypatch):
    """同一个指标名的 HELP/TYPE 只能声明一次,多个样本靠标签区分。"""
    names = re.findall(r"^# HELP (\S+)", _rendered(monkeypatch), re.M)
    assert len(names) == len(set(names)), f"重复声明:{[n for n in names if names.count(n) > 1]}"


def test_multi_sample_metrics_share_one_declaration(monkeypatch):
    text = _rendered(monkeypatch)
    assert text.count("# TYPE ruixue_dep_up ") == 1
    for dep in ("postgres", "redis", "milvus"):
        assert f'ruixue_dep_up{{dep="{dep}"}}' in text


# ── 部分失败不能让整个端点挂掉 ──────────────────────────────────


def test_db_failure_degrades_instead_of_500(monkeypatch):
    """监控端点自己变成故障点很讽刺:DB 抖一下就返回 500,
    监控系统于是认为服务挂了 —— 而服务好好的。"""

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(metrics, "_runs_window", boom)
    monkeypatch.setattr(metrics, "_table_bytes", boom)
    monkeypatch.setattr(metrics, "_probe_pg", lambda: False)
    monkeypatch.setattr(metrics, "_probe_redis", lambda: True)
    monkeypatch.setattr(metrics, "_probe_milvus", lambda: True)

    text = metrics.render()
    assert "ruixue_up 1" in text, "进程还活着就该报 1"
    assert 'ruixue_dep_up{dep="postgres"} 0' in text
    m = parse_metrics(text)
    assert m["ruixue_metrics_errors"] >= 2, "取数失败要计数,否则没人知道指标不全"


# ── 解析 ──────────────────────────────────────────────────────


def test_parser_skips_comments_and_keeps_labels():
    m = parse_metrics(
        '# HELP x y\n# TYPE x gauge\nx 1\nruixue_dep_up{dep="redis"} 0\nbad line here\n'
    )
    assert m == {"x": 1.0, 'ruixue_dep_up{dep="redis"}': 0.0}


# ── 告警规则:不许发假警报 ──────────────────────────────────────


def test_no_alert_when_everything_is_fine():
    ok = {
        'ruixue_dep_up{dep="postgres"}': 1,
        'ruixue_dep_up{dep="redis"}': 1,
        'ruixue_dep_up{dep="milvus"}': 1,
        "ruixue_runs_total_1h": 50,
        "ruixue_runs_failure_ratio_1h": 0.02,
        "ruixue_inflight_runs": 2,
        "ruixue_capacity_runs": 24,
    }
    assert evaluate(ok) == []


@pytest.mark.parametrize("total,ratio", [(0, 0.0), (1, 1.0), (4, 1.0)])
def test_tiny_traffic_never_triggers_failure_alert(total, ratio):
    """**这条是防假警报的核心。**

    凌晨没人用,1 次运行失败 1 次 = 100% 失败率。按比例报警就会每天半夜
    响一次假警 —— 而假警报的代价是让人开始忽略所有警报。
    """
    fired = evaluate({"ruixue_runs_total_1h": total, "ruixue_runs_failure_ratio_1h": ratio})
    assert not [r for r, _ in fired if "failure" in r.name]


def test_high_failure_ratio_with_real_traffic_is_critical():
    fired = evaluate({"ruixue_runs_total_1h": 20, "ruixue_runs_failure_ratio_1h": 0.4})
    names = {r.name: r for r, _ in fired}
    assert names["high_failure_ratio"].level == "critical"


def test_dependency_down_is_critical_and_says_what_breaks():
    """告警必须说清"该做什么"——只报"milvus 挂了"的话,值班的人第一反应是"然后呢"。"""
    fired = evaluate({'ruixue_dep_up{dep="milvus"}': 0})
    r, _ = fired[0]
    assert r.level == "critical"
    assert "docker ps" in r.action, "要给出第一步该做什么"
    assert "记忆召回" in r.action or "知识问答" in r.action, "要说清挂了会影响什么"


def test_missing_dependency_metric_is_not_treated_as_down():
    """指标里没有这一项时不能默认当成"挂了" —— 那是把"没测到"说成"坏了"。"""
    assert not [r for r, _ in evaluate({}) if r.name.startswith("dep_down")]


def test_every_rule_carries_an_action():
    """所有规则都必须带 action。没有 action 的告警等于噪声。"""
    bad = {
        'ruixue_dep_up{dep="postgres"}': 0,
        "ruixue_runs_total_1h": 20,
        "ruixue_runs_failure_ratio_1h": 0.9,
        "ruixue_inflight_runs": 23,
        "ruixue_capacity_runs": 24,
        "ruixue_runs_running": 3,
        "ruixue_metrics_errors": 1,
        'ruixue_table_bytes{table="checkpoints"}': 9 * 1024**3,
    }
    fired = evaluate(bad)
    assert len(fired) >= 5
    for r, _ in fired:
        assert isinstance(r, Rule) and r.action.strip(), f"规则 {r.name} 没写 action"
