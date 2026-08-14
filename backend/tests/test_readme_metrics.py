"""README 的指标必须与最近一次标准评测跑出的基线一致。

【这个测试为什么存在】
2026-08-11 发现 README 的检索指标已经错了很久:
    README 写 R@1 0.807 / R@10 0.953 / MRR 0.866
    实测(338 题多标注)0.769 / 0.979 / 0.849
错因不是有人瞎写,而是**评测集从 150 题扩到 338 题、改成多标注**之后,
指标自然变了,但 README 是手写的、没人记得跟着改。更糟的是 0.953 其实是
R@5(0.952)被抄到了 R@10 那一栏——**错位抄写没有任何机制能发现**。

这类 bug 的特点是:代码全对、测试全绿、系统跑得好好的,**只有对外说的话是假的**。
简历、汇报、面试全部引用 README,于是一个人的笔误变成了对外的不实陈述。

所以这里用测试把"文档"也纳入 CI:数字是一种会腐烂的资产,得有人盯着。

【为什么盯 README 而不是让脚本改 README】
见 scripts/run_eval.py::_save_baseline 里的说明 —— 简单说是"机器不该替人
决定哪个数才算数"。测试红的时候有两种可能:文档过期了,或者这次跑法不标准。
这个判断必须由人做。

【没有基线文件时怎么办】
skip,不 fail。基线要跑真实评测才有(需要 PG + Milvus + 26 万向量),
CI 和新克隆的仓库里不该因为"没跑过评测"就红。
它防的是**跑过之后忘记同步文档**,不是强制每个人都跑评测。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
# README 在【仓库根】(项目门面),基线数据在 backend/data 下 —— 两个锚不同
README = BACKEND.parent / "README.md"
BASELINE = BACKEND / "data" / "eval" / "baselines" / "retrieval_latest.json"

# 上线管线在基线 json 里的层名(与 run_eval.py 的 configs 标签一致)
ONLINE_LAYER = "+BM25+rerank"
VECTOR_ONLY = "纯向量"
HYBRID = "+BM25(混合)"


def _load_baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip(
            f"没有评测基线({BASELINE.relative_to(BACKEND)})。"
            "跑 `uv run python scripts/run_eval.py --ab` 生成后本测试才生效。"
        )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _readme_numbers() -> set[str]:
    """README 里出现的所有三位小数,如 '0.769'。

    只做【存在性】检查而不解析表格结构:表格排版随时会改(加一列、换措辞),
    解析结构的测试会因为无关改动天天红,那样的测试很快就会被人加 skip 绕过。
    存在性检查弱一些,但它精确命中我们真正要防的那件事——
    **README 里出现了一个基线里根本没有的数**。
    """
    return set(re.findall(r"\d\.\d{3}", README.read_text(encoding="utf-8")))


def test_readme_reports_online_pipeline_metrics():
    """README 必须写着上线管线真实跑出的 R@1 / R@10 / MRR。"""
    online = _load_baseline()["layers"][ONLINE_LAYER]
    found = _readme_numbers()

    missing = {
        name: f"{online[key]:.3f}"
        for name, key in (("R@1", "recall@1"), ("R@10", "recall@10"), ("MRR", "mrr"))
        if f"{online[key]:.3f}" not in found
    }
    assert not missing, (
        f"README.md 的检索指标与最近一次评测对不上,缺失:{missing}\n"
        f"基线文件:{BASELINE.relative_to(BACKEND)}\n"
        "要么更新 README,要么这次评测跑法不标准——请人工判断,不要直接改这个测试。"
    )


def test_readme_ablation_chain_matches_baseline():
    """逐层增益链(纯向量 → 混合 → 重排)也要对得上。

    单独一条 case:这三个数最容易【只改了最后一个】。
    2026-08-11 那次就是三个数全错,但如果只测最终值就只能发现一个。
    """
    layers = _load_baseline()["layers"]
    found = _readme_numbers()

    missing = {
        tag: f"{layers[tag]['recall@1']:.3f}"
        for tag in (VECTOR_ONLY, HYBRID, ONLINE_LAYER)
        if f"{layers[tag]['recall@1']:.3f}" not in found
    }
    assert not missing, (
        f"README.md 的逐层增益链与基线对不上,缺失:{missing}\n逐层数字最容易只改末位、漏掉前两级。"
    )


def test_readme_evalset_size_matches_file():
    """README 声称的评测集题数,必须等于文件真实行数。

    这条防的是另一半:**评测集变大了但文档没跟上**。
    README 曾长期写"150 题",而 data/eval/evalset.jsonl 早已是 338 行——
    上面那些指标之所以全错,根源就是这个。
    """
    evalset = BACKEND / "data" / "eval" / "evalset.jsonl"
    if not evalset.exists():
        pytest.skip("评测集未随仓库分发")

    n = sum(1 for line in evalset.read_text(encoding="utf-8").splitlines() if line.strip())
    text = README.read_text(encoding="utf-8")
    assert f"{n} 题评测集" in text or f"{n} 题" in text, (
        f"README.md 没有反映评测集的真实规模({n} 题)。"
        "评测集增删后必须同步文档,否则指标口径无从对照。"
    )
