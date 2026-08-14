"""数据集分析工具:让 agent 能分析用户上传的实测数据。

## dataset_id 从哪来 —— 以及为什么这很重要

工具只接收一个 **uuid 形式的 dataset_id**,它由上传接口返回、由前端带进对话。
两个后果:

- **模型编不出别人的 id**(uuid 不可枚举),而且取数时还要再校验一次 user_id;
- 工具**不碰文件系统** —— 数据在 PG 里,没有路径、没有清理、没有配额问题。

user_id 同样不由模型提供:它从 thread_id 前缀解析(和记忆注入同一套),
**模型说自己是谁一律不可信**。
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, tool

from ruixue_agent.analysis import analyze, store

logger = logging.getLogger("ruixue.tools.dataset")


def _load(dataset_id: str):
    """按 id + 当前用户取数据集。取不到统一返回一句人话,不抛。

    user_id 从运行配置里解析(thread_id 形如 "alice:t1"),**不接受模型传参** ——
    否则模型只要说"我是 alice"就能读别人的数据。
    """
    from ruixue_agent.agents.middlewares import _user_id_from

    user_id = _user_id_from(None)
    if not user_id:
        return None, "拿不到你的身份,无法读取数据集。"
    ds = store.get(str(dataset_id).strip(), user_id)
    if ds is None:
        # 不区分"不存在"和"不属于你" —— 区分了就是可枚举的信息泄露
        return None, f"找不到编号为 {dataset_id} 的数据集(可能已删除,或不属于你)。"
    return ds, ""


@tool
def describe_dataset(dataset_id: str) -> str:
    """查看已上传数据集的概览:有哪些列、多少行、各列的范围与缺失情况。

    参数:dataset_id 上传后返回的数据集编号。
    适用:分析任何上传数据的【第一步】——先看清表里有什么、数据质量如何,
    再决定后面怎么分析。用户说"看看我传的数据"也用它。
    """
    ds, err = _load(dataset_id)
    return err or analyze.describe(ds)


@tool
def compare_dataset_with_model(dataset_id: str, target: str = "DR") -> str:
    """把数据集里的【实测值】和【模型预测】逐行对比,给出偏差方向与幅度。

    参数:
      dataset_id 数据集编号;
      target 对比哪个指标:"DR"(降解率)/"TS"(拉伸强度)/"WVTR"(水蒸气透过率)。
    适用:用户问"我实测的和你们模型算的差多少""我这块地是不是不一样"。
    返回里包含平均偏差的【方向】(实测整体高于还是低于预测)——
    方向比幅度更能指向原因。
    """
    ds, err = _load(dataset_id)
    if err:
        return err
    t = str(target or "DR").strip().upper()
    if t not in analyze.TARGET_INFO:
        return f"不支持的指标「{target}」,可选:DR(降解率)、TS(拉伸强度)、WVTR(水蒸气透过率)。"
    return analyze.compare_with_model(ds, t)


@tool
def detect_dataset_outliers(dataset_id: str, z: float = 2.5) -> str:
    """找出数据集里偏离整体过远的行(可能是记录笔误,也可能是真实极端地块)。

    参数:dataset_id 数据集编号;z 判定阈值(几个标准差,默认 2.5)。
    适用:分析结论异常时,先排查数据本身有没有问题;或用户问"有没有记错的"。
    注意:样本少于 5 行时不做判断 —— 小样本的标准差没有意义。
    """
    ds, err = _load(dataset_id)
    if err:
        return err
    try:
        zz = float(z)
    except (TypeError, ValueError):
        zz = 2.5
    return analyze.detect_outliers(ds, max(1.0, zz))


@tool
def check_dataset_against_standard(dataset_id: str) -> str:
    """对数据集里能明确判定的指标做国标符合性检查(目前:厚度)。

    参数:dataset_id 数据集编号。
    适用:用户问"我这批膜合不合国标"。
    注意:只判有明确国标条文的项;降解率、透过率的合格线随产品与作物而变,
    没有统一阈值,那类问题请改用 search_knowledge 查具体条文。
    """
    ds, err = _load(dataset_id)
    return err or analyze.check_standards(ds)


def get_dataset_tools() -> list[BaseTool]:
    return [
        describe_dataset,
        compare_dataset_with_model,
        detect_dataset_outliers,
        check_dataset_against_standard,
    ]
