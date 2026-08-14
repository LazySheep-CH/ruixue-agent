"""数据集的存取。归属校验是这一层的核心职责,不是可选项。

每个读取入口都必须带 user_id:
和 `runs.get_run` 同一条规矩:猜到 dataset_id 就能看别人的数据 = 数据泄露。
所以这里不提供"按 id 取"的裸接口 —— 只有 `get(dataset_id, user_id)`。
少一个参数的便利,换来的是一个随时可能被误用的越权入口。

dataset_id 用 uuid,不用自增:
自增 id 可枚举:知道自己的是 7,就能去试 8。uuid 猜不出来。
这一条和 `runs` 一致 —— 对外暴露的标识符一律不可枚举。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ruixue_agent.analysis.schema import ColumnMap
from ruixue_agent.persistence.engine import get_engine
from ruixue_agent.persistence.models import DatasetRow

# 每个用户最多留多少份。超了删最旧的 ——
# 不做的话数据只增不减,而这类数据的价值随时间快速衰减(去年的田间记录很少再看)。
MAX_PER_USER = 20


@dataclass(frozen=True)
class Dataset:
    """取出来的数据集。和 ORM 解耦,便于测试与序列化。"""

    dataset_id: str
    user_id: str
    filename: str
    n_rows: int
    n_cols: int
    columns: dict
    rows: list[dict]
    created_at: datetime

    @property
    def features(self) -> dict[str, str]:
        return self.columns.get("features") or {}

    @property
    def targets(self) -> dict[str, str]:
        return self.columns.get("targets") or {}


def _to_dataset(r: DatasetRow) -> Dataset:
    return Dataset(
        dataset_id=r.dataset_id,
        user_id=r.user_id,
        filename=r.filename,
        n_rows=r.n_rows,
        n_cols=r.n_cols,
        columns=r.columns or {},
        rows=r.rows or [],
        created_at=r.created_at,
    )


def save(user_id: str, filename: str, cm: ColumnMap, rows: list[dict]) -> str:
    """存一份数据集,返回 dataset_id。顺带按用户做数量上限清理。"""
    ds_id = str(uuid.uuid4())
    with Session(get_engine()) as s:
        s.add(
            DatasetRow(
                dataset_id=ds_id,
                user_id=user_id,
                # 只作展示。不用它拼路径(不落盘)、也不进 HTTP 头 ——
                # 和 report.filename_for 同一条:不可信输入不进这两个地方。
                filename=filename[:255],
                n_rows=len(rows),
                n_cols=len(cm.features) + len(cm.targets),
                columns={
                    "features": cm.features,
                    "targets": cm.targets,
                    "unknown": list(cm.unknown),
                },
                rows=rows,
            )
        )
        s.commit()
        _prune(s, user_id)
    return ds_id


def _prune(s: Session, user_id: str) -> int:
    """只保留该用户最近 MAX_PER_USER 份,多的删掉。返回删了几份。

    在写入之后立即做,而不是靠定时任务:定时任务是个需要单独部署和监控的
    活动件,而这里的清理量极小(每次最多删几行),顺手做掉最省事。
    """
    keep = [
        r[0]
        for r in s.execute(
            select(DatasetRow.id)
            .where(DatasetRow.user_id == user_id)
            .order_by(DatasetRow.created_at.desc())
            .limit(MAX_PER_USER)
        )
    ]
    if len(keep) < MAX_PER_USER:
        return 0
    res = s.execute(
        delete(DatasetRow).where(DatasetRow.user_id == user_id, DatasetRow.id.notin_(keep))
    )
    s.commit()
    return res.rowcount or 0


def get(dataset_id: str, user_id: str) -> Dataset | None:
    """取一份数据集。user_id 不匹配就当作不存在 —— 返回 None 而不是抛权限错误。

    返回 404 而非 403 是刻意的:403 等于告诉对方"这个 id 存在",
    是一个可枚举的信息泄露。和 runs.get_run 保持一致。
    """
    with Session(get_engine()) as s:
        row = s.scalar(
            select(DatasetRow).where(
                DatasetRow.dataset_id == dataset_id, DatasetRow.user_id == user_id
            )
        )
        return _to_dataset(row) if row else None


def list_for(user_id: str, limit: int = 20) -> list[Dataset]:
    """列出该用户的数据集(不含 rows —— 列表页不需要,拉出来纯属浪费)。"""
    with Session(get_engine()) as s:
        rows = s.scalars(
            select(DatasetRow)
            .where(DatasetRow.user_id == user_id)
            .order_by(DatasetRow.created_at.desc())
            .limit(limit)
        ).all()
        return [
            Dataset(
                dataset_id=r.dataset_id,
                user_id=r.user_id,
                filename=r.filename,
                n_rows=r.n_rows,
                n_cols=r.n_cols,
                columns=r.columns or {},
                rows=[],  # 刻意留空
                created_at=r.created_at,
            )
            for r in rows
        ]


def delete_one(dataset_id: str, user_id: str) -> bool:
    """删除一份;不属于该用户则返回 False(不报错、不透露存在性)。"""
    with Session(get_engine()) as s:
        res = s.execute(
            delete(DatasetRow).where(
                DatasetRow.dataset_id == dataset_id, DatasetRow.user_id == user_id
            )
        )
        s.commit()
        return bool(res.rowcount)


def count_for(user_id: str) -> int:
    with Session(get_engine()) as s:
        return (
            s.scalar(
                select(func.count()).select_from(DatasetRow).where(DatasetRow.user_id == user_id)
            )
            or 0
        )
