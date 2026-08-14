"""上传文件 → 校验过的结构化数据。只做解析和校验,不碰数据库。

## 只认 CSV,不认 Excel

Excel 要多一个依赖(openpyxl),而且 `.xlsx` 里可以嵌公式、宏、外部链接 ——
解析器的攻击面比纯文本大一个量级。田间数据表从 Excel 另存为 CSV 是一步操作,
不值得为这一步便利引入那个面。**能用受限格式做到的,不开放通用格式。**

## 编码:必须自己判,不能假定 UTF-8

用户的表大多是从 Excel 导出的,Windows 上默认是 GBK/GB18030。
直接按 UTF-8 解会抛 UnicodeDecodeError,而错误信息("invalid start byte")
对用户毫无意义。按顺序试几种编码,全失败才报一句人话。

BOM 也要处理:Excel 导出的 UTF-8 CSV 常带 BOM,不剥掉的话第一个列名会变成
"﻿PLA含量",列名归一直接认不出来 —— 而且**不报错**,只是少认一列。

## 数值解析:宽进严出

田间记录里的数字长这样:`"12.5"`、`"12.5%"`、`"1,234"`、`"约 30"`、`""`、`"—"`。
硬转 float 会整行报废。这里尽量抠出数字,抠不出记为 None(缺失),
由下游用中位数兜底 —— 和 predictors 的兜底逻辑一致。

**但不猜**:`"高"`、`"偏酸"` 这种描述性文字不会被硬编成数字,它们是真缺失。
"""

from __future__ import annotations

import csv
import io
import re

from ruixue_agent.analysis.schema import ColumnMap, DatasetError, validate

# 上传体积上限。5000 行 × 80 列的纯数字 CSV 约 2MB,留一倍余量。
# 这道闸门在【读进内存之前】就该拦 —— 见 app 层的上传端点。
MAX_BYTES = 4 * 1024 * 1024

# 按可能性排序。gb18030 是 gbk 的超集,放在 gbk 之后能兜住生僻字。
_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030")

# 抠数字:允许前后有中文/空格/单位,允许千分位和百分号
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# 明确表示"没有"的写法。不列进来的话,"—" 会被当成描述性文字,效果一样,
# 但列出来能让意图更清楚。
_NULLISH = {"", "-", "—", "–", "na", "n/a", "null", "none", "nan", "无", "未测", "缺失"}


def decode(raw: bytes) -> str:
    """按几种常见编码解码。全失败抛 DatasetError(消息面向用户)。"""
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DatasetError("文件编码无法识别。请在 Excel 里用「另存为 → CSV UTF-8」重新导出后上传。")


def parse_number(cell: str) -> float | None:
    """把单元格抠成数字;抠不出返回 None(= 缺失,由下游兜底)。

    认:`12.5` `12.5%` `1,234` `约 30` ` 8.2 ℃`
    不认:`高` `偏酸` `见备注` —— 这些是真缺失,**不猜**。
      猜错一个数字,整行的预测就跟着错,而且不会报错。
    """
    s = str(cell).strip()
    if s.lower() in _NULLISH:
        return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def load_csv(raw: bytes) -> tuple[ColumnMap, list[dict]]:
    """解析上传的 CSV,返回 (列映射, 数据行)。不合契约抛 DatasetError。

    数据行的键用【标准特征名】(如 PLA_pct),值为 float 或 None ——
    归一只做一次,做在入口。下游拿到的结构是确定的,不必再猜列名。
    """
    if not raw:
        raise DatasetError("文件是空的。")
    if len(raw) > MAX_BYTES:
        raise DatasetError(
            f"文件 {len(raw) / 1024 / 1024:.1f}MB 超过上限 {MAX_BYTES // 1024 // 1024}MB。"
            "请只保留需要分析的列和行。"
        )

    text = decode(raw)
    # 用 Sniffer 猜分隔符:国内导出的 CSV 有用逗号也有用分号/制表符的。
    # 猜不出就按逗号 —— 猜错的后果是"整张表只有一列",下面的列头校验会拦住。
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(text), dialect)
    try:
        headers = [h.strip() for h in next(reader)]
    except StopIteration:
        raise DatasetError("文件是空的。") from None

    raw_rows = [r for r in reader if any(str(c).strip() for c in r)]
    cm = validate(headers, len(raw_rows))

    # 原列名 → 标准名。目标列用 "target:DR" 这样的键,和特征区分开。
    to_std: dict[str, str] = {orig: std for std, orig in cm.features.items()}
    to_std.update({orig: f"target:{model}" for model, orig in cm.targets.items()})

    rows: list[dict] = []
    for r in raw_rows:
        row: dict[str, float | None] = {}
        for i, h in enumerate(headers):
            key = to_std.get(h)
            if key is None:
                continue  # 没认出来的列直接不进数据行 —— 它们在 cm.unknown 里有记录
            row[key] = parse_number(r[i]) if i < len(r) else None
        rows.append(row)
    return cm, rows


def summarize(cm: ColumnMap, rows: list[dict]) -> dict:
    """给上传响应用的概览:认出了什么、缺多少、每个数值列的范围。

    **缺失率要单独报**:一列 90% 是空的,分析时它基本无效 ——
    用户应该在上传那一刻就知道,而不是等结论出来才发现是拿默认值算的。
    """
    stats: dict[str, dict] = {}
    for key in {k for r in rows for k in r}:
        vals = [r[key] for r in rows if r.get(key) is not None]
        stats[key] = {
            "n": len(vals),
            "missing_pct": round(100 * (1 - len(vals) / len(rows)), 1) if rows else 100.0,
            "min": round(min(vals), 4) if vals else None,
            "max": round(max(vals), 4) if vals else None,
            "mean": round(sum(vals) / len(vals), 4) if vals else None,
        }
    return {
        "n_rows": len(rows),
        "features": sorted(cm.features),
        "targets": sorted(cm.targets),
        "unrecognized_columns": list(cm.unknown),
        "stats": stats,
    }
