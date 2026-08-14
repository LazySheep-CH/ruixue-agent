"""上传数据集的分析能力:契约校验、加载、与模型预测对比。

主入口:
    schema.validate / map_columns   —— 列名归一与数据契约校验
    loader.load_csv / summarize     —— 字节流 → 结构化数据 + 概览
"""

from ruixue_agent.analysis.loader import load_csv, parse_number, summarize
from ruixue_agent.analysis.schema import ColumnMap, DatasetError, map_columns, validate

__all__ = [
    "ColumnMap",
    "DatasetError",
    "load_csv",
    "map_columns",
    "parse_number",
    "summarize",
    "validate",
]
