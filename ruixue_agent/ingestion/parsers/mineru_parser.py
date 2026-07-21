"""MinerU 解析结果 → Document 的映射层。

MinerU(重依赖,独立环境)预先批量解析产出 *_content_list.json;
本模块只做纯映射:一条 MinerU 元素 → 一个 Element(或丢弃)。
无重依赖,可单测。
"""

from __future__ import annotations

from collections import Counter

from ruixue_agent.ingestion.schema import Document, Element

# 页面家具(页眉/页脚/页码),MinerU 已分类,直接丢弃
_FURNITURE = {"header", "footer", "page_number"}

# 未知类型计数。批处理结束后输出,防止新元素类型被静默丢弃 ——
# 全量语料中出现过 aside_text(72)、code(13) 等小样本中不存在的类型
UNKNOWN_TYPES: Counter[str] = Counter()


def _join(items: list | None) -> str:
    """caption / list_items 等字符串列表拼为一段文本,空列表返回空串。"""
    return "\n".join(items) if items else ""


def _map_element(raw: dict) -> Element | None:
    """一条 MinerU 元素 → Element;页面家具与未知类型返回 None。"""
    mtype = raw["type"]
    page = raw["page_idx"]
    bbox = raw.get("bbox")

    if mtype in _FURNITURE:
        return None

    # text:带 text_level 为标题,否则为正文段落
    if mtype == "text":
        level = raw.get("text_level")
        etype = "heading" if level else "paragraph"
        return Element(type=etype, text=raw["text"], page=page, level=level, bbox=bbox)

    # table:text 存 table_body(HTML)。必须用 .get() 防御 ——
    # 全量 7794 个表格中 251 个(3.2%)缺 table_body(MinerU 识别出表格
    # 区域但未解析出内容,img_path/caption 同为空),按空壳丢弃。
    if mtype == "table":
        body = raw.get("table_body")
        if not body:
            return None
        return Element(
            type="table",
            text=body,
            page=page,
            bbox=bbox,
            meta={
                "caption": _join(raw.get("table_caption")),
                "footnote": _join(raw.get("table_footnote")),
                "img_path": raw.get("img_path"),
            },
        )

    # equation:text 存 LaTeX,同样防御式取值(全量中出现过缺 text 的个例)
    if mtype == "equation":
        latex = raw.get("text")
        if not latex:
            return None
        return Element(
            type="formula",
            text=latex,
            page=page,
            bbox=bbox,
            meta={
                "img_path": raw.get("img_path"),
                "text_format": raw.get("text_format"),
            },
        )

    # image / chart 统一归为 figure,text 存图注
    if mtype in ("image", "chart"):
        caption = raw.get("image_caption") or raw.get("chart_caption")
        return Element(
            type="figure",
            text=_join(caption),
            page=page,
            bbox=bbox,
            meta={"img_path": raw.get("img_path")},
        )

    if mtype == "list":
        return Element(
            type="list",
            text=_join(raw.get("list_items")),
            page=page,
            bbox=bbox,
            meta={"sub_type": raw.get("sub_type")},
        )

    if mtype == "page_footnote":
        return Element(
            type="footnote",
            text=raw["text"],
            page=page,
            bbox=bbox,
        )

    # 未知类型:丢弃但计数,不静默
    UNKNOWN_TYPES[mtype] += 1
    return None


def parse_content_list(content_list: list[dict], meta: dict) -> Document:
    """整份 content_list.json → Document。meta 来自 raw_manifest 台账。"""
    elements = [el for raw in content_list if (el := _map_element(raw)) is not None]
    return Document(
        document_id=meta["document_id"],
        filename=meta["original_filename"],
        file_type="pdf",
        sha256=meta["sha256"],
        source=meta["source"],
        parser="mineru",
        elements=elements,
    )
